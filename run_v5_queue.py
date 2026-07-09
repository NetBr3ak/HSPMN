"""Robust v5 experiment queue - overnight/multi-day GPU orchestrator.

Fixes the orchestrate_v5_chain.sh anti-pattern: every job's success is decided
by whether its ARTIFACT EXISTS (a checkpoint / output file), never by the
process exit status. Jobs are skipped if already done (resumable), run
sequentially (single GPU), and a failure logs FAILED and moves on instead of
poisoning the chain. Optionally waits for a currently-running job (the dense
relaunch) to finish first.

Usage:
    nohup python3 run_v5_queue.py --wait-pid 1336839 > results/v5_queue.out 2>&1 &
    python3 run_v5_queue.py --list          # show the queue and done/pending state
    python3 run_v5_queue.py --only finalize_dense niah_p4a   # run a subset
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/opt/docker/LLM/HSPMN")
LOG = ROOT / "results" / "v5_queue.log"

P4A = "checkpoints_p4_350m/hymba-with-nsa_p4_final.pt"

# Shared 350M recipes (matched for fair Δ; seed varies per job).
HYMBA = ("--variant hymba-with-nsa --n_layers 24 --dim 896 --num_heads 14 "
         "--num_kv_heads 2 --seq_len 1024 --batch_size 8 --grad_accum 8 "
         "--steps 15000 --warmup_steps 500 --lr 1e-3 --kill_clock_h 10.0")
DENSE = ("--variant dense --n_layers 24 --dim 1024 --num_heads 16 "
         "--num_kv_heads 4 --seq_len 1024 --batch_size 8 --grad_accum 8 "
         "--steps 15000 --warmup_steps 500 --lr 1e-3 --weight_decay 0.1 "
         "--kill_clock_h 10.0")
GATED = HYMBA.replace("--variant hymba-with-nsa ", "--variant hymba-with-nsa-gated ")

GATE_MEAS = ("--n_batches 4 --batch 2 --seq_len 512 --n_layers 12 --dim 768 "
             "--num_heads 12 --num_kv_heads 4")


def job(name, cmd, artifact, critical=False):
    return {"name": name, "cmd": cmd, "artifact": artifact, "critical": critical}


# Ordered: cheap/high-value first, then the long 3-seed 350M runs, then scale tests.
QUEUE = [
    job("finalize_dense",
        "python3 finalize_v5_after_dense.py",
        "paper/HSPMN_v5_final_numbers.tex", critical=True),
    job("niah_p4a",
        f"python3 niah_v5_long_context.py --variant hymba-with-nsa --ckpt {P4A} "
        f"--contexts 1024 2048 4096 --out_md results/niah_v5_p4a.md",
        "results/niah_v5_p4a.md"),
    job("hymba_350m_s1337",
        f"python3 train_p4_350m.py {HYMBA} --seed 1337 --save_dir checkpoints_p4_350m_s1337",
        "checkpoints_p4_350m_s1337/hymba-with-nsa_p4_final.pt", critical=True),
    job("dense_350m_s1337",
        f"python3 train_p4_350m.py {DENSE} --seed 1337 --save_dir checkpoints_p4b_dense_s1337",
        "checkpoints_p4b_dense_s1337/dense_p4_final.pt", critical=True),
    job("hymba_350m_s2026",
        f"python3 train_p4_350m.py {HYMBA} --seed 2026 --save_dir checkpoints_p4_350m_s2026",
        "checkpoints_p4_350m_s2026/hymba-with-nsa_p4_final.pt", critical=True),
    job("dense_350m_s2026",
        f"python3 train_p4_350m.py {DENSE} --seed 2026 --save_dir checkpoints_p4b_dense_s2026",
        "checkpoints_p4b_dense_s2026/dense_p4_final.pt", critical=True),
    job("gated_350m_s42",
        f"python3 train_p4_350m.py {GATED} --seed 42 --save_dir checkpoints_p4_gated_s42",
        "checkpoints_p4_gated_s42/hymba-with-nsa-gated_p4_final.pt"),
    job("mi_gated_350m",
        "python3 measure_gate_channel_mi.py --variant hymba-with-nsa-gated "
        "--ckpt checkpoints_p4_gated_s42/hymba-with-nsa-gated_p4_final.pt "
        f"--label_ckpt {P4A} "  # non-circular: label from the gate-free 350M base
        "--n_batches 4 --batch 2 --seq_len 512 --n_layers 24 --dim 896 "
        "--num_heads 14 --num_kv_heads 2 "
        "--out_md results/gate_channel_mi_350m_gated.md",
        "results/gate_channel_mi_350m_gated.md"),
]


def stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg):
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def done(j):
    return (ROOT / j["artifact"]).exists()


def run_job(j):
    if done(j):
        log(f"SKIP {j['name']} - artifact exists ({j['artifact']})")
        return True
    log(f"START {j['name']}: {j['cmd']}")
    jlog = ROOT / "results" / f"v5_queue_{j['name']}.log"
    with open(jlog, "w") as f:
        rc = subprocess.call(j["cmd"], shell=True, cwd=ROOT, stdout=f,
                             stderr=subprocess.STDOUT)
    ok = done(j)
    log(f"END   {j['name']}: rc={rc} artifact_exists={ok} "
        f"({'OK' if ok else 'FAILED - artifact missing'}); log={jlog.name}")
    return ok


def wait_for_pid(pid):
    log(f"waiting for PID {pid} to exit before starting queue")
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(120)
    log(f"PID {pid} exited; starting queue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=None,
                    help="Block until this PID exits (the running dense job).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these job names.")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        total = 0
        for j in QUEUE:
            state = "DONE" if done(j) else "pending"
            crit = " [critical]" if j["critical"] else ""
            print(f"  {state:8s} {j['name']}{crit}")
            total += 0 if done(j) else 1
        print(f"\n{total} pending of {len(QUEUE)}")
        return

    jobs = QUEUE if not args.only else [j for j in QUEUE if j["name"] in args.only]
    log(f"=== v5 queue start: {len(jobs)} jobs ===")
    if args.wait_pid:
        wait_for_pid(args.wait_pid)

    results = {}
    for j in jobs:
        try:
            results[j["name"]] = run_job(j)
        except Exception as e:  # never let one job kill the queue
            log(f"ERROR {j['name']}: {e!r}")
            results[j["name"]] = False

    n_ok = sum(results.values())
    log(f"=== v5 queue done: {n_ok}/{len(jobs)} produced artifacts ===")
    for name, ok in results.items():
        log(f"    {'OK  ' if ok else 'MISS'} {name}")


if __name__ == "__main__":
    main()
