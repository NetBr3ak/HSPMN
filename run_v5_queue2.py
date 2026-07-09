"""v5 experiment queue #2 - the central thesis test at scale.

After the queue-1 surprise (gated 350M seed42 PPL 78.25 beat base hymba 87.38,
dense 83.75, while base hymba ~= dense at 350M), the open question is whether the
LEARNED GATE beats the Aquino-Michaels RANDOM GATE at 350M with 3-seed CIs --
the absorption-resistance thesis at scale. queue-1 left us with gated s42 only
and no randgate-350M control. This queue fills both.

Same robust contract as its predecessor queue (verify-by-artifact,
skip-if-done, continue-on-fail). GPU is free; launch immediately (no
--wait-pid).

    nohup python3 run_v5_queue2.py > results/v5_queue2.out 2>&1 &
    python3 run_v5_queue2.py --list
"""

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "results" / "v5_queue2.log"
P4A = "checkpoints_p4_350m/hymba-with-nsa_p4_final.pt"

BASE = (
    "--n_layers 24 --dim 896 --num_heads 14 --num_kv_heads 2 --seq_len 1024 "
    "--batch_size 8 --grad_accum 8 --steps 15000 --warmup_steps 500 --lr 1e-3 "
    "--kill_clock_h 10.0"
)
MIARGS = (
    "--n_batches 4 --batch 2 --seq_len 512 --n_layers 24 --dim 896 "
    "--num_heads 14 --num_kv_heads 2"
)


def train(variant, seed, save):
    return (
        f"python3 train_p4_350m.py --variant {variant} {BASE} --seed {seed} "
        f"--save_dir {save}",
        f"{save}/{variant}_p4_final.pt",
    )


def mi(variant, ckpt, out):
    return (
        f"python3 measure_gate_channel_mi.py --variant {variant} --ckpt {ckpt} "
        f"--label_ckpt {P4A} {MIARGS} --out_md {out}",
        out,
    )


def job(name, cmd, art, crit=False):
    return {"name": name, "cmd": cmd, "artifact": art, "critical": crit}


# Interleave so gated-vs-randgate becomes comparable as early as possible.
_jobs = []
_c, _a = train("hymba-with-nsa-randgate", 42, "checkpoints_p4_randgate_s42")
_jobs.append(job("randgate_350m_s42", _c, _a, True))
_c, _a = train("hymba-with-nsa-gated", 1337, "checkpoints_p4_gated_s1337")
_jobs.append(job("gated_350m_s1337", _c, _a, True))
_c, _a = train("hymba-with-nsa-randgate", 1337, "checkpoints_p4_randgate_s1337")
_jobs.append(job("randgate_350m_s1337", _c, _a, True))
_c, _a = train("hymba-with-nsa-gated", 2026, "checkpoints_p4_gated_s2026")
_jobs.append(job("gated_350m_s2026", _c, _a, True))
_c, _a = train("hymba-with-nsa-randgate", 2026, "checkpoints_p4_randgate_s2026")
_jobs.append(job("randgate_350m_s2026", _c, _a, True))
# Non-circular MI at scale for each fresh model (cheap; label = gate-free p4a base).
_c, _a = mi(
    "hymba-with-nsa-randgate",
    "checkpoints_p4_randgate_s42/hymba-with-nsa-randgate_p4_final.pt",
    "results/gate_channel_mi_350m_randgate.md",
)
_jobs.append(job("mi_randgate_350m", _c, _a))
_c, _a = mi(
    "hymba-with-nsa-gated",
    "checkpoints_p4_gated_s1337/hymba-with-nsa-gated_p4_final.pt",
    "results/gate_channel_mi_350m_gated_s1337.md",
)
_jobs.append(job("mi_gated_350m_s1337", _c, _a))

QUEUE = _jobs


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
        log(f"SKIP {j['name']} - {j['artifact']} exists")
        return True
    log(f"START {j['name']}: {j['cmd']}")
    jlog = ROOT / "results" / f"v5_queue2_{j['name']}.log"
    with open(jlog, "w") as f:
        rc = subprocess.call(
            j["cmd"], shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
        )
    ok = done(j)
    log(
        f"END   {j['name']}: rc={rc} artifact={'OK' if ok else 'MISSING'}; log={jlog.name}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    if args.list:
        for j in QUEUE:
            print(
                f"  {'DONE' if done(j) else 'pending':8s} {j['name']}"
                f"{' [critical]' if j['critical'] else ''}"
            )
        print(f"\n{sum(1 for j in QUEUE if not done(j))} pending of {len(QUEUE)}")
        return
    jobs = QUEUE if not args.only else [j for j in QUEUE if j["name"] in args.only]
    log(f"=== v5 queue2 start: {len(jobs)} jobs ===")
    res = {}
    for j in jobs:
        try:
            res[j["name"]] = run_job(j)
        except Exception as e:
            log(f"ERROR {j['name']}: {e!r}")
            res[j["name"]] = False
    log(f"=== v5 queue2 done: {sum(res.values())}/{len(jobs)} artifacts ===")
    for n, ok in res.items():
        log(f"    {'OK  ' if ok else 'MISS'} {n}")


if __name__ == "__main__":
    main()
