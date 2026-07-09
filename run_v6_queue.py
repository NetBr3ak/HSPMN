"""v6 experiment queue - the ROUTABILITY PHASE DIAGRAM (pre-registered).

The v5 paper measured that per-token routing between attention and SSM streams
is impossible on uniform prose: the routing target carries ~0.03 decodable bits
against the ~0.72-bit full-benefit threshold of the gate-channel bound, and the
optimal policy is always-on. The bound's constructive corollary: routing can
only work where the data makes "which stream helps" predictable. This queue
tests that corollary directly, with predictions registered BEFORE the decisive
perplexity numbers exist.

Pre-registered predictions (2026-07-07, before any v6 run):
  P1  The optimal-probe ceiling I_max(pi) rises monotonically with the recall
      fraction pi of the corpus (measured on the gated checkpoint at each pi,
      seed 42, before the remaining PPL seeds finish).
  P2  Learned gate beats the frozen-random gate (3-seed CIs separated) only at
      the pi where I_max approaches or crosses the threshold; at pi=0 the known
      null must reproduce.
  P3  Any gate benefit concentrates at recall-answer positions (split eval);
      background-prose CE shows no gate effect at any pi.
  P4  Stream decorrelation (v6 architecture) raises I_max at fixed pi relative
      to the plain gated model; if it does but PPL does not improve, the
      redundancy explanation is wrong and the bound's premise fails (kill
      criterion).

Falsification is symmetric: if I_max stays flat in pi, the mixture design is
wrong or the ceiling is architecture-limited, and the paper's "data is the
lever" claim dies. Either outcome is publishable; that is the point.

Grid (140M recipe of the v5 paper: 12L/768d/12h/4kv, S=1024, B=16, ga=4,
3000 steps, lr 1e-3, ~98M tokens/run, ~35 min/run on the RTX 5090):
  Stage 0  build mixture corpora pi in {0, 0.10, 0.25, 0.50}   (CPU)
  Stage 1  {gated, randgate, base, pcgate} x 4 pi x seeds {42,1337,2026}
           + ceiling probe on gated s42 per pi (runs FIRST at each pi)
  Stage 2  MI (non-circular, label = base at same pi) for gated+randgate per pi
  Stage 3  v6 decor architecture: coef sweep {0.03,0.1,0.3} at pi=0.25 s42,
           then {decor, decor-gated} x pi {0, 0.25} x 3 seeds at coef 0.1
           + ceiling probe on decor-gated s42
  Stage 4  split eval (answer vs prose CE) for every final checkpoint
  Stage 5  aggregate -> results/v6_routability_summary.md

Total ~63 training runs + ~40 measurements ~= 2 GPU-days.

Contract (same as run_v5_queue*): verify-by-artifact, skip-if-done,
continue-on-fail, per-job logs. Waits for the GPU to be free before stage 1.

    nohup python3 run_v6_queue.py > results/v6_queue.out 2>&1 &
    python3 run_v6_queue.py --list
"""
import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/opt/docker/LLM/HSPMN")
LOG = ROOT / "results" / "v6_queue.log"
CKPT = "checkpoints_v6"

PIS = ["00", "10", "25", "50"]           # tags; pi = tag/100
SEEDS = [42, 1337, 2026]
DIMS = "--n_layers 12 --dim 768 --num_heads 12 --num_kv_heads 4"
BASE = (f"{DIMS} --seq_len 1024 --batch_size 16 --grad_accum 4 "
        "--steps 3000 --warmup_steps 150 --lr 1e-3 --eval_every 500 "
        "--eval_iters 32")
PROBE_ARGS = f"--n_batches 16 --batch 2 --seq_len 512 {DIMS}"
MI_ARGS = f"--n_batches 4 --batch 2 --seq_len 512 {DIMS}"


def jb(name, cmd, art, critical=False):
    return {"name": name, "cmd": cmd, "artifact": art, "critical": critical}


def data_dir(pi):
    return f"data/mix_pi{pi}"


def save_dir(variant, pi, seed):
    return f"{CKPT}/{variant}_pi{pi}_s{seed}"


def final_ckpt(variant, pi, seed):
    return f"{save_dir(variant, pi, seed)}/{variant}_final.pt"


def train_job(variant, pi, seed, extra=""):
    sd = save_dir(variant, pi, seed)
    cmd = (f"python3 train_v4.py --variant {variant} {BASE} --seed {seed} "
           f"--data_dir {data_dir(pi)} --save_dir {sd} {extra}")
    return jb(f"train_{variant}_pi{pi}_s{seed}", cmd,
              final_ckpt(variant, pi, seed), critical=True)


QUEUE = []

# ---- Stage 0: corpora (CPU; runs even while GPU is busy) -------------------
QUEUE.append(jb("corpora",
                "python3 build_mix_corpus.py --pis 0.0 0.1 0.25 0.5",
                "data/mix_pi50/meta.json", critical=True))

# ---- Stage 1+2 interleaved per pi: gated s42 -> ceiling -> controls --------
for pi in PIS:
    QUEUE.append(train_job("hymba-with-nsa-gated", pi, 42))
    QUEUE.append(jb(
        f"ceiling_gated_pi{pi}",
        f"python3 probe_gate_ceiling.py --variant hymba-with-nsa-gated "
        f"--ckpt {final_ckpt('hymba-with-nsa-gated', pi, 42)} "
        f"--data_dir {data_dir(pi)} {PROBE_ARGS} "
        f"--out_md results/v6_ceiling_gated_pi{pi}.md",
        f"results/v6_ceiling_gated_pi{pi}.md", critical=True))
    QUEUE.append(train_job("hymba-with-nsa-randgate", pi, 42))
    QUEUE.append(train_job("hymba-with-nsa", pi, 42))
    QUEUE.append(train_job("hymba-with-nsa-pcgate", pi, 42))

for seed in (1337, 2026):
    for pi in PIS:
        for v in ("hymba-with-nsa-gated", "hymba-with-nsa-randgate",
                  "hymba-with-nsa", "hymba-with-nsa-pcgate"):
            QUEUE.append(train_job(v, pi, seed))

for pi in PIS:
    for v in ("hymba-with-nsa-gated", "hymba-with-nsa-randgate"):
        QUEUE.append(jb(
            f"mi_{v.split('-')[-1]}_pi{pi}",
            f"python3 measure_gate_channel_mi.py --variant {v} "
            f"--ckpt {final_ckpt(v, pi, 42)} "
            f"--label_ckpt {final_ckpt('hymba-with-nsa', pi, 42)} "
            f"--data_dir {data_dir(pi)} {MI_ARGS} "
            f"--out_md results/v6_mi_{v.split('-')[-1]}_pi{pi}.md",
            f"results/v6_mi_{v.split('-')[-1]}_pi{pi}.md"))

# ---- Stage 3: v6 decorrelated-stream architecture --------------------------
for coef in ("0.03", "0.3"):   # 0.1 is covered by the 3-seed arm below
    sd = f"{CKPT}/hymba-with-nsa-decor_pi25_s42_c{coef}"
    QUEUE.append(jb(
        f"decor_sweep_c{coef}",
        f"python3 train_v4.py --variant hymba-with-nsa-decor {BASE} --seed 42 "
        f"--decor_coef {coef} --data_dir {data_dir('25')} --save_dir {sd}",
        f"{sd}/hymba-with-nsa-decor_final.pt"))

for pi in ("00", "25"):
    for seed in SEEDS:
        for v in ("hymba-with-nsa-decor", "hymba-with-nsa-decor-gated"):
            QUEUE.append(train_job(v, pi, seed, extra="--decor_coef 0.1"))

QUEUE.append(jb(
    "ceiling_decor_gated_pi25",
    f"python3 probe_gate_ceiling.py --variant hymba-with-nsa-decor-gated "
    f"--ckpt {final_ckpt('hymba-with-nsa-decor-gated', '25', 42)} "
    f"--data_dir {data_dir('25')} {PROBE_ARGS} "
    f"--out_md results/v6_ceiling_decor_gated_pi25.md",
    "results/v6_ceiling_decor_gated_pi25.md", critical=True))

# ---- Stage 4: split eval for every trained checkpoint ----------------------
_split_variants = [("hymba-with-nsa-gated", PIS), ("hymba-with-nsa-randgate", PIS),
                   ("hymba-with-nsa", PIS), ("hymba-with-nsa-pcgate", PIS),
                   ("hymba-with-nsa-decor", ("00", "25")),
                   ("hymba-with-nsa-decor-gated", ("00", "25"))]
for v, pis in _split_variants:
    for pi in pis:
        for seed in SEEDS:
            out = f"results/v6_split_{v}_pi{pi}_s{seed}.md"
            QUEUE.append(jb(
                f"split_{v}_pi{pi}_s{seed}",
                f"python3 eval_mix_split.py --variant {v} "
                f"--ckpt {final_ckpt(v, pi, seed)} "
                f"--data_dir {data_dir(pi)} {DIMS} --out_md {out}", out))

# ---- Stage 5: aggregate ----------------------------------------------------
QUEUE.append(jb("aggregate", "python3 aggregate_v6.py",
                "results/v6_routability_summary.md", critical=True))


def stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg):
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def done(j):
    return (ROOT / j["artifact"]).exists()


def gpu_free(min_free_mib=24000):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0]) >= min_free_mib
    except Exception:
        return True


def wait_for_gpu(job_name):
    if job_name in ("corpora", "aggregate"):
        return
    waited = 0
    while not gpu_free():
        if waited == 0:
            log("GPU busy; waiting (poll 300 s) ...")
        time.sleep(300)
        waited += 300
    if waited:
        log(f"GPU free after {waited/3600:.1f} h wait")


def run_job(j):
    if done(j):
        log(f"SKIP {j['name']} - {j['artifact']} exists")
        return True
    wait_for_gpu(j["name"])
    log(f"START {j['name']}: {j['cmd']}")
    jlog = ROOT / "results" / f"v6_{j['name']}.log"
    with open(jlog, "w") as f:
        rc = subprocess.call(j["cmd"], shell=True, cwd=ROOT, stdout=f,
                             stderr=subprocess.STDOUT)
    ok = done(j)
    log(f"END   {j['name']}: rc={rc} artifact={'OK' if ok else 'MISSING'}; "
        f"log={jlog.name}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    if args.list:
        for j in QUEUE:
            print(f"  {'DONE' if done(j) else 'pending':8s} {j['name']}"
                  f"{' [critical]' if j['critical'] else ''}")
        print(f"\n{sum(1 for j in QUEUE if not done(j))} pending of {len(QUEUE)}")
        return
    jobs = QUEUE if not args.only else [j for j in QUEUE if j["name"] in args.only]
    log(f"=== v6 routability queue start: {len(jobs)} jobs ===")
    res = {}
    for j in jobs:
        try:
            res[j["name"]] = run_job(j)
        except Exception as e:
            log(f"ERROR {j['name']}: {e!r}")
            res[j["name"]] = False
    log(f"=== v6 queue done: {sum(res.values())}/{len(jobs)} artifacts ===")
    for n, ok in res.items():
        log(f"    {'OK  ' if ok else 'MISS'} {n}")


if __name__ == "__main__":
    main()
