"""Train + evaluate every Phase-1 variant on MQAR / Dyck-2 / parity.

Each task uses a small LM (~1.7M params, 4 layers, dim=128, vocab task-specific)
and trains 2k steps of teacher-forced next-token CE. Reports test accuracy
(MQAR, parity) or held-out CE/PPL (Dyck-2).

Goal: in Phase 1 we measure WHICH reflexive primitive can pass each state-
tracking benchmark. Mamba-3 / RWKV-7 should pass parity + Dyck-2; pure
Transformer should fail at long lengths (TC^0 limit).

Usage:
    python3 train_synthetic.py --task mqar --variant v4-mamba3-nsa
    python3 train_synthetic.py --task parity --variant dense
    python3 train_synthetic.py --task dyck2 --variant v4-rwkv7-nsa
    python3 train_synthetic.py --all
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from synthetic_mqar import build_mqar_batch, evaluate_mqar
from synthetic_dyck import build_dyck2_batch, evaluate_dyck2, VOCAB_DYCK2
from synthetic_parity import build_parity_batch, evaluate_parity, VOCAB_PARITY


VARIANTS = ["dense", "v4-gdn-nsa", "v4-mamba3-nsa", "v4-rwkv7-nsa",
            "hymba", "hymba-with-nsa", "hymba-with-nsa-gated",
            "hymba-with-nsa-randgate", "hymba-with-nsa-pcgate",
            "hymba-with-asa"]

TASK_CFG = {
    "mqar":   {"vocab": 8192, "length": 256, "n_kv": 32, "n_query": 16},
    "dyck2":  {"vocab": VOCAB_DYCK2, "length": 256, "max_depth": 8},
    "parity": {"vocab": VOCAB_PARITY, "length": 256, "p_query": 0.1},
}


def build_lm(variant: str, vocab: int, dim: int = 128, n_layers: int = 4,
             num_heads: int = 4, num_kv_heads: int = 2, max_seq_len: int = 1024):
    if variant == "dense":
        from train_v4 import DenseLM
        return DenseLM(vocab=vocab, n_layers=n_layers, dim=dim,
                       n_heads=num_heads, n_kv_heads=num_kv_heads,
                       max_len=max_seq_len)
    if variant == "hymba":
        from hymba_lm import HymbaLM
        return HymbaLM(vocab_size=vocab, n_layers=n_layers, dim=dim,
                       num_heads=num_heads, num_kv_heads=num_kv_heads,
                       max_seq_len=max_seq_len)
    if variant.startswith("hymba-with-nsa") or variant == "hymba-with-asa":
        from hymba_with_nsa_lm import HymbaWithNSALM
        random_gate = variant == "hymba-with-nsa-randgate"
        use_attn_gate = variant in ("hymba-with-nsa-gated",
                                    "hymba-with-nsa-randgate")
        gate_mode = "predictive_coding" if variant == "hymba-with-nsa-pcgate" else "linear"
        attn_mode = "asa" if variant == "hymba-with-asa" else "nsa"
        return HymbaWithNSALM(vocab_size=vocab, n_layers=n_layers, dim=dim,
                              num_heads=num_heads, num_kv_heads=num_kv_heads,
                              max_seq_len=max_seq_len,
                              random_gate=random_gate, attn_mode=attn_mode,
                              use_attn_gate=use_attn_gate,
                              gate_mode=gate_mode)
    if variant.startswith("v4"):
        from hspmn_v4_lm import HSPMNv4LM, HSPMNv4LMConfig
        if "mamba3" in variant:
            reflexive = "mamba3"
        elif "rwkv7" in variant:
            reflexive = "rwkv7"
        else:
            reflexive = "gdn"
        attention = "asa" if "asa" in variant else ("nsa" if "nsa" in variant else "sqsk")
        cfg = HSPMNv4LMConfig(vocab_size=vocab, n_layers=n_layers, dim=dim,
                              num_heads=num_heads, num_kv_heads=num_kv_heads,
                              max_seq_len=max_seq_len, reflexive=reflexive,
                              attention=attention,
                              random_gate=("randgate" in variant))
        return HSPMNv4LM(cfg)
    raise ValueError(variant)


def make_batch(task: str, B: int, device: str, seed: int):
    cfg = TASK_CFG[task]
    if task == "mqar":
        return build_mqar_batch(B, cfg["n_kv"], cfg["n_query"],
                                vocab=cfg["vocab"], device=device, seed=seed)
    if task == "dyck2":
        ids, labels = build_dyck2_batch(B, cfg["length"], cfg["max_depth"],
                                        device=device, seed=seed)
        return ids, labels
    if task == "parity":
        return build_parity_batch(B, cfg["length"], p_query=cfg["p_query"],
                                  device=device, seed=seed)


def evaluate(task: str, model, device: str, n_examples: int = 256):
    cfg = TASK_CFG[task]
    if task == "mqar":
        return evaluate_mqar(model, cfg["n_kv"], cfg["n_query"],
                             vocab=cfg["vocab"], n_examples=n_examples,
                             device=device, batch_size=16)
    if task == "dyck2":
        return evaluate_dyck2(model, cfg["length"], cfg["max_depth"],
                              n_examples=n_examples, device=device, batch_size=16)
    if task == "parity":
        return evaluate_parity(model, cfg["length"], p_query=cfg["p_query"],
                               n_examples=n_examples, device=device, batch_size=16)


def train_one(task: str, variant: str, steps: int = 2000, B: int = 32,
              lr: float = 3e-4, device: str = "cuda", seed: int = 42) -> dict:
    cfg = TASK_CFG[task]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_lm(variant, vocab=cfg["vocab"], max_seq_len=cfg["length"] + 64)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.01, fused=(device == "cuda"))

    model.train()
    losses = []
    for step in range(1, steps + 1):
        ids, labels = make_batch(task, B, device, seed=seed * 100000 + step)
        out = model(ids, labels=labels)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError(f"task={task} variant={variant} returned no loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if step % 200 == 0:
            print(f"  step {step:4d}/{steps}  CE={np.mean(losses[-200:]):.4f}")

    eval_out = evaluate(task, model, device=device, n_examples=512)
    eval_out["params_M"] = n_params / 1e6
    eval_out["variant"] = variant
    eval_out["task"] = task
    eval_out["final_train_ce"] = float(np.mean(losses[-100:]))
    return eval_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_CFG.keys()))
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--all", action="store_true",
                    help="Run all (variant, task) combinations and dump JSON.")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="/opt/docker/LLM/HSPMN/results/phase1_synthetics.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.all:
        results = []
        for task in TASK_CFG:
            for variant in VARIANTS:
                print(f"\n=== task={task}  variant={variant} ===")
                try:
                    res = train_one(task, variant, steps=args.steps, lr=args.lr,
                                    device=device)
                    print(f"  → {res}")
                    results.append(res)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    results.append({"task": task, "variant": variant,
                                    "error": f"{type(e).__name__}: {e}"})
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults dumped → {args.out}")
    else:
        if not (args.task and args.variant):
            raise SystemExit("Must pass --task and --variant, or --all")
        res = train_one(args.task, args.variant, steps=args.steps, lr=args.lr,
                        device=device)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
