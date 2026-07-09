"""Phase-4 350M proxy training on FineWeb-Edu.

Architecture: `hymba-with-nsa` (P2 winner) at 24L/1024d/16h/4kv config.
Token budget: ~1 B from FineWeb-Edu sample-10BT (5-10 shards tokenised).
Recipe: AdamW, lr 1e-3 cosine + warmup 500, bf16, grad clip 1.0.

Resume support: writes checkpoint every 1000 steps; reads existing checkpoint
on launch if present. Use this to amortise an interrupted run.

Usage:
    python3 train_p4_350m.py --variant hymba-with-nsa --seed 42 --steps 7000
"""

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from train_v4 import build_model, get_batch, run_validation, cosine_lr, TrainConfig
from utils_v3_0 import setup_env, setup_logging, seed_everything, get_device

setup_env()
logger = setup_logging("train_p4_350m")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="hymba-with-nsa")
    p.add_argument("--data_dir", default="./data")
    p.add_argument(
        "--train_npy",
        default=None,
        help="Path to train tokens .npy. If omitted, prefer fineweb_edu_tokens.npy "
        "then fall back to train_tokens.npy (wikitext-103).",
    )
    p.add_argument(
        "--valid_npy",
        default=None,
        help="Path to valid tokens .npy. Default: matched valid file.",
    )
    p.add_argument("--save_dir", default="./checkpoints_p4_350m")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_layers", type=int, default=24)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--num_kv_heads", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--steps", type=int, default=7000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=32)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--nsa_window", type=int, default=512)
    p.add_argument(
        "--kill_clock_h",
        type=float,
        default=12.0,
        help="Hard kill after this many wall-clock hours.",
    )
    return p.parse_args()


def find_data(args):
    """Decide which tokenised stream to use."""
    if args.train_npy is not None:
        train_path = args.train_npy
    else:
        fw = Path(args.data_dir) / "fineweb_edu_tokens.npy"
        wt = Path(args.data_dir) / "train_tokens.npy"
        train_path = str(fw) if fw.is_file() else str(wt)
    if args.valid_npy is not None:
        valid_path = args.valid_npy
    else:
        fw_v = Path(args.data_dir) / "fineweb_edu_valid_tokens.npy"
        wt_v = Path(args.data_dir) / "valid_tokens.npy"
        valid_path = str(fw_v) if fw_v.is_file() else str(wt_v)
    return train_path, valid_path


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    logger.info(f"Device: {device}  dtype: {dtype}  variant: {args.variant}")

    train_path, valid_path = find_data(args)
    logger.info(f"train data: {train_path}")
    logger.info(f"valid data: {valid_path}")
    train_tok = np.load(train_path, mmap_mode="r")
    valid_tok = np.load(valid_path, mmap_mode="r")
    vocab = int(max(train_tok.max(), valid_tok.max())) + 1
    logger.info(
        f"train tokens: {len(train_tok):,}, valid tokens: {len(valid_tok):,}, vocab={vocab}"
    )

    cfg = TrainConfig(
        variant=args.variant,
        n_layers=args.n_layers,
        dim=args.dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        steps=args.steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_iters=args.eval_iters,
        log_every=args.log_every,
        nsa_window=args.nsa_window,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    model = build_model(cfg, vocab, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Params: {n_params / 1e6:.2f}M (target 350M)")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    log_path = os.path.join(cfg.save_dir, f"{cfg.variant}_p4_log.csv")
    state_path = os.path.join(cfg.save_dir, f"{cfg.variant}_p4_state.pt")

    # Resume from checkpoint if present.
    start_step = 1
    if os.path.isfile(state_path):
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"] + 1
        logger.info(f"Resuming from step {start_step} ({state_path})")
    else:
        with open(log_path, "w") as f:
            f.write("step,train_ce,valid_ce,lr,tok_per_sec,vram_gb,wall_s\n")

    model.train()
    t_start = time.time()
    losses_window = []
    tokens_seen = 0

    KILL_S = args.kill_clock_h * 3600.0

    for step in range(start_step, cfg.steps + 1):
        wall = time.time() - t_start
        if wall > KILL_S:
            logger.warning(
                f"Kill clock {args.kill_clock_h}h reached at step {step}; saving and exiting."
            )
            break

        opt.zero_grad(set_to_none=True)
        cur_lr = cosine_lr(step, cfg.warmup_steps, cfg.steps, cfg.lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        accum_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch(train_tok, cfg.batch_size, cfg.seq_len, device)
            out = model(x, labels=y)
            loss = out["loss"] / cfg.grad_accum
            loss.backward()
            accum_loss += float(loss.item())
            tokens_seen += cfg.batch_size * cfg.seq_len
        losses_window.append(accum_loss * cfg.grad_accum)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.log_every == 0:
            mean_loss = np.mean(losses_window[-cfg.log_every :])
            elapsed = time.time() - t_start
            tps = tokens_seen / elapsed
            vram_gb = (
                torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda"
                else 0.0
            )
            logger.info(
                f"step {step:5d}/{cfg.steps}  CE={mean_loss:.4f}  lr={cur_lr:.5f}  "
                f"tok/s={tps:,.0f}  vram={vram_gb:.1f}GB  wall={elapsed / 3600:.2f}h"
            )
            with open(log_path, "a") as f:
                f.write(
                    f"{step},{mean_loss:.4f},,{cur_lr:.6f},{tps:.0f},{vram_gb:.2f},{elapsed:.0f}\n"
                )

        if step % cfg.eval_every == 0 or step == cfg.steps:
            v_ce = run_validation(model, valid_tok, cfg, device)
            logger.info(f"  → valid CE={v_ce:.4f}  ppl={math.exp(v_ce):.2f}")
            with open(log_path, "a") as f:
                f.write(
                    f"{step},,{v_ce:.4f},{cur_lr:.6f},,,{(time.time() - t_start):.0f}\n"
                )

        if step % args.ckpt_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "config": asdict(cfg),
                },
                state_path,
            )
            logger.info(f"  ckpt saved → {state_path}")

    # Final save (full eval).
    v_ce = run_validation(model, valid_tok, cfg, device)
    final_path = os.path.join(cfg.save_dir, f"{cfg.variant}_p4_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": min(step, cfg.steps),
            "config": asdict(cfg),
            "final_valid_ce": v_ce,
            "final_valid_ppl": math.exp(v_ce),
            "n_params": n_params,
            "tokens_seen": tokens_seen,
        },
        final_path,
    )
    logger.info(
        f"Final valid CE={v_ce:.4f}  ppl={math.exp(v_ce):.2f}  ({n_params / 1e6:.2f}M params)"
    )
    logger.info(f"Saved → {final_path}")


if __name__ == "__main__":
    main()
