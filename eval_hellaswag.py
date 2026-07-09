"""Simple HellaSwag evaluator for HSPMN models - zero-shot completion accuracy.

For each example:
  - ctx (sentence start)
  - 4 endings, one correct (label)
Score each (ctx, ending) by per-token CE loss; pick the lowest-loss ending.

Output: accuracy + per-example log.

Usage:
    python3 eval_hellaswag.py \
        --ckpt /opt/docker/LLM/HSPMN/checkpoints_p4_350m/hymba-with-nsa_p4_final.pt \
        --variant hymba-with-nsa
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tokenizers import Tokenizer
from train_v4 import build_model, TrainConfig
from utils_v3_0 import setup_env, setup_logging, seed_everything, get_device

setup_env()
logger = setup_logging("hellaswag")


@torch.no_grad()
def score_completion(model, tokenizer, ctx_text, ending_text, device):
    """Return per-token CE for the ending tokens, conditioned on ctx."""
    ctx_ids = tokenizer.encode(ctx_text).ids
    full_ids = ctx_ids + tokenizer.encode(" " + ending_text).ids
    full_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    out = model(full_t)
    logits = out["logits"]
    # Score ending tokens: positions len(ctx_ids) .. len(full)-1 predict tokens
    # at len(ctx_ids)+1 .. len(full).
    n_ctx = len(ctx_ids)
    if len(full_ids) <= n_ctx + 1:
        return float("inf")
    target = torch.tensor(full_ids[n_ctx + 1 :], dtype=torch.long, device=device)
    pred_logits = logits[0, n_ctx:-1]  # [n_ending - 1, vocab]
    loss = F.cross_entropy(pred_logits, target, reduction="mean")
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--data", default="/opt/docker/LLM/HSPMN/data/hellaswag_val.jsonl")
    ap.add_argument(
        "--tokenizer", default="/opt/docker/LLM/HSPMN/data/tokenizer/tokenizer.json"
    )
    ap.add_argument("--max_examples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_d = dict(ckpt["config"])
    train_cfg = TrainConfig(**cfg_d)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = build_model(train_cfg, 50257, device, dtype)
    model.load_state_dict(ckpt["model"])
    model.train(False)
    logger.info(f"loaded {args.ckpt}")

    correct = 0
    total = 0
    per_example_loss_gap = []
    with open(args.data) as f:
        for line in f:
            ex = json.loads(line)
            ctx = ex["ctx"]
            endings = ex["endings"]
            label = ex["label"]
            losses = [
                score_completion(model, tokenizer, ctx, e, device) for e in endings
            ]
            pred = int(np.argmin(losses))
            if pred == label:
                correct += 1
            total += 1
            gap = losses[label] - min(losses[i] for i in range(4) if i != label)
            per_example_loss_gap.append(gap)
            if total % 100 == 0:
                logger.info(f"  {total}/{args.max_examples}: acc={correct / total:.4f}")
            if total >= args.max_examples:
                break

    acc = correct / total
    mean_loss_gap = float(np.mean(per_example_loss_gap))
    logger.info("\n=== Final ===")
    logger.info(f"Accuracy: {acc:.4f} ({correct}/{total})")
    logger.info(
        f"Mean (target loss − best wrong) gap: {mean_loss_gap:+.4f}  (negative = correct lower CE)"
    )

    out_md = Path(
        args.out_md or f"/opt/docker/LLM/HSPMN/results/hellaswag_{args.variant}_eval.md"
    )
    out_md.write_text(
        f"# HellaSwag - {args.variant}\n\n"
        f"**Checkpoint:** `{args.ckpt}`\n"
        f"**Examples:** {total}\n"
        f"**Accuracy:** {acc:.4f} ({correct}/{total})\n"
        f"**Mean target − best-wrong CE gap:** {mean_loss_gap:+.4f}\n\n"
        f"**Note:** random baseline = 0.25 (4-way). Small-model floor ~0.27-0.30.\n"
        f"Strong sub-2B models reach 0.30-0.45.\n"
    )
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
