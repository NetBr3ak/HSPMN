"""Split-position evaluation for v6 mixture corpora.

Reports next-token CE separately on RECALL-ANSWER positions (the value token
after "? k :") and on all other (prose/background) positions of a mixture
validation stream. The gate-channel prediction: any routing benefit must
concentrate at answer positions, where the attention stream is decisive by
construction; prose positions should show none.

Usage:
    python3 eval_mix_split.py --variant hymba-with-nsa-gated \
        --ckpt checkpoints_v6/.../hymba-with-nsa-gated_final.pt \
        --data_dir data/mix_pi25 --out_md results/v6_split_gated_pi25_s42.md
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from train_v4 import TrainConfig, build_model
from utils_v3_0 import seed_everything, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_windows", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg = TrainConfig(
        variant=args.variant,
        n_layers=args.n_layers,
        dim=args.dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        seq_len=args.seq_len,
        batch_size=args.batch,
        grad_accum=1,
        steps=1,
        lr=1e-3,
        warmup_steps=1,
        nsa_window=256,
    )
    valid = np.load(f"{args.data_dir}/valid_tokens.npy", mmap_mode="r")
    mask = np.load(f"{args.data_dir}/valid_answer_mask.npy", mmap_mode="r")
    model = build_model(cfg, 50257, device, dtype)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    S = args.seq_len
    rng = np.random.default_rng(args.seed)
    n_win = (args.n_windows // args.batch) * args.batch  # divisible groups
    starts = rng.integers(0, len(valid) - S - 1, size=(n_win,)).reshape(-1, args.batch)
    ce_ans, ce_bg, n_ans, n_bg = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for group in starts:
            x = np.stack([valid[a : a + S] for a in group]).astype(np.int64)
            m = np.stack([mask[a + 1 : a + S] for a in group]).astype(bool)
            x_t = torch.from_numpy(x).to(device)
            logits = model(x_t)["logits"][:, :-1, :]
            tgt = x_t[:, 1:]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none"
            )
            ce = ce.view(x_t.size(0), -1).float().cpu().numpy()
            ce_ans += float(ce[m].sum())
            n_ans += int(m.sum())
            ce_bg += float(ce[~m].sum())
            n_bg += int((~m).sum())

    mean_ans = ce_ans / max(1, n_ans)
    mean_bg = ce_bg / max(1, n_bg)
    lines = [
        f"# Split eval: {args.variant} on {args.data_dir}",
        f"ckpt: {args.ckpt}",
        f"answer positions: n={n_ans}  CE={mean_ans:.4f}  "
        f"PPL={math.exp(min(mean_ans, 20)):.2f}",
        f"background positions: n={n_bg}  CE={mean_bg:.4f}  "
        f"PPL={math.exp(min(mean_bg, 20)):.2f}",
    ]
    report = "\n".join(lines)
    print(report)
    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
