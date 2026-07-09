"""HSPMN v4.0 LM training on wikitext-103.

Real next-token CE on a real corpus (~119M tokens). Sized for single 5090.
"""

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_v3_0 import setup_env, setup_logging, seed_everything, get_device
from hspmn_v4_lm import HSPMNv4LM, HSPMNv4LMConfig

setup_env()
logger = setup_logging(__name__)

DATA_DIR = "/opt/docker/LLM/HSPMN/data"


@dataclass
class TrainConfig:
    variant: str = "v4-gdn-nsa"
    n_layers: int = 8
    dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 2
    seq_len: int = 1024
    batch_size: int = 16
    grad_accum: int = 4
    steps: int = 2000
    lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    log_every: int = 25
    eval_every: int = 250
    eval_iters: int = 20
    seed: int = 42
    target_sparsity: float = 0.25
    aux_loss_coef: float = 0.0
    nsa_window: int = 256
    save_dir: str = "/opt/docker/LLM/HSPMN/checkpoints"
    random_gate: bool = False
    data_dir: str = DATA_DIR
    decor_coef: float = 0.0


class DenseBlock(nn.Module):
    """μP-tuned modern dense Transformer block.
    Per Cerebras-GPT μP guide + Phase-1 protocol:
      GQA 16/4, RMSNorm, RoPE, SwiGLU 8/3·d hidden, no biases on linears.
    """

    def __init__(self, dim, n_heads, n_kv_heads, swiglu_ratio=8 / 3, max_len=2048):
        super().__init__()
        from bench_dense_baseline import RoPE

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.kv_groups = n_heads // n_kv_heads
        self.norm1 = nn.RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim, max_len)
        self.norm2 = nn.RMSNorm(dim)
        # SwiGLU 8/3·d (Llama-3 / Mamba-3 standard, not 4·d).
        hidden = int(round(dim * swiglu_ratio / 64.0)) * 64  # round to 64
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        n = self.norm1(x)
        q = self.q_proj(n).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(n).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(n).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))
        n2 = self.norm2(x)
        return x + self.down(F.silu(self.gate(n2)) * self.up(n2))


class DenseLM(nn.Module):
    """μP-tuned dense baseline. Untied embeddings; μP-style scaled init.

    Init recipe (Cerebras-GPT μP):
      - Embeddings: N(0, 1)         (token unit-variance signal)
      - Linear weights: N(0, σ_base² / fan_in)  with σ_base independent of width
      - lm_head: N(0, σ_base² / dim)  (separate from embed, untied)
      - Per-layer down_proj rescaled by 1/√(2·n_layers)  (residual scaling)
    """

    def __init__(
        self,
        vocab,
        n_layers,
        dim,
        n_heads,
        n_kv_heads,
        max_len,
        swiglu_ratio=8 / 3,
        mup_base_std=0.02,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [
                DenseBlock(
                    dim, n_heads, n_kv_heads, swiglu_ratio=swiglu_ratio, max_len=max_len
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(dim)
        # UNTIED embedding ↔ lm_head (Cerebras-GPT μP recommendation).
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.mup_base_std = mup_base_std
        self._init_weights_mup()

    def _init_weights_mup(self):
        sigma = self.mup_base_std
        # Embeddings: N(0, 1) per μP / Cerebras-GPT.
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)
        # lm_head: scaled small.
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=sigma / (self.dim**0.5))
        # Linears: scaled by 1/√fan_in. Residual projections (o_proj, down)
        # additionally scaled by 1/√(2·n_layers) for stable residual stream.
        res_scale = 1.0 / math.sqrt(2.0 * self.n_layers)
        for layer in self.layers:
            for name, lin in [
                ("q", layer.q_proj),
                ("k", layer.k_proj),
                ("v", layer.v_proj),
                ("o", layer.o_proj),
                ("gate", layer.gate),
                ("up", layer.up),
                ("down", layer.down),
            ]:
                fan_in = lin.weight.shape[1]
                std = sigma / math.sqrt(fan_in)
                if name in ("o", "down"):
                    std = std * res_scale
                nn.init.normal_(lin.weight, mean=0.0, std=std)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, labels=None):
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            sl = logits[..., :-1, :].contiguous()
            sb = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                sl.view(-1, sl.size(-1)), sb.view(-1), ignore_index=-100
            )
        return {"logits": logits, "loss": loss, "aux_loss": logits.new_zeros(())}


def build_model(cfg: TrainConfig, vocab: int, device, dtype):
    if cfg.variant == "dense":
        m = DenseLM(
            vocab, cfg.n_layers, cfg.dim, cfg.num_heads, cfg.num_kv_heads, cfg.seq_len
        )
    elif cfg.variant == "hymba":
        from hymba_lm import HymbaLM

        m = HymbaLM(
            vocab,
            cfg.n_layers,
            cfg.dim,
            cfg.num_heads,
            cfg.num_kv_heads,
            max_seq_len=cfg.seq_len,
        )
    elif cfg.variant in (
        "hymba-with-nsa",
        "hymba-with-nsa-gated",
        "hymba-with-nsa-randgate",
        "hymba-with-asa",
        "hymba-with-nsa-pcgate",
        "hymba-with-nsa-select",
        "hymba-with-nsa-decor",
        "hymba-with-nsa-decor-gated",
        "hymba-with-nsa-decor-pcgate",
    ):
        from hymba_with_nsa_lm import HymbaWithNSALM

        random_gate = cfg.variant == "hymba-with-nsa-randgate"
        use_attn_gate = cfg.variant in (
            "hymba-with-nsa-gated",
            "hymba-with-nsa-randgate",
            "hymba-with-nsa-decor-gated",
        )
        gate_mode = (
            "predictive_coding"
            if cfg.variant in ("hymba-with-nsa-pcgate", "hymba-with-nsa-decor-pcgate")
            else "linear"
        )
        attn_mode = "asa" if cfg.variant == "hymba-with-asa" else "nsa"
        nsa_select_from_compress = cfg.variant == "hymba-with-nsa-select"
        stream_decor = cfg.variant.startswith("hymba-with-nsa-decor")
        m = HymbaWithNSALM(
            vocab,
            cfg.n_layers,
            cfg.dim,
            cfg.num_heads,
            cfg.num_kv_heads,
            max_seq_len=cfg.seq_len,
            nsa_window_size=cfg.nsa_window,
            random_gate=random_gate,
            attn_mode=attn_mode,
            use_attn_gate=use_attn_gate,
            gate_mode=gate_mode,
            nsa_select_from_compress=nsa_select_from_compress,
            stream_decor=stream_decor,
            decor_coef=(cfg.decor_coef if stream_decor else 0.0),
        )
    elif cfg.variant.startswith("v4"):
        if "mamba3" in cfg.variant:
            reflexive = "mamba3"
        elif "rwkv7" in cfg.variant:
            reflexive = "rwkv7"
        elif "gdn" in cfg.variant:
            reflexive = "gdn"
        else:
            reflexive = "elu1"
        if "asa" in cfg.variant:
            attention = "asa"
        elif "nsa" in cfg.variant:
            attention = "nsa"
        else:
            attention = "sqsk"
        v4 = HSPMNv4LMConfig(
            vocab_size=vocab,
            n_layers=cfg.n_layers,
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            max_seq_len=cfg.seq_len,
            target_sparsity=cfg.target_sparsity,
            reflexive=reflexive,
            attention=attention,
            nsa_window_size=cfg.nsa_window,
            router_l1_coef_init=0.0,
            router_z_loss_coef=1e-4,
            aux_loss_coef=cfg.aux_loss_coef,
            random_gate=(
                getattr(cfg, "random_gate", False) or "randgate" in cfg.variant
            ),
        )
        m = HSPMNv4LM(v4)
    else:
        raise ValueError(f"unknown variant: {cfg.variant}")
    return m.to(device, dtype=dtype)


def get_batch(tokens: np.ndarray, B: int, S: int, device):
    starts = np.random.randint(0, len(tokens) - S - 1, size=(B,))
    x = np.stack([tokens[s : s + S] for s in starts])
    y = np.stack([tokens[s + 1 : s + S + 1] for s in starts])
    x_t = torch.from_numpy(x).long().to(device, non_blocking=True)
    y_t = torch.from_numpy(y).long().to(device, non_blocking=True)
    return x_t, y_t


@torch.no_grad()
def run_validation(model, tokens, cfg, device):
    was_training = model.training
    model.train(False)
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(tokens, cfg.batch_size, cfg.seq_len, device)
        out = model(x, labels=y)
        losses.append(float(out["loss"].item()))
    model.train(was_training)
    return float(np.mean(losses))


def cosine_lr(step, warmup, total, peak):
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.9 + peak * 0.1


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        default="v4-gdn-nsa",
        choices=[
            "dense",
            "hymba",
            "hymba-with-nsa",
            "hymba-with-nsa-gated",
            "hymba-with-nsa-randgate",
            "hymba-with-nsa-pcgate",
            "hymba-with-nsa-select",
            "hymba-with-asa",
            "hymba-with-nsa-decor",
            "hymba-with-nsa-decor-gated",
            "hymba-with-nsa-decor-pcgate",
            "v4-elu1",
            "v4-gdn",
            "v4-gdn-nsa",
            "v4-mamba3-nsa",
            "v4-rwkv7-nsa",
            "v4-gdn-asa",
            "v4-mamba3-nsa-randgate",
        ],
    )
    p.add_argument(
        "--random_gate",
        action="store_true",
        help="Freeze ReMoE gate at N(0,1) init - Aquino-Michaels protocol baseline",
    )
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_kv_heads", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nsa_window", type=int, default=256)
    p.add_argument("--target_sparsity", type=float, default=0.25)
    p.add_argument("--aux_loss_coef", type=float, default=0.0)
    p.add_argument(
        "--decor_coef",
        type=float,
        default=0.1,
        help="stream-decorrelation weight (decor variants only)",
    )
    p.add_argument(
        "--data_dir",
        default=DATA_DIR,
        help="directory with train_tokens.npy / valid_tokens.npy",
    )
    p.add_argument("--save_dir", default="/opt/docker/LLM/HSPMN/checkpoints")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=20)
    args = p.parse_args()

    cfg = TrainConfig(
        **{k: v for k, v in vars(args).items() if k in TrainConfig.__dataclass_fields__}
    )
    seed_everything(cfg.seed)
    device = get_device()
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    logger.info(f"Device: {device}  dtype: {dtype}")
    logger.info(f"Variant: {cfg.variant}")
    logger.info(f"Config: {asdict(cfg)}")

    train_tok = np.load(f"{cfg.data_dir}/train_tokens.npy", mmap_mode="r")
    valid_tok = np.load(f"{cfg.data_dir}/valid_tokens.npy", mmap_mode="r")
    # GPT-2 BPE floor: mixture corpora may not contain the top token id, but
    # checkpoints must stay shape-compatible across corpora.
    vocab = max(int(max(train_tok.max(), valid_tok.max())) + 1, 50257)
    logger.info(
        f"Train tokens: {len(train_tok):,}  Valid: {len(valid_tok):,}  vocab={vocab}"
    )

    model = build_model(cfg, vocab, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Params: {n_params / 1e6:.2f}M")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    os.makedirs(cfg.save_dir, exist_ok=True)
    log_path = os.path.join(cfg.save_dir, f"{cfg.variant}_log.csv")
    with open(log_path, "w") as f:
        f.write("step,train_ce,valid_ce,lr,tok_per_sec,vram_gb\n")

    model.train()
    t_start = time.time()
    losses_window = []
    tokens_seen = 0

    for step in range(1, cfg.steps + 1):
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
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
                f"step {step:4d}/{cfg.steps}  CE={mean_loss:.4f}  lr={cur_lr:.5f}  "
                f"tok/s={tps:,.0f}  vram={vram_gb:.1f}GB"
            )
            with open(log_path, "a") as f:
                f.write(
                    f"{step},{mean_loss:.4f},,{cur_lr:.6f},{tps:.0f},{vram_gb:.2f}\n"
                )

        if step % cfg.eval_every == 0 or step == cfg.steps:
            v_ce = run_validation(model, valid_tok, cfg, device)
            logger.info(f"  → valid CE={v_ce:.4f}  ppl={math.exp(v_ce):.2f}")
            with open(log_path, "a") as f:
                f.write(f"{step},,{v_ce:.4f},{cur_lr:.6f},,\n")

    save_path = os.path.join(cfg.save_dir, f"{cfg.variant}_final.pt")
    torch.save(
        {"model": model.state_dict(), "config": asdict(cfg), "step": cfg.steps},
        save_path,
    )
    logger.info(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
