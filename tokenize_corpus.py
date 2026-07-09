"""Tokenize wikitext-103 with GPT2 BPE → packed token streams.

Output: data/{train,valid}_tokens.npy as int32 arrays. Single contiguous stream
per split; the training loop slices windows of S+1 from it.
"""

import argparse
import os
import time
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

DEFAULT_DATA_DIR = "/opt/docker/LLM/HSPMN/data"


def load_parquet_text(path):
    table = pq.read_table(path)
    col = table.column("text").to_pylist()
    return col


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help="Data directory (default: /opt/docker/LLM/HSPMN/data)",
    )
    p.add_argument("--out_train", default=None)
    p.add_argument("--out_valid", default=None)
    args = p.parse_args()

    # Use provided output paths or default to data_dir
    if args.out_train is None:
        args.out_train = f"{args.data_dir}/train_tokens.npy"
    if args.out_valid is None:
        args.out_valid = f"{args.data_dir}/valid_tokens.npy"

    tok_path = f"{args.data_dir}/tokenizer/tokenizer.json"
    print(f"Loading tokenizer from {tok_path}")
    tokenizer = Tokenizer.from_file(tok_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = 50256
    print(f"EOS id: {eos_id}  vocab size: {tokenizer.get_vocab_size()}")

    for split, files, out in [
        ("valid", [f"{args.data_dir}/wikitext_valid.parquet"], args.out_valid),
        (
            "train",
            [
                f"{args.data_dir}/wikitext_train_0.parquet",
                f"{args.data_dir}/wikitext_train_1.parquet",
            ],
            args.out_train,
        ),
    ]:
        if os.path.exists(out):
            print(
                f"{split}: {out} exists, skipping ({os.path.getsize(out) / 1e6:.1f} MB)"
            )
            continue
        print(f"\n=== {split} ===")
        t0 = time.time()
        all_tokens = []
        for fp in files:
            print(f"  reading {fp}")
            texts = load_parquet_text(fp)
            print(f"    {len(texts)} rows")
            # Filter empty rows.
            texts = [t for t in texts if t and t.strip()]
            print(f"    {len(texts)} non-empty rows")
            # Encode in batches - tokenizers' encode_batch handles parallel.
            BATCH = 4096
            for start in range(0, len(texts), BATCH):
                chunk = texts[start : start + BATCH]
                encs = tokenizer.encode_batch(chunk)
                for e in encs:
                    all_tokens.extend(e.ids)
                    all_tokens.append(eos_id)
                if start % (BATCH * 10) == 0:
                    print(
                        f"    {start}/{len(texts)} done  total tokens={len(all_tokens):,}",
                        flush=True,
                    )
        arr = np.array(all_tokens, dtype=np.int32)
        np.save(out, arr)
        elapsed = time.time() - t0
        print(f"  {split}: {len(arr):,} tokens  {elapsed:.1f}s  saved → {out}")


if __name__ == "__main__":
    main()
