"""Tokenize a text file into a uint16 numpy array of token ids.

Splits the input at <|endoftext|> boundaries and encodes chunks in parallel
(results are concatenated in order, identical to encoding the whole file).

Example:
    uv run tokenize_dataset.py --input data/TinyStoriesV2-GPT4-train.txt \
        --output data/tinystories_train_tokens.npy
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import Tiny_BPETokenizer

EOT = b"<|endoftext|>"


def _encode_chunk(task) -> np.ndarray:
    input_path, tokenizer_path, start, end = task
    # 每个 worker 各自构造 tokenizer, 避免跨进程序列化大对象
    tok = torch.load(tokenizer_path, weights_only=False)
    tokenizer = Tiny_BPETokenizer(
        vocab=tok["vocab"], merges=tok["merges"], special_tokens=["<|endoftext|>"],
        max_pretoken_bytes=64,  # 网页垃圾文本的超长 pretoken 截段, 防止逐对合并 O(L^2) 卡死
    )
    with open(input_path, "r", encoding="utf-8") as f:
        f.seek(start)
        text = f.read(end - start)
    ids = tokenizer.encode(text)
    return np.asarray(ids, dtype=np.uint16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", default="data/tinystories_tokenizer.pt")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--num-chunks", type=int, default=0,
                   help="分块数(默认等于 num-workers); 大文件应多分块, "
                        "避免单块 token 列表占数 GB 内存")
    args = p.parse_args()

    num_chunks = args.num_chunks or args.num_workers
    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, EOT)
    tasks = [
        (args.input, args.tokenizer, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    print(f"{len(tasks)} chunks, {args.num_workers} workers")

    t0 = time.time()
    with mp.Pool(processes=args.num_workers) as pool:
        chunks = pool.map(_encode_chunk, tasks)  # map 保序
    ids = np.concatenate(chunks)

    np.save(args.output, ids)
    dt = time.time() - t0
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"saved {len(ids)} tokens to {args.output} ({size_mb:.0f} MB) in {dt:.1f}s")


if __name__ == "__main__":
    main()
