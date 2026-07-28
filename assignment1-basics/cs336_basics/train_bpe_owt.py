"""Train a BPE tokenizer on OpenWebText (vocab size 32000).

并行方式: 父进程用 subprocess.Popen 拉起若干独立子进程,
每个子进程处理一个文本块并把 Counter pickle 到临时文件;
父进程等全部子进程退出后合并文件, 并落盘一份 word_counts 检查点
(data/owt_word_counts.pkl), 之后合并阶段崩了可直接从检查点续跑。

合并循环优化: 朴素实现每轮 max() 全扫 pair_counts(OWT 规模约千万级条目,
一轮约 1~2 秒, 31743 轮要约 10 小时)。这里改用 lazy heap:
- 堆里放过期条目不删, 弹出时校验计数是否仍新鲜;
- 每轮合并后只把计数发生变化的 pair 重新压堆;
- 选择规则与原实现严格一致: count 大者优先, 平局 pair 字节序大者优先
  (等价于 max(pair_counts.items(), key=lambda x: (x[1], x[0])))。

内存设计(容器 cgroup 上限 90G):
- worker 二进制模式按字节边界读块, 逐段 decode, 计数用 bytes 键;
- 转 tuple 键时用共享 BYTE_TOKENS 表, 避免每字节新建 bytes 对象。

产物: owt_vocab.pkl / owt_merges.pkl (仓库根目录)
     + data/owt_tokenizer.pt (打包成 torch 文件, 供 tokenize_dataset.py 使用)
"""

import argparse
import heapq
import os
import pickle
import subprocess
import sys
import time
from collections import Counter

import torch
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.train_bpe import (
    BYTE_TOKENS,
    compiled_pattern,
    build_pair_index,
    apply_merge,
    merge_word,
)
from cs336_basics.train_bpe_tinystories import (
    save_merges,
    save_vocab,
    find_longest_token,
)

INPUT_PATH = "/root/autodl-tmp/Stanford-CS336-LanguageModeling/assignment1-basics/data/owt_train.txt"
TMP_DIR = "/root/autodl-tmp/Stanford-CS336-LanguageModeling/assignment1-basics/data/_owt_counts"
COUNTS_CKPT = "/root/autodl-tmp/Stanford-CS336-LanguageModeling/assignment1-basics/data/owt_word_counts.pkl"
SPECIAL_TOKENS = ["<|endoftext|>"]
EOT_BYTES = SPECIAL_TOKENS[0].encode("utf-8")
VOCAB_SIZE = 32000


class PairKey:
    """lazy heap 元素: count 大者优先, 平局 pair 字节序大者优先。

    与原实现 max(pair_counts.items(), key=lambda x: (x[1], x[0])) 完全等价。
    """

    __slots__ = ("count", "pair")

    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def run_worker(chunk_id: int, start: int, end: int, out_path: str):
    t0 = time.time()
    with open(INPUT_PATH, "rb") as f:
        f.seek(start)
        data = f.read(end - start)

    counts: Counter = Counter()
    pos = 0
    n = len(data)
    while pos < n:
        idx = data.find(EOT_BYTES, pos)
        seg_end = idx if idx != -1 else n
        if seg_end > pos:
            text = data[pos:seg_end].decode("utf-8")
            for match in compiled_pattern.finditer(text):
                counts[match.group(0).encode("utf-8")] += 1
        if idx == -1:
            break
        pos = idx + len(EOT_BYTES)

    with open(out_path, "wb") as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"chunk {chunk_id}: {len(counts)} unique pretokens in {time.time() - t0:.1f}s", flush=True)


def build_word_counts(num_workers: int) -> Counter:
    os.makedirs(TMP_DIR, exist_ok=True)

    with open(INPUT_PATH, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, EOT_BYTES)

    procs = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        out_path = os.path.join(TMP_DIR, f"counts_{i}.pkl")
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker", "--chunk-id", str(i),
            "--start", str(start), "--end", str(end), "--out", out_path,
        ]
        procs.append(subprocess.Popen(cmd))
    print(f"launched {len(procs)} worker subprocesses", flush=True)

    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"worker {i} exited with code {rc}")
    print("all workers done, merging counts", flush=True)

    total_bytes: Counter = Counter()
    for i in range(len(procs)):
        path = os.path.join(TMP_DIR, f"counts_{i}.pkl")
        with open(path, "rb") as f:
            total_bytes.update(pickle.load(f))
        os.remove(path)
    os.rmdir(TMP_DIR)

    # 换回 merge 循环需要的 tuple-of-bytes 键形式(用共享字节对象表)
    print("converting keys to tuple-of-bytes form", flush=True)
    total = Counter(
        {tuple(BYTE_TOKENS[b] for b in w): c for w, c in total_bytes.items()}
    )
    del total_bytes
    return total


def merge_loop_owt(word_counts, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    merges = []

    print("building pair index", flush=True)
    pair_counts, pair_to_words = build_pair_index(word_counts)
    print(f"unique pairs: {len(pair_counts)}", flush=True)

    heap = [PairKey(c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    num_merges = vocab_size - len(vocab)
    for _ in tqdm(range(num_merges), desc="BPE merges"):
        if not pair_counts:
            break

        # 弹出“新鲜”的最优 pair, 丢弃计数已过期的堆条目
        top = None
        while top is None:
            if not heap:  # 防御: 堆空但 pair_counts 非空时重建
                heap = [PairKey(c, p) for p, c in pair_counts.items()]
                heapq.heapify(heap)
            cand = heapq.heappop(heap)
            if pair_counts.get(cand.pair) == cand.count:
                top = cand
        best_pair = top.pair

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        # 受影响的词 & 计数会变化的 pair 集合
        affected = list(pair_to_words.get(best_pair, {}).keys())
        changed = set()
        for w in affected:
            changed.update(zip(w, w[1:]))

        word_counts, pair_counts, pair_to_words = apply_merge(
            word_counts, pair_counts, pair_to_words, best_pair
        )

        for w in affected:
            nw = merge_word(w, best_pair)
            changed.update(zip(nw, nw[1:]))

        # 计数有变化的 pair 压入新条目(旧条目成为垃圾, 弹出时被丢弃)
        for p in changed:
            c = pair_counts.get(p)
            if c:
                heapq.heappush(heap, PairKey(c, p))

        # 垃圾条目过多时重建堆, 控制内存
        if len(heap) > 3 * len(pair_counts) + 1000:
            heap = [PairKey(c, p) for p, c in pair_counts.items()]
            heapq.heapify(heap)

    return vocab, merges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--chunk-id", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    if args.worker:
        run_worker(args.chunk_id, args.start, args.end, args.out)
        return

    num_workers = 8  # 90G cgroup 内存上限, 少开 worker 控制峰值

    if os.path.exists(COUNTS_CKPT):
        print(f"loading cached word counts from {COUNTS_CKPT}", flush=True)
        t0 = time.time()
        with open(COUNTS_CKPT, "rb") as f:
            word_counts = pickle.load(f)
        t1 = time.time()
        print(f"unique pretokens: {len(word_counts)} (loaded in {t1 - t0:.1f}s)", flush=True)
    else:
        print("Building word counts with parallel pretokenization", flush=True)
        t0 = time.time()
        word_counts = build_word_counts(num_workers)
        t1 = time.time()
        print(f"unique pretokens: {len(word_counts)}", flush=True)
        print("saving word counts checkpoint", flush=True)
        with open(COUNTS_CKPT, "wb") as f:
            pickle.dump(word_counts, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Running merges", flush=True)
    vocab, merges = merge_loop_owt(
        word_counts=word_counts,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    t2 = time.time()

    print("Saving outputs", flush=True)
    save_vocab(vocab, "owt_vocab.pkl")
    save_merges(merges, "owt_merges.pkl")
    torch.save({"vocab": vocab, "merges": merges}, "data/owt_tokenizer.pt")

    token_id, token_bytes = find_longest_token(vocab)

    print(f"Pretokenization + word_counts time: {t1 - t0:.2f}s")
    print(f"Merge loop time: {t2 - t1:.2f}s")
    print(f"Total time: {t2 - t0:.2f}s")
    print(f"Final vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")
    print(f"Longest token id: {token_id}")
    print(f"Longest token length: {len(token_bytes)}")
    print(f"Longest token bytes: {token_bytes!r}")
    print(f"Longest token decoded: {token_bytes.decode('utf-8', errors='replace')!r}")


if __name__ == "__main__":
    main()