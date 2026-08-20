"""Activation checkpointing granularity sweep (2.1.6 / gradient_checkpointing b).

Runs a full training step (forward + backward + AdamW) while checkpointing every
k TransformerBlocks, and reports peak GPU memory.
"""
import argparse
import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

hyper_model_para_dict = {
    "small":  dict(vocab_size=10000, context_length=512, d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(vocab_size=10000, context_length=512, d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(vocab_size=10000, context_length=512, d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(vocab_size=10000, context_length=512, d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", choices=list(hyper_model_para_dict), default="medium")
parser.add_argument("--context_length", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--checkpoint_every", type=int, default=1,
                    help="wrap every k blocks in one checkpoint segment; 0 = no checkpointing")
parser.add_argument("--no_optimizer", action="store_true", help="skip the optimizer step (saves 2x params of memory)")
parser.add_argument("--mixed_precision", action="store_true", help="run forward under bf16 autocast")
parser.add_argument("--steps", type=int, default=3)
args = parser.parse_args()

cfg = dict(hyper_model_para_dict[args.model_size])
cfg["context_length"] = args.context_length
model = BasicsTransformerLM(**cfg).to("cuda")
optimizer = AdamW(model.parameters(), lr=1e-4)

x = torch.randint(0, cfg["vocab_size"], (args.batch_size, args.context_length), device="cuda")
y_labels = torch.randint(0, cfg["vocab_size"], (args.batch_size, args.context_length), device="cuda")

L = len(model.layers)
k = args.checkpoint_every
if k > 0 and L % k != 0:
    raise ValueError(f"num_layers={L} not divisible by k={k}")


def run_segment(layers, h):
    for layer in layers:
        h = layer(h)
    return h


def forward(x):
    h = model.token_embeddings(x)
    if k == 0:
        for layer in model.layers:
            h = layer(h)
    else:
        for start in range(0, L, k):
            seg = [model.layers[i] for i in range(start, start + k)]
            h = checkpoint(run_segment, seg, h, use_reentrant=False)
    return model.lm_head(model.ln_final(h))


from contextlib import nullcontext

mp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if args.mixed_precision else nullcontext

for step in range(args.steps):
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()
    with mp_ctx():
        logits = forward(x)
        loss = cross_entropy(logits, y_labels)
    loss.backward()
    if not args.no_optimizer:
        optimizer.step()
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"step {step}: peak {peak:.1f} MiB", flush=True)

print(f"RESULT size={args.model_size} ctx={args.context_length} k={k} "
      f"mp={args.mixed_precision} opt={not args.no_optimizer} peak={peak:.1f} MiB")
