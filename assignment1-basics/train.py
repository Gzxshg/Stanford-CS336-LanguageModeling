"""Training loop for the Transformer LM (assignment problem `training_together`).

All model/optimizer/training hyperparameters are configurable via CLI so that
later sweeps and ablations can be launched with different flags.

Example:
    uv run train.py \
        --train-data data/tinystories_train_tokens.npy \
        --valid-data data/tinystories_valid_tokens.npy \
        --batch-size 128 --total-iters 10000 --lr 3e-3
"""

import argparse
import contextlib
import os
import time

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    p.add_argument("--train-data", required=True, help="path to training tokens (.npy or raw uint16 .bin)")
    p.add_argument("--valid-data", required=True, help="path to validation tokens")
    # model
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # optimizer
    p.add_argument("--lr", type=float, default=1e-3, help="max learning rate (peak of the schedule)")
    p.add_argument("--min-lr", type=float, default=1e-4, help="final learning rate of the schedule")
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # lr schedule
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--cosine-cycle-iters", type=int, default=None, help="defaults to --total-iters")
    # training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--total-iters", type=int, default=5000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # logging / checkpoints
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=20, help="batches used to estimate validation loss")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    # misc
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", help="run forward/backward under bf16 autocast")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def load_tokens(path: str) -> np.ndarray:
    """Memory-map a token array without loading it into RAM."""
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.uint16, mode="r")


def autocast_ctx(args: argparse.Namespace):
    if args.bf16:
        device_type = "cuda" if args.device.startswith("cuda") else "cpu"
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def compute_loss(model, x, y, args):
    with autocast_ctx(args):
        logits = model(x)
    # cross-entropy always in fp32 for numerical stability
    return cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def evaluate(model, data, args) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(data, args.batch_size, args.context_length, args.device)
        losses.append(compute_loss(model, x, y, args).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_data = load_tokens(args.train_data)
    valid_data = load_tokens(args.valid_data)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_it = 0
    if args.resume:
        start_it = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from {args.resume} at iteration {start_it}")

    cosine_cycle_iters = args.cosine_cycle_iters or args.total_iters
    tokens_per_iter = args.batch_size * args.context_length
    print(f"params: {num_params/1e6:.1f}M | tokens/iter: {tokens_per_iter} | "
          f"budget: {args.total_iters * tokens_per_iter / 1e6:.0f}M tokens")

    logger = None
    if args.wandb:
        import wandb
        logger = wandb.init(project="cs336-a1", name=args.run_name, config=vars(args))

    model.train()
    best_val = float("inf")
    t0 = time.time()
    window_tokens = 0

    for it in range(start_it, args.total_iters):
        lr = get_lr_cosine_schedule(
            it, args.lr, args.min_lr, args.warmup_iters, cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        loss = compute_loss(model, x, y, args)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.norm(
            torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
        ).item()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        window_tokens += tokens_per_iter

        if it % args.log_every == 0:
            tok_per_s = window_tokens / (time.time() - t0)
            msg = (f"it {it:>6} | train loss {loss.item():.4f} | lr {lr:.2e} | "
                   f"grad_norm {grad_norm:.2f} | {tok_per_s:.3g} tok/s")
            print(msg, flush=True)
            if logger:
                logger.log({"train_loss": loss.item(), "lr": lr, "grad_norm": grad_norm,
                            "tok_per_s": tok_per_s}, step=it)
            t0 = time.time()
            window_tokens = 0

        if it % args.eval_every == 0 or it == args.total_iters - 1:
            val_loss = evaluate(model, valid_data, args)
            print(f"it {it:>6} | valid loss {val_loss:.4f}", flush=True)
            if logger:
                logger.log({"valid_loss": val_loss}, step=it)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, it, os.path.join(args.ckpt_dir, "ckpt_best.pt"))

        if (it + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, it + 1,
                            os.path.join(args.ckpt_dir, f"ckpt_step{it + 1}.pt"))

    save_checkpoint(model, optimizer, args.total_iters,
                    os.path.join(args.ckpt_dir, "ckpt_final.pt"))
    print(f"done. best valid loss {best_val:.4f}; checkpoints in {args.ckpt_dir}/")
    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
