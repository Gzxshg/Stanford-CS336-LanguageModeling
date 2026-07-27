"""Generate text from a trained checkpoint (assignment problem `decoding`).

Example:
    uv run generate.py --ckpt checkpoints/ckpt_best.pt \
        --prompt "Once upon a time" --temperature 0.8 --top-p 0.9
"""

import argparse

import torch

from cs336_basics.generate import generate
from cs336_basics.tokenizer import Tiny_BPETokenizer
from cs336_basics.transformer_lm import TransformerLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/ckpt_best.pt")
    p.add_argument("--tokenizer", default="data/tinystories_tokenizer.pt",
                   help="torch-saved dict with 'vocab' and 'merges'")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0, help="<= 0 means greedy")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    # model config, must match the checkpoint
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    return p.parse_args()


def main():
    args = parse_args()

    tok = torch.load(args.tokenizer, weights_only=False)
    tokenizer = Tiny_BPETokenizer(
        vocab=tok["vocab"], merges=tok["merges"], special_tokens=["<|endoftext|>"]
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt} (iteration {ckpt.get('iteration', '?')})")

    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print("=" * 40)
    print(text)


if __name__ == "__main__":
    main()
