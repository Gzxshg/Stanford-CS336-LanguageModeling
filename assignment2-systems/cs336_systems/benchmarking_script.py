from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import torch
import torch.cuda.nvtx as nvtx
import timeit
import statistics
import argparse
from contextlib import nullcontext

hyper_model_para_dict={
    "small": {
        "vocab_size": 10000,
        "context_length": 512,
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12
    },

    "medium": {
        "vocab_size": 10000,
        "context_length": 512,
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16
    },

    "large": {
        "vocab_size": 10000,
        "context_length": 512,
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20
    },

    "xl": {
        "vocab_size": 10000,
        "context_length": 512,
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32
    },

    "10B": {
        "vocab_size": 10000,
        "context_length": 512,
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36
    }
}

parser=argparse.ArgumentParser()
parser.add_argument("--model_size", choices=["small", "medium", "large", "xl", "10B"], type=str, default="small", help="model size")
parser.add_argument("--mode", choices=["forward-only", "forward-and-backward", "full-training-steps", "forward-save"], type=str, default="forward-only", help="mode, forward-only or forward-and-backward or full-training-steps; forward-save runs the forward pass with grad enabled (keeping autograd residuals alive) but no backward")
parser.add_argument("--warmup_steps", type=int, default=5, help="number of warmup steps")
parser.add_argument("--measurement_steps", type=int, default=10, help="number of measurement steps")
parser.add_argument("--context_length", type=int, default=None, help="override the context length (default: model config value)")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument("--annotate", action="store_true", help="swap in an NVTX-annotated scaled_dot_product_attention")
parser.add_argument("--mixed_precision", action="store_true", help="run the forward pass under torch.autocast with bfloat16")
parser.add_argument("--memory_profile", action="store_true", help="record CUDA memory history during the measurement steps and dump a snapshot pickle")
args=parser.parse_args()

model_config=hyper_model_para_dict[args.model_size]
context_length=args.context_length if args.context_length is not None else model_config["context_length"]

# BF16 autocast when requested, no-op context otherwise. Autocast only wraps the forward
# pass: op dtypes recorded in the autograd graph during forward fully determine backward.
mp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if args.mixed_precision else nullcontext

scaled_test_GPT=BasicsTransformerLM(
    vocab_size=model_config["vocab_size"],
    context_length=context_length,
    d_model=model_config["d_model"],
    d_ff=model_config["d_ff"],
    num_layers=model_config["num_layers"],
    num_heads=model_config["num_heads"]
).to("cuda")

if args.annotate:
    import math
    import cs336_basics.model
    from einops import einsum
    from cs336_basics.nn_utils import softmax

    @nvtx.range("scaled dot product attention")
    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = K.shape[-1]
        with nvtx.range("computing attention scores"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
        with nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)
        with nvtx.range("final matmul"):
            attention_output = einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")
        return attention_output

    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

optimizer=AdamW(scaled_test_GPT.parameters(),lr=1e-4)

x=torch.randint(0,10000,(args.batch_size,context_length),device="cuda")
y_labels=torch.randint(0,10000,(args.batch_size,context_length),device="cuda")


# forward-only 是纯推理基准，用 inference_mode 避免 autograd 图把激活一直挂在显存里
grad_ctx = torch.inference_mode if args.mode == "forward-only" else torch.enable_grad

for i in range(args.warmup_steps):
    with grad_ctx():
        with mp_ctx():
            warm_up_y_pred=scaled_test_GPT(x)
            warm_up_loss=cross_entropy(warm_up_y_pred,y_labels)
    if args.mode in ("forward-only", "forward-save"):
        continue
    warm_up_loss.backward()
    if args.mode=="full-training-steps":
        optimizer.step()
    optimizer.zero_grad()

forward_times=[]
backward_times=[]
optimizer_times=[]

if args.memory_profile:
    # Record allocation history for the measurement phase only (warmup excluded),
    # then dump a snapshot loadable at pytorch.org/memory_viz.
    torch.cuda.memory._record_memory_history(max_entries=1000000)

for i in range(args.measurement_steps):
    with nvtx.range("measure"):
        torch.cuda.synchronize()
        t_start=timeit.default_timer()

        with grad_ctx():
            with mp_ctx():
                with nvtx.range("forward"):
                    y_pred=scaled_test_GPT(x)
                    loss=cross_entropy(y_pred,y_labels)
        torch.cuda.synchronize()
        t_forward=timeit.default_timer()

        if args.mode=="forward-and-backward" or args.mode=="full-training-steps":

            optimizer.zero_grad()
            with nvtx.range("backward"):
                loss.backward()

            torch.cuda.synchronize()
            t_backward=timeit.default_timer()

        if args.mode=="full-training-steps":
            with nvtx.range("optimizer_step"):
                optimizer.step()
            torch.cuda.synchronize()
            t_end=timeit.default_timer()

    forward_times.append(t_forward - t_start)

    if args.mode=="forward-and-backward" or args.mode=="full-training-steps":
        backward_times.append(t_backward - t_forward)

    if args.mode=="full-training-steps":
        optimizer_times.append(t_end - t_backward)

if args.memory_profile:
    snapshot_name = f"memory_{args.model_size}_ctx{context_length}_{args.mode}{'_bf16' if args.mixed_precision else ''}.pickle"
    torch.cuda.memory._dump_snapshot(snapshot_name)
    torch.cuda.memory._record_memory_history(enabled=None)
    print("Memory snapshot saved to", snapshot_name)
    print("peak memory allocated (MiB):", torch.cuda.max_memory_allocated() / 2**20)

print("Average forward time:",sum(forward_times)/len(forward_times))
print("forward time deviation",statistics.stdev(forward_times))
print("origin forward time",forward_times)

if args.mode=="forward-and-backward" or args.mode=="full-training-steps":
    print("Average backward time:",sum(backward_times)/len(backward_times))
    print("backward time deviation",statistics.stdev(backward_times))
    print("origin backward time",backward_times)

    if args.mode=="full-training-steps":
        print("Average optimizer time:",sum(optimizer_times)/len(optimizer_times))
        print("optimizer time deviation",statistics.stdev(optimizer_times))
        print("origin optimizer time",optimizer_times)