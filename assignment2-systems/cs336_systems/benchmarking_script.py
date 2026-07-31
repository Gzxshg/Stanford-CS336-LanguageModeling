from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import torch
import timeit
import statistics
import argparse

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
parser.add_argument("--mode", choices=["forward-only", "forward-and-backward", "full-training-steps"], type=str, default="forward-only", help="mode, forward-only or forward-and-backward or full-training-steps")
parser.add_argument("--warmup_steps", type=int, default=5, help="number of warmup steps")
parser.add_argument("--measurement_steps", type=int, default=10, help="number of measurement steps")
args=parser.parse_args()

model_config=hyper_model_para_dict[args.model_size]

scaled_test_GPT=BasicsTransformerLM(
    vocab_size=model_config["vocab_size"],
    context_length=model_config["context_length"],
    d_model=model_config["d_model"],
    d_ff=model_config["d_ff"],
    num_layers=model_config["num_layers"],
    num_heads=model_config["num_heads"]
).to("cuda")

optimizer=AdamW(scaled_test_GPT.parameters(),lr=1e-4)

x=torch.randint(0,10000,(4,512),device="cuda")
y_labels=torch.randint(0,10000,(4,512),device="cuda")


for i in range(args.warmup_steps):
    warm_up_y_pred=scaled_test_GPT(x)
    warm_up_loss=cross_entropy(warm_up_y_pred,y_labels)
    warm_up_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

forward_times=[]
backward_times=[]
optimizer_times=[]

for i in range(args.measurement_steps):    
    torch.cuda.synchronize()
    t_start=timeit.default_timer()

    y_pred=scaled_test_GPT(x)
    loss=cross_entropy(y_pred,y_labels)
    torch.cuda.synchronize()
    t_forward=timeit.default_timer()

    if args.mode=="forward-and-backward" or args.mode=="full-training-steps":
        
        optimizer.zero_grad()
        loss.backward()

        torch.cuda.synchronize()
        t_backward=timeit.default_timer()

    if args.mode=="full-training-steps":
        optimizer.step()
        torch.cuda.synchronize()
        t_end=timeit.default_timer()

    forward_times.append(t_forward - t_start)

    if args.mode=="forward-and-backward" or args.mode=="full-training-steps":
        backward_times.append(t_backward - t_forward)

    if args.mode=="full-training-steps":
        optimizer_times.append(t_end - t_backward)
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