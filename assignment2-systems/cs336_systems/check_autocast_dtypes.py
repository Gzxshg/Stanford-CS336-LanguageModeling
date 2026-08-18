"""Empirically check autocast dtypes for benchmarking_mixed_precision (a)."""
import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


for dtype in (torch.float16, torch.bfloat16):
    torch.manual_seed(0)
    model = ToyModel(16, 8).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 16, device="cuda")

    with torch.autocast(device_type="cuda", dtype=dtype):
        p = next(model.parameters())
        print(f"--- autocast {dtype} ---")
        print("param dtype            :", p.dtype)
        h1 = model.relu(model.fc1(x))
        print("fc1 (matmul) out dtype :", h1.dtype)
        h2 = model.ln(h1)
        print("layer_norm out dtype   :", h2.dtype)
        logits = model.fc2(h2)
        print("logits dtype           :", logits.dtype)
        loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 8, (4,), device="cuda"))
        print("loss dtype             :", loss.dtype)

    opt.zero_grad()
    loss.backward()
    print("grad dtype             :", p.grad.dtype)
