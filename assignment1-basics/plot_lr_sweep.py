"""绘制 LR sweep 对比图: train loss / grad norm / lr schedule 三子图。

用法: uv run --with matplotlib python plot_lr_sweep.py
输出: logs/lr_sweep.png
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("1e-3", "logs/lr_sweep_1e-3.log"),
    ("3e-3", "logs/tinystories_full_train.log"),
    ("1e-2", "logs/lr_sweep_1e-2.log"),
    ("3e-2", "logs/lr_sweep_3e-2.log"),
]

TRAIN_RE = re.compile(
    r"it\s+(\d+) \| train loss ([\d.]+) \| lr ([\d.e+-]+) \| grad_norm ([\d.]+)"
)
VALID_RE = re.compile(r"it\s+(\d+) \| valid loss ([\d.]+)")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for lr_tag, path in RUNS:
    its, losses, lrs, gns, vits, vlosses = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                its.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                gns.append(float(m.group(4)))
                continue
            m = VALID_RE.search(line)
            if m:
                vits.append(int(m.group(1)))
                vlosses.append(float(m.group(2)))

    label = f"lr={lr_tag}"
    axes[0].plot(its, losses, lw=1, alpha=0.85, label=label)
    axes[0].plot(vits, vlosses, "o", ms=3, color=axes[0].lines[-1].get_color())
    axes[1].plot(its, gns, lw=1, alpha=0.85, label=label)
    axes[2].plot(its, lrs, lw=1, alpha=0.85, label=label)

axes[0].set_title("train loss (markers: valid)")
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("cross-entropy")
axes[0].set_ylim(0, 10)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_title("grad norm (log scale)")
axes[1].set_xlabel("iteration")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3)

axes[2].set_title("lr schedule")
axes[2].set_xlabel("iteration")
axes[2].set_yscale("log")
axes[2].grid(alpha=0.3)

fig.suptitle("LR sweep on TinyStories (22.7M, 327.68M tokens each)")
fig.tight_layout()
fig.savefig("logs/lr_sweep.png", dpi=150)
print("saved logs/lr_sweep.png")
