"""2.1.5 mixed_precision_accumulation: accumulate 0.01 a thousand times in different dtypes.

Code is transcribed verbatim from the assignment handout; run it and comment on the
accuracy of each result.
"""

import torch

# 1. FP32 accumulator, FP32 addend
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

# 2. FP16 accumulator, FP16 addend
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

# 3. FP32 accumulator, FP16 addend (implicit promotion on +=)
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

# 4. FP32 accumulator, FP16 addend explicitly cast to FP32 before adding
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
