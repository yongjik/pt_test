#!/usr/bin/env python

from __future__ import print_function
import torch

# Performance increases by ~20%.
A = torch.cuda.FloatTensor(1000, 256).fill_(1.0)
B = torch.cuda.FloatTensor(128).fill_(1.01)
A = A[:, :128]

for _ in range(100):
    A.pow_(B)

# "Worst" case: performance degrades by ~5%.
A = torch.cuda.IntTensor(2048, 2048)
A = A[:, :2000]
B = torch.cuda.IntTensor(2000).fill_(10)

for _ in range(100):
    A.remainder_(B)
