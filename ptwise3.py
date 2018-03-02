#!/usr/bin/env python

from __future__ import print_function
import time
import torch

# kernelPointwiseApply3<TensorCPowOp<float>, ..., Adims=2, BDims=2, CDims=2>
if False:
    # Duration reduced by 24% (~8.97 to ~6.78 us).
    nrows, ncols, ncols2 = 1000, 256, 128

    # Duration reduced by 15% (~32.4 to ~27.4 us)
    nrows, ncols, ncols2 = 4000, 256, 128

    # Duration reduced by 9% (~71 to ~65 us).
    nrows, ncols, ncols2 = 1024, 4096, 1024

    # Duration reduced by 3% (~1.43 to ~1.39 ms).
    nrows, ncols, ncols2 = 8000, 4000, 3000

    A = torch.cuda.FloatTensor(nrows, ncols).fill_(1.0)
    B = torch.cuda.FloatTensor(nrows, ncols).fill_(2.0)
    C = torch.cuda.FloatTensor(nrows, ncols)

    for _ in range(500):
        torch.pow(A[:, :ncols2], B[:, :ncols2], out=C[:, :ncols2])

# kernalPointwiseApply2<..., ADims=2, BDims=2>
# Duration reduced by 15% (~365 to ~312 us).
if False:
    A = torch.cuda.FloatTensor(4096, 4096).fill_(1.0)
    B = torch.cuda.FloatTensor(2048).fill_(0.99999)
    A = A[:, :2048]

    for _ in range(2000):
        #time.sleep(0.001)
        A.pow_(B)

# kernalPointwiseApply3<..., ADims=-2, BDims=-2, CDims=-1>
# Duration reduced by 12% (~2.58 to ~2.26 ms).
if False:
    A = torch.cuda.FloatTensor(4096, 4096, 4).fill_(1.0)
    B = torch.cuda.FloatTensor(4096, 8).fill_(0.999)
    C = torch.cuda.FloatTensor(4096, 4096, 4).fill_(1.0)
    B = B[:, :4]

    for _ in range(2000):
        torch.add(A, B, out=C)

# kernelPointwiseApply3<TensorMulOp<__half>, ..., ADims=2, BDims=2, CDims=2>
# (broadcast operation)
# Duration reduced by 35% (~266 to ~172 us).
if False:
    A = torch.cuda.HalfTensor(4096, 4096).fill_(1.0)
    B = torch.cuda.HalfTensor(2048).fill_(0.999)
    C = torch.cuda.HalfTensor(4096, 4096).fill_(1.0)

    for _ in range(2000):
        torch.mul(A[:, :2048], B, out=C[:, :2048])

#*************************************************
# kernelPointwiseApply3<TensorMulOp<__half>, ..., ADims=2, BDims=2, CDims=2>
# (broadcast operation)
# Duration reduced by 35% (~1.06 to ~0.69 ms).
if True:
    A = torch.cuda.HalfTensor(8192, 8192).fill_(1.0)
    B = torch.cuda.HalfTensor(4096).fill_(0.999)
    C = torch.cuda.HalfTensor(8192, 8192).fill_(1.0)

    for _ in range(2000):
        torch.mul(A[:, :4096], B, out=C[:, :4096])
#*************************************************

# Hmm, this was "worst case" before, but it's actually positive now...
# (~154 to ~137 us)
if False:
    A = torch.cuda.IntTensor(2048, 2048)
    A = A[:, :2000]
    B = torch.cuda.IntTensor(2000).fill_(10)

    for _ in range(100):
        A.remainder_(B)

# Shown as "bad" (+5.99%) in cmp.0301B.log, but no meaningful difference when
# run here (~1% worse?)
if False:
    A = torch.cuda.DoubleTensor(1000, 256).fill_(1.0)
    C = torch.cuda.DoubleTensor(1000, 256).fill_(1.0)

    for _ in range(100):
        torch.tanh(A[:, :200], out=C[:, :200])

#-----------------------------------------------
# Old benchmarks.

if False:
    # Performance increases by ~20%.
    A = torch.cuda.FloatTensor(1000, 256).fill_(1.0)
    B = torch.cuda.FloatTensor(128).fill_(1.01)
    A = A[:, :128]

    #for _ in range(100):
    #    A.pow_(B)

    # A larger example.
    A = torch.cuda.FloatTensor(10000, 8192).fill_(1.0)
    B = torch.cuda.FloatTensor(4096).fill_(1.01)
    A = A[:, :4096]

    for _ in range(100):
        A.pow_(B)

    # "Worst" case: performance degrades by ~5%.
    A = torch.cuda.IntTensor(2048, 2048)
    A = A[:, :2000]
    B = torch.cuda.IntTensor(2000).fill_(10)

    #for _ in range(100):
    #    A.remainder_(B)

torch.cuda.synchronize()
