#!/usr/bin/env python
#
# Test 32-bit index math with large tensors that barely fits in 32bit.
#
# Try: nvprof --print-gpu-trace --print-api-trace ./large_tensor.py 2> nvprof.log

from __future__ import print_function
import os
import time

import torch

# Initialize CUDA.
dummy = torch.cuda.FloatTensor(10, 10).fill_(50)
torch.cuda.synchronize()

def show_duration(cmd_id, T0):
    torch.cuda.synchronize()
    T1 = time.time()
    print('%-10s : Elapsed %.3f ms' % (cmd_id, (T1 - T0) * 1000.0))

def do_test(size):
    print('tensor size = %d (%x) = %.3f GB' % (size, size, float(size)/GB))

    # Initialize a huge tensor.
    T0 = time.time()
    A = torch.cuda.ByteTensor(size).fill_(1)
    assert len(A) == size, len(A)
    show_duration('fill_', T0)

    # Try addition.
    T0 = time.time()
    A += 1
    show_duration('addition', T0)

    print('A[:20] = ', list(A[:20]))
    if A[0] != 2: print('ERROR!!!!!!!')

if True:
    GB = 1024 * 1024 * 1024
    do_test(2 * GB - 3)
    do_test(2 * GB - 2)
    do_test(2 * GB - 1)
    do_test(2 * GB)
    do_test(2 * GB + 1)
    do_test(2 * GB + 2)

    do_test(3 * GB - 1)
    do_test(3 * GB)
    do_test(3 * GB + 1)

    do_test(4 * GB - 3)
    do_test(4 * GB - 2)
    do_test(4 * GB - 1)
    do_test(4 * GB)
    do_test(4 * GB + 1)
    do_test(4 * GB + 2)

os.system('nvidia-smi')
