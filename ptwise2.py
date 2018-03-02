#!/usr/bin/env python
#
# Testing the performance impact of pointwise operations.

from __future__ import print_function

import torch
import test_util

torch.manual_seed(1)

# test_util.batch_count = 10

def do_test(title, A, B, C):
    use_pow = True
    try:
        A.pow_(B)
    except:
        use_pow = False

    # Fill A/B/C with random values.
    for X in A, B, C:
        Y = torch.cuda.FloatTensor(X.shape).uniform_(1, 15)
        X.copy_(Y)

    # Verify result: they all share the same index logic, so let's just check add().
    do_test = True
    try:
        A0, B0 = A.cpu(), B.cpu()
        A0.add_(B0)
    except:
        # Apparently HalfTensor doesn't work in CPU.
        do_test = False

    if do_test:
        A.add_(B)
        A0, B0 = A.cpu(), B.cpu()
        assert (A.cpu() - A0).float().norm() < 1e-4

        A0, B0 = A.cpu(), B.cpu()
        torch.add(A, B, out=C)
        C0 = A0.add(B0)
        assert (C.cpu() - C0).float().norm() < 1e-4

    test_util.time_cuda(title, 'add_', lambda: A.add_(B))
    test_util.time_cuda(title, 'add', lambda: torch.add(A, B, out=C))

    test_util.time_cuda(title, 'mul_', lambda: A.mul_(B))
    test_util.time_cuda(title, 'mul', lambda: torch.mul(A, B, out=C))

    if use_pow:
        test_util.time_cuda(title, 'tanh_', lambda: A.tanh_())
        test_util.time_cuda(title, 'tanh', lambda: torch.tanh(A, out=C))
        test_util.time_cuda(title, 'pow_', lambda: A.pow_(B))
        test_util.time_cuda(title, 'pow', lambda: torch.pow(A, B, out=C))
    else:
        test_util.time_cuda(title, 'remainder_', lambda: A.remainder_(B))
        test_util.time_cuda(title, 'remainder', lambda: torch.remainder(A, B, out=C))

def run(tensor_type, sz1, sz2, sz2_cuts):
    RUN_PHASE = -1

    if RUN_PHASE in [-1, 1]:
        # Contiguous
        #
        # void kernelPointwiseApply3<TensorAddOp<float>, float, float, float,
        # unsigned int, int=-2, int=-2, int=-2>(...)

        A = tensor_type(sz1, sz2).fill_(0)
        B = tensor_type(sz1, sz2).fill_(1)
        C = tensor_type(sz1, sz2).fill_(2)

        do_test('cont (-2) -2 -2', A, B, C)

    if RUN_PHASE in [-1, 2]:
        # Non-contiguous (Dims=1)
        #
        # void kernelPointwiseApply3<TensorAddOp<float>, float, float, float,
        # unsigned int, int=-2, int=-2, int=1>(...)

        A = tensor_type(sz1, sz2).fill_(0)
        B = tensor_type(sz1, sz2, 4).fill_(1)
        C = tensor_type(sz1, sz2).fill_(2)

        B = B[:, :, 0]
        do_test('non-cont (-2) -2 1', A, B, C)

    if RUN_PHASE in [-1, 3]:
        # Non-contiguous (Dims=2)
        #
        # void kernelPointwiseApply3<TensorAddOp<float>, float, float, float,
        # unsigned int, int=2, int=2, int=2>(...)

        for sz2_cut in sz2_cuts:
            A = tensor_type(sz1, sz2).fill_(0)
            B = tensor_type(sz1, sz2).fill_(1)
            C = tensor_type(sz1, sz2).fill_(2)

            A = A[:, :sz2_cut]
            B = B[:, :sz2_cut]
            C = C[:, :sz2_cut]
            do_test('non-cont (cut=%d) (2) 2 2' % sz2_cut, A, B, C)

    if RUN_PHASE in [-1, 4]:
        # Contiguous broadcast
        #
        # void kernelPointwiseApply3<TensorAddOp<float>, float, float, float,
        # unsigned int, int=-2, int=-2, int=2>(...)

        A = tensor_type(sz1, sz2).fill_(0)
        B = tensor_type(sz2).fill_(1)
        C = tensor_type(sz1, sz2).fill_(2)
        do_test('bcast (-2) -2 2', A, B, C)

    if RUN_PHASE in [-1, 5]:
        # Non-contiguous broadcast
        #
        # void kernelPointwiseApply3<TensorAddOp<float>, float, float, float,
        # unsigned int, int=2, int=2, int=2>(...)

        for sz2_cut in sz2_cuts:
            A = tensor_type(sz1, sz2).fill_(0)
            B = tensor_type(sz2_cut).fill_(1)
            C = tensor_type(sz1, sz2).fill_(2)

            A = A[:, :sz2_cut]
            C = C[:, :sz2_cut]
            do_test('bcast (cut=%d) (2) 2 2' % sz2_cut, A, B, C)

    if RUN_PHASE in [-1, 6]:
        # Non-contiguous broadcast (larger dimension)
        #
        # void kernelPointwiseApply3<TensorAddOp<float>, float, float, float,
        # unsigned int, int=-2, int=-2, int=-1>(...)

        A = tensor_type(sz1, sz2, 4).fill_(0)
        B = tensor_type(sz2, 8).fill_(1)
        C = tensor_type(sz1, sz2, 4).fill_(2)

        B = B[:, :4]
        do_test('bcast (-2) -2 -1', A, B, C)

types = [
    torch.cuda.ByteTensor,
    torch.cuda.ShortTensor,
    torch.cuda.IntTensor,
    torch.cuda.LongTensor,
    torch.cuda.HalfTensor,
    torch.cuda.FloatTensor,
    torch.cuda.DoubleTensor,
]

# dimension 1, dimension 2, how to slice dimension 2,
# number of repetition for benchmarks.
SIZES = [
#   [64, 16, [8, 15], 1000],
#   [200, 200, [50, 100, 180], 500],
    [1000, 256, [64, 128, 200], 500],
    [2048, 2048, [512, 2000], 100],
    [4096, 4096, [512, 2048, 4000], 25],
]

for tensor_type in types:
    print('========== Testing ', tensor_type, ' ==========')
    for sizes in SIZES:
        print('  Testing sizes = ', sizes)
        sz1, sz2, sz2_cuts, rep = sizes
        test_util.rep_count = rep
        run(tensor_type, sz1, sz2, sz2_cuts)
