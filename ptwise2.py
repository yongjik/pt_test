#!/usr/bin/env python
#
# Testing the performance impact of pointwise operations.

from __future__ import print_function

import torch
import test_util

torch.manual_seed(1)

#test_util.batch_count = 10
#test_util.rep_count = 500

def do_test(title, A, B):
    test_util.time_cuda(title, 'add', lambda: A.add_(B))
    #test_util.time_cuda(title, 'pow', lambda: A.pow_(B))

if True:
    # kernelPointwiseApply2<...-2, 2>
    # void kernelPointwiseApply2<TensorAddOp<float>, float, float, unsigned int, int=-2, int=2>(TensorInfo<TensorAddOp<float>, float>, TensorInfo<float, float>, float, float)
    A = torch.cuda.FloatTensor(256, 256, 4).fill_(0.1)
    B = torch.cuda.FloatTensor(256, 4).fill_(0.2)
    do_test('bcast -2 2', A, B)

if False:
    # kernelPointwiseApply2<...-2, -1>
    A = torch.cuda.FloatTensor(256, 256, 4).fill_(0.1)
    B = torch.cuda.FloatTensor(256, 8).fill_(0.2)
    B = B[:, :4]
    do_test('bcast -2 -1', A, B)
