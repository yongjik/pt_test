#!/usr/bin/env python
#
# Testing the performance impact of incrementer.

from __future__ import print_function

import torch

import test_util

torch.manual_seed(1)

SZ = 2048

def do_test(sz, cb):
    print('  Continguous')
    def _setup1():
        A0 = torch.FloatTensor(sz, sz).uniform_(0.1, 1.0)
        A = A0.cuda()
        assert A0.is_contiguous() and A.is_contiguous()
        return A0, A
    do_test2(_setup1, cb)

    print('  Gapped')
    def _setup2():
        A0, A = _setup1()
        A0 = A0[:, :sz//2]
        A = A[:, :sz//2]
        assert not A0.is_contiguous() and not A.is_contiguous()
        return A0, A
    do_test2(_setup2, cb)

    print('  Gapped and transposed')
    def _setup3():
        A0, A = _setup2()
        return A0.t(), A.t()
    do_test2(_setup3, cb)

def do_test2(setup, cb):
    A0, A = setup()
    B0, B = setup()

    cb(A, B)
    cb(A0, B0)
    assert torch.norm(A.cpu() - A0) < 1e-4

    test_util.time_cuda(lambda: cb(A, B))

print('Add')
def do_add(A, B): A += B
do_test(SZ, do_add)

print('log')
do_test(SZ, lambda A, B: torch.log(B, out=A))

print('pow')
do_test(SZ, lambda A, B: torch.pow(A, B, out=A))

# SZ=2048, Debug build
# 14.200 msec / 75.251 msec
