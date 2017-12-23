#!/usr/bin/env python
#
# Testing the performance impact of incrementer.

from __future__ import print_function

import torch

import test_util

torch.manual_seed(1)

SZ = 2048
#test_util.rep_count = 20

def do_test(sz, cb):
    Z0 = torch.FloatTensor(sz, sz).uniform_(0.1, 1.0)
    Z = Z0.cuda()

    def _setup1():
        A0 = Z0.clone()
        A = Z.clone()
        assert A0.is_contiguous() and A.is_contiguous()
        return A0, A
    #do_test2('Contiguous', _setup1, _setup1, cb)

    def _setup_row():
        A0 = torch.FloatTensor(sz).uniform_(0.1, 1.0)
        A = A0.cuda()
        assert A0.is_contiguous() and A.is_contiguous()
        return A0, A
    #do_test2('Broadcast', _setup1, _setup_row, cb)

    def _setup2():
        A0, A = _setup1()
        A0 = A0[:, :-1]
        A = A[:, :-1]
        assert not A0.is_contiguous() and not A.is_contiguous()
        return A0, A
    do_test2('Non-contiguous', _setup2, _setup2, cb)

    def _setup3():
        A0, A = _setup2()
        return A0.t(), A.t()
    do_test2('Non-contiguous transposed', _setup3, _setup3, cb)

def do_test2(title, setupA, setupB, cb):
    #print('  ' + title)
    A0, A = setupA()
    B0, B = setupB()

    torch.cuda.synchronize()

    #print('diff norm = ', torch.norm(A.cpu() - A0))

    p, q = A.data_ptr(), B.data_ptr()
    cb(A, B)
    assert p == A.data_ptr()
    assert q == B.data_ptr()
    cb(A0, B0)
    assert torch.max(torch.abs(A0.cuda() - A)) < 1e-4

    test_util.time_cuda(title, lambda: cb(A, B))

print('Add')
def do_add(A, B): A += B
do_test(SZ, do_add)

if False:
    print('log')
    do_test(SZ, lambda A, B: torch.log(B, out=A))

    print('pow')
    do_test(SZ, lambda A, B: torch.pow(A, B, out=A))

# SZ=2048, Debug build
# 14.200 msec / 75.251 msec
