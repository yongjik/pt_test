#!/usr/bin/env python
#
# Testing dimension validation of index_add_/index_copy_/index_fill_.

from __future__ import print_function
import sys

import numpy as np
import torch

from test_util import *

if True:
  A = torch.zeros(4, 3).cuda()
  B = torch.zeros(3, 3).cuda()
  idxs = torch.LongTensor([3, 1, 0]).cuda()

  B0 = torch.cuda.FloatTensor()
  i0 = torch.cuda.LongTensor()
  i2 = torch.cuda.LongTensor(2, 2)

  D = torch.zeros(4, 3).cuda()

  assert not try_cmd('index dim', lambda: A.index_copy_(0, i0, B))
  assert not try_cmd('index dim', lambda: A.index_copy_(0, i2, B))
  assert not try_cmd('dim < src', lambda: A.index_copy_(2, idxs, B))
  assert not try_cmd('src empty', lambda: A.index_copy_(0, idxs, B0))
  assert not try_cmd('idx len', lambda: A.index_copy_(0, idxs, D))

  assert not try_cmd('index dim', lambda: A.index_add_(0, i0, B))
  assert not try_cmd('index dim', lambda: A.index_add_(0, i2, B))
  assert not try_cmd('dim < src', lambda: A.index_add_(2, idxs, B))
  assert not try_cmd('src empty', lambda: A.index_add_(0, idxs, B0))
  assert not try_cmd('idx len', lambda: A.index_add_(0, idxs, D))

  assert not try_cmd('index dim', lambda: A.index_fill_(0, i0, 1.0))
  assert not try_cmd('index dim', lambda: A.index_fill_(0, i2, 1.0))
  assert not try_cmd('dim < dest', lambda: A.index_fill_(2, idxs, 1.0))
  assert not try_cmd('dst empty', lambda: B0.index_fill_(0, idxs, 1.0))

#sys.exit(0)

def do_test(test_id, destdim, dim, idxs, srcdim, cb):
  A = torch.zeros(*destdim).cuda()
  B = torch.zeros(*srcdim).cuda()
  B2 = B.view(-1)
  B2[:] = torch.arange(0, len(B2))
  idxs = torch.LongTensor(idxs).cuda()

  success = try_cmd(test_id, lambda: cb(A, dim, idxs, B))
  #print('B = ', B)
  #print('A = ', A)

  if (success != ('OK' in test_id)):
    print('===== ERROR for [%s] =====' % test_id)

def do_test2(cb):
  do_test('OK', (4, 3), 0, [2, 1, 0], (3, 3), cb)
  do_test('OK', (3, 4), 1, [2, 1, 0], (3, 3), cb)
  do_test('OK', (4, 3, 2), 0, [2, 1, 0], (3, 3, 2), cb)
  do_test('OK', (3, 4, 2), 1, [2, 1, 0], (3, 3, 2), cb)
  do_test('OK', (2, 3, 4), 2, [2, 1, 0], (2, 3, 3), cb)

  do_test('dim > len(destdim)', (4, 3), 2, [2, 1, 0], (3, 3, 3), cb)
  do_test('dim > len(srcdim)', (4, 3, 3), 2, [2, 1, 0], (3, 3), cb)

  do_test('OK (deprecated warning)', (4, 3, 2), 0, [2, 1, 0], (3, 6), cb)
  do_test('OK (deprecated warning)', (4, 6), 0, [2, 1, 0], (3, 3, 2), cb)
  do_test('OK (deprecated warning)', (4, 3, 2), 1, [2, 1, 0], (8, 3), cb)

  do_test('mismatch 0', (4, 2), 0, [2, 1, 0], (3, 3), cb)
  do_test('mismatch 1', (4, 4, 2), 1, [2, 1, 0], (3, 3, 3), cb)
  do_test('mismatch 2', (4, 3, 4), 2, [2, 1, 0], (3, 3, 3), cb)

print('\n===== index_copy_')
do_test2(lambda A, dim, idxs, B: A.index_copy_(dim, idxs, B))
print('\n===== index_add_')
do_test2(lambda A, dim, idxs, B: A.index_add_(dim, idxs, B))

def test_fill(test_id, destdim, dim, idxs):
  A = torch.zeros(*destdim).cuda()
  idxs = torch.LongTensor(idxs).cuda()

  success = try_cmd(test_id, lambda: A.index_fill_(dim, idxs, 1.0))
  #print('B = ', B)
  #print('A = ', A)

  if (success != ('OK' in test_id)):
    print('===== ERROR for [%s] =====' % test_id)

def test_fill2():
  test_fill('OK', (4, 3), 0, [2, 1, 0])
  test_fill('OK', (3, 4), 1, [2, 1, 0])
  test_fill('OK', (4, 3, 2), 0, [2, 1, 0])
  test_fill('OK', (3, 4, 2), 1, [2, 1, 0])
  test_fill('OK', (2, 3, 4), 2, [2, 1, 0])

  test_fill('dim > len(destdim)', (4, 3), 2, [2, 1, 0])

  test_fill('OK', (4, 3, 2), 0, [2, 1, 0])
  test_fill('OK', (4, 6), 0, [2, 1, 0])
  test_fill('OK', (4, 3, 2), 1, [2, 1, 0])

print('\n===== index_fill_')
test_fill2()
