#!/bin/sh
# coding=utf-8
"exec" "python" "-u" "$0" "$@"
#
# Checking the speed of index_select (indexSelectLargeIndex).

import collections
import fractions
import itertools
import random
import time

import numpy as np
import torch

import test_util

STRIDES_ = [3, 5, 7, 8, 16, 64, 100, 255, 256, 257]

class Tester(object):
    def __init__(self):
        pass

    # Given a tuple 'dimensions' of sizes, run the test on every permutation
    # such that:
    #   (1) pick one size as 'out_sz',
    #   (2) pick a permutation of the rest as 'in_shape',
    #   (3) pick one dimension as 'idx_dim'.
    #
    # 'prob' is used when there are too many possible permutations: we randomly
    # sample test cases with the given probability.
    def run(self, rep_count, dimensions, prob=1.0):
        test_util.rep_count = rep_count
        self.prob = prob

        print('\n\n====================\n'
              'rep_count = {} dimensions = {}'
              .format(rep_count, dimensions))

        for idx in range(len(dimensions)):
            out_sz = dimensions[idx]
            dims = dimensions.copy()
            del dims[idx]

            for perm in itertools.permutations(range(len(dims))):
                in_shape = [dims[k] for k in perm]
                for idx_dim in range(len(dims)):
                    self._run_test2(in_shape, idx_dim, out_sz)

    # Build tensors A, B in every possible strides (as long as the storage has
    # no missing holes), and run the test.
    def _run_test2(self, in_shape, idx_dim, out_sz):
        in_sz = in_shape[idx_dim]
        out_shape = in_shape.copy()
        out_shape[idx_dim] = out_sz

        print('out_shape = {} in_shape = {} idx_dim = {}'
              .format(out_shape, in_shape, idx_dim))

        for perm1 in itertools.permutations(range(len(out_shape))):
            B = None

            test_type = '{} {} {} {}'.format(
                out_shape, in_shape, idx_dim, perm1)
            r = test_util.stable_pseudorandom(test_type)
            if r < self.prob:
                B = self._mktensor(out_shape, perm1)

                # 'in_sz' is used as the # of rows to fill here.
                print('  B = {} (stride {}) dim = {} fill_cnt = {}'
                      .format(out_shape, B.stride(), idx_dim, in_sz))
                np.random.seed(int(r * 100000))
                self._test_fill(in_sz, B, idx_dim)

            for perm2 in itertools.permutations(range(len(in_shape))):
                # Randomly sample test cases.
                test_type = '{} {} {} {} {}'.format(
                    out_shape, in_shape, idx_dim, perm1, perm2)
                r = test_util.stable_pseudorandom(test_type)
                if r > self.prob:
                    continue

                if B is None: B = self._mktensor(out_shape, perm1)
                A = self._mktensor(in_shape, perm2)

                print('  B = {} (stride {}) A = {} (stride {}) dim = {}'
                      .format(out_shape, B.stride(),
                              in_shape, A.stride(), idx_dim))
                np.random.seed(int(r * 100000))
                self._run_test3(A, B, idx_dim)

    def _test_fill(self, fill_cnt, B, idx_dim):
        out_sz = B.shape[idx_dim]

        for name, idxs in self._make_fill_idx(fill_cnt, out_sz).items():
            B.uniform_(-0.1, 0.1)
            B0 = B.cpu()
            B.index_fill_(idx_dim, idxs, 1.0)
            B0.index_fill_(idx_dim, idxs.cpu(), 1.0)

            assert (B.cpu() - B0).norm() < 1e-4

            dummy = self._random_situation()
            test_util.time_cuda(
                'index_fill_', name,
                lambda: B.index_fill_(idx_dim, idxs, 1.0))

    def _run_test3(self, A, B, idx_dim):
        in_sz = A.shape[idx_dim]
        out_sz = B.shape[idx_dim]

        for name, idxs in self._make_scatter_idx(in_sz, out_sz).items():
            A.uniform_(-0.1, 0.1)
            B.uniform_(-0.1, 0.1)
            A0, B0, idxs0 = A.cpu(), B.cpu(), idxs.cpu()
            B.index_add_(idx_dim, idxs, A)
            B0.index_add_(idx_dim, idxs0, A0)
            assert (B.cpu() - B0).norm() < 1e-4

            B.index_copy_(idx_dim, idxs, A)
            B0.index_copy_(idx_dim, idxs0, A0)
            assert (B.cpu() - B0).norm() < 1e-4

            dummy = self._random_situation()
            test_util.time_cuda(
                'index_add_', name,
                lambda: B.index_add_(idx_dim, idxs, A))

            test_util.time_cuda(
                'index_copy_', name,
                lambda: B.index_copy_(idx_dim, idxs, A))

        ptr = B.data_ptr()
        stride = B.stride()
        storage_sz = B.storage().size()

        for name, idxs in self._make_gather_idx(in_sz, out_sz).items():
            A.uniform_(-0.1, 0.1)
            B.uniform_(-0.1, 0.1)
            A0, B0, idxs0 = A.cpu(), B.cpu(), idxs.cpu()
            A.index_select(idx_dim, idxs, out=B)
            A0.index_select(idx_dim, idxs0, out=B0)
            assert (B.cpu() - B0).norm() < 1e-4

            dummy = self._random_situation()
            test_util.time_cuda(
                'index_select', name,
                lambda: A.index_select(idx_dim, idxs, out=B))

        # Sanity check.
        assert B.data_ptr() == ptr and \
               B.stride() == stride and \
               B.storage().size() == storage_sz

    # Create a tensor of the given shape, where dimensions are ordered by
    # 'perm'.
    def _mktensor(self, shape, perm):
        assert len(shape) == len(perm)
        strides = [0] * len(shape)

        stride = 1
        for dim in reversed(perm):
            strides[dim] = stride
            stride *= shape[dim]
        total_size = stride

        A = torch.cuda.FloatStorage(total_size)
        B = torch.cuda.FloatTensor().set_(A, 0, shape, strides)
        assert B.shape == tuple(shape)
        assert B.stride() == tuple(strides)
        B.uniform_(-0.1, 0.1)

        return B

    # Build and return a randomly sized tuple of tensors, so that we have
    # slightly different "environment" when each test case is run.  (Otherwise I
    # observed strange test cases that are consistently slow: but when I run
    # them by themselves I cannot reliably replicate it.)
    def _random_situation(self):
        def _mk():
            sz1 = random.randint(1024, 2048)
            sz2 = random.randint(1024, 2048)
            return torch.cuda.FloatTensor(sz1, sz2).fill_(0.1)

        x = [_mk() for _ in range(random.randint(1, 5))]
        torch.cuda.synchronize()
        return x

    #---------------------------------------------
    # Create indices.

    # A "scatter index" is used to test index_add_() and index_copy_(): it
    # scatters each input row in [0, in_sz) in to some output row in [0,
    # out_sz).  Hence, an index contains 'in_sz' values, with each value in [0,
    # out_sz).
    #
    # Indices must be distinct (because otherwise two input rows will try to
    # write into the same output row).
    def _make_scatter_idx(self, in_sz, out_sz):
        idxs = collections.OrderedDict()

        if out_sz < in_sz: return idxs  # Impossible!

        A = np.arange(in_sz)
        idxs['linear'] = A
        idxs['reverse'] = A[::-1].copy()

        # Spread evenly over [0, out_sz).
        A = np.arange(in_sz) * (out_sz / in_sz)
        idxs['spread'] = A.astype(np.int)

        # Strides.
        for stride in STRIDES_:
            if stride < out_sz and fractions.gcd(stride, out_sz) == 1:
                A = (np.arange(in_sz) * stride) % out_sz
                assert len(np.unique(A)) == in_sz  # Sanity check.
                idxs['strided %d' % stride] = A

        # Random permutations and sorted version.
        A = np.random.permutation(out_sz)[:in_sz]
        idxs['perm'] = A
        idxs['perm_sorted'] = np.sort(A)

        return self._transform(idxs)

    # A "fill index" simply selects rows for index_fill_().  Indices can be
    # duplicated (because we will simply write the same value).
    def _make_fill_idx(self, fill_cnt, out_sz):
        idxs = collections.OrderedDict()

        def _cap(A): return np.minimum(A, out_sz - 1)

        idxs['const'] = np.zeros(fill_cnt, dtype=np.int)

        A = _cap(np.arange(fill_cnt))
        idxs['linear'] = A
        idxs['reverse'] = A[::-1].copy()
        idxs['skip64'] = np.floor_divide(A, 64) * 64
        idxs['skip256'] = np.floor_divide(A, 256) * 256

        # Spread evenly over [0, out_sz).
        A = np.arange(fill_cnt) * (out_sz / fill_cnt)
        idxs['spread'] = _cap(A.astype(np.int))

        # Strides.
        for stride in STRIDES_:
            if stride < out_sz:
                A = (np.arange(fill_cnt) * stride) % out_sz
                idxs['strided %d' % stride] = A

        # Random values.
        A = np.random.randint(0, out_sz, fill_cnt)
        idxs['random'] = A
        idxs['random_sorted'] = np.sort(A)

        # Random permutations and sorted version.
        if fill_cnt <= out_sz:
            A = np.random.permutation(out_sz)[:fill_cnt]
            idxs['perm'] = A
            idxs['perm_sorted'] = np.sort(A)

        return self._transform(idxs)

    # A "gather index" is used to test index_select(): it selects the input row
    # for each output row.  An index contains 'out_sz' values, with each value
    # in [0, in_sz).
    #
    # Indices can be duplicatd.
    def _make_gather_idx(self, in_sz, out_sz):
        idxs = collections.OrderedDict()

        def _cap(A): return np.minimum(A, in_sz - 1)

        idxs['const'] = np.zeros(out_sz, dtype=np.int)
        idxs['wrap'] = np.arange(out_sz) % in_sz

        A = _cap(np.arange(out_sz))
        idxs['linear'] = A
        idxs['reverse'] = A[::-1].copy()
        idxs['skip64'] = np.floor_divide(A, 64) * 64
        idxs['skip256'] = np.floor_divide(A, 256) * 256

        # Spread evenly over [0, in_sz).
        A = np.arange(out_sz) * (in_sz / out_sz)
        idxs['spread'] = _cap(A.astype(np.int))

        # Strides.
        for stride in STRIDES_:
            if stride < in_sz:
                A = (np.arange(out_sz) * stride) % in_sz
                idxs['strided %d' % stride] = A

        # Random values.
        A = np.random.randint(0, in_sz, out_sz)
        idxs['random'] = A
        idxs['random_sorted'] = np.sort(A)

        # Random permutations and sorted version.
        if out_sz <= in_sz:
            A = np.random.permutation(in_sz)[:out_sz]
            idxs['perm'] = A
            idxs['perm_sorted'] = np.sort(A)

        return self._transform(idxs)

    def _transform(self, idxs):
        for key in idxs.keys():
            idxs[key] = torch.LongTensor(idxs[key]).cuda()
        return idxs

test_util.batch_count = 1

tester = Tester()

# Permutations for 4-dim tensors: takes ~50 min.
tester.run(20, [20, 40, 50, 100, 256], 0.0005)
tester.run(100, [4, 5, 16, 20, 40], 0.01)

# Permutations for smaller tensors.
tester.run(100, [15, 50, 150, 250], 0.1)
tester.run(100, [1, 5, 200, 500], 0.1)

tester.run(1000, [2, 3, 5])
tester.run(500, [32, 256, 512])
tester.run(100, [255, 256, 512])
tester.run(100, [15, 1000, 2048])
