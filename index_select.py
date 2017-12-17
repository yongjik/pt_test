#!/usr/bin/env python
#
# Checking the speed of index_select (indexSelectLargeIndex).

import collections
import itertools
import time
import numpy as np
import torch

class Tester(object):
    def __init__(self):
        pass

    def run(self, cnt, dimensions):
        self.cnt = cnt

        for idx in range(len(dimensions)):
            out_sz = dimensions[idx]
            dims = dimensions.copy()
            del dims[idx]

            for perm in itertools.permutations(range(len(dims))):
                in_shape = [dims[k] for k in perm]
                for idx_dim in range(len(dims)):
                    self._run_test2(in_shape, idx_dim, out_sz)

    def _run_test2(self, in_shape, idx_dim, out_sz):
        in_sz = in_shape[idx_dim]
        out_shape = in_shape.copy()
        out_shape[idx_dim] = out_sz

        print()

        for perm1 in itertools.permutations(range(len(in_shape))):
            A = self._mktensor(in_shape, perm1)

            for perm2 in itertools.permutations(range(len(in_shape))):
                B = self._mktensor(out_shape, perm2)

                print(
                    'in = {} (stride {}) out = {} (stride {}) dim = {}'
                        .format(in_shape, A.stride(),
                                out_shape, B.stride(), idx_dim))

                self._run_test3(A, B, idx_dim)

    def _run_test3(self, A, B, idx_dim):
        in_sz = A.shape[idx_dim]
        out_sz = B.shape[idx_dim]

        for name, idxs in self._make_scatter_idx(in_sz, out_sz).items():
            self._timeit(
                'index_add_', name,
                lambda: B.index_add_(idx_dim, idxs, A))

            self._timeit(
                'index_copy_', name,
                lambda: B.index_copy_(idx_dim, idxs, A))

        for name, idxs in self._make_fill_idx(in_sz, out_sz).items():
            self._timeit(
                'index_fill_', name,
                lambda: B.index_fill_(idx_dim, idxs, 1.0))

        ptr = B.data_ptr()
        for name, idxs in self._make_gather_idx(in_sz, out_sz).items():
            self._timeit(
                'index_select', name,
                lambda: A.index_select(idx_dim, idxs, out=B))
        assert B.data_ptr() == ptr  # Sanity check.

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

    def _timeit(self, name1, name2, cb):
        torch.cuda.synchronize()
        T0 = time.time()
        for step in range(self.cnt):
            cb()
        torch.cuda.synchronize()
        T1 = time.time()

        print('  %-15s %-15s : Elapsed %.4f ms' %
              (name1, name2, (T1 - T0) / self.cnt * 1000.0))

    #---------------------------------------------
    # Create indices.

    # A "scatter index" scatters each input row to some output row.  Indices
    # must be distinct (because otherwise two input rows will try to write into
    # the same output row).
    def _make_scatter_idx(self, in_sz, out_sz):
        idxs = collections.OrderedDict()

        if out_sz < in_sz: return idxs  # Impossible!

        A = np.arange(in_sz)
        idxs['linear'] = A
        idxs['reverse'] = A[::-1].copy()

        A = np.arange(in_sz) * (out_sz / in_sz)
        idxs['strided'] = A.astype(np.int)

        A = np.random.permutation(out_sz)[:in_sz]
        idxs['perm'] = A
        idxs['perm_sorted'] = np.sort(A)

        return self._transform(idxs)

    # A "fill index" simply selects some rows to fill.  Indices can be
    # duplicated (because we will simply write the same value).
    def _make_fill_idx(self, idx_sz, out_sz):
        idxs = collections.OrderedDict()

        def _cap(A): return np.minimum(A, out_sz - 1)

        idxs['const'] = np.zeros(idx_sz, dtype=np.int)

        A = _cap(np.arange(idx_sz))
        idxs['linear'] = A
        idxs['reverse'] = A[::-1].copy()
        idxs['skip64'] = np.floor_divide(A, 64) * 64
        idxs['skip256'] = np.floor_divide(A, 256) * 256

        A = np.arange(idx_sz) * (out_sz / idx_sz)
        idxs['strided'] = _cap(A.astype(np.int))

        A = np.random.randint(0, out_sz, idx_sz)
        idxs['random'] = A
        idxs['random_sorted'] = np.sort(A)

        if idx_sz <= out_sz:
            A = np.random.permutation(out_sz)[:idx_sz]
            idxs['perm'] = A
            idxs['perm_sorted'] = np.sort(A)

        return self._transform(idxs)

    # A "gather index" selects the input row for each output row.  Indices can
    # be duplicatd.
    def _make_gather_idx(self, in_sz, out_sz):
        idxs = collections.OrderedDict()

        def _cap(A): return np.minimum(A, in_sz - 1)

        idxs['const'] = np.zeros(out_sz, dtype=np.int)

        A = _cap(np.arange(out_sz))
        idxs['linear'] = A
        idxs['reverse'] = A[::-1].copy()
        idxs['skip64'] = np.floor_divide(A, 64) * 64
        idxs['skip256'] = np.floor_divide(A, 256) * 256

        A = np.arange(out_sz) * (in_sz / out_sz)
        idxs['strided'] = _cap(A.astype(np.int))

        A = np.random.randint(0, in_sz, out_sz)
        idxs['random'] = A
        idxs['random_sorted'] = np.sort(A)

        if out_sz <= in_sz:
            A = np.random.permutation(in_sz)[:out_sz]
            idxs['perm'] = A
            idxs['perm_sorted'] = np.sort(A)

        return self._transform(idxs)

    def _transform(self, idxs):
        for key in idxs.keys():
            idxs[key] = torch.LongTensor(idxs[key]).cuda()
        return idxs

tester = Tester()
tester.run(100, [2, 3, 5])

