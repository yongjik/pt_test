Benchmarks for PyTorch Pull Request #4493.

clean: ran on a clean branch at 58f600
idxselect: ran on PR #4493 on top of 58f600

I ran index_select.py three times each side, and compared the best (shortest)
duration for each test case:

    ./cmp_perf.py --base results/index_select/clean.*.out \
                  --exp results/index_select/idxselect.*.out \
                  > results/index_select/SCORES.txt

** NOTE: In this document, all percentages refer to "relative change of
         duration", so negative numbers are BETTER.

Statistics:
        relative duration increase
  =========================
  BAD   > 20%        :     6
  Bad   10 .. 20%    :    15
  bad   5 .. 10%     :    34
        -5 .. 5%     : 79785
  good  -10 .. -5%   :   744
  Good  -20 .. -10%  :   482
  GOOD  -50% .. -20% :   472
  BEST  < -50%       :   331

All six cases where performance degrades >20% belong to a single pathological
case (calling index_fill_ with 40 element indices when the target has only 4
columns), shown below.

NOTE: Percentages are relative to "best base-side time", which is one of the
      first three in [...] (hence one will be +0.00%, and the other two will be
      non-negative).  The next three after '/' are experiment-side times.

      The first "... -> ... (+..%)" part compares between best base-side time
      and best experiment-side time.

  > out_shape = [16, 20, 5, 4] in_shape = [16, 20, 5, 40] idx_dim = 3
  >   B = [16, 20, 5, 4] (stride (400, 20, 4, 1)) dim = 3 fill_cnt = 40
  > BAD    0.985 -> 1.196 (+21.42%) [ +0.71%  +0.00%  +0.10% / +21.73% +21.42% +21.73%]     index_fill_ const              : Elapsed 0.010 ms (0.992 ms / 100)
  > BAD    0.974 -> 1.171 (+20.23%) [ +0.51%  +0.21%  +0.00% / +22.18% +20.23% +20.64%]     index_fill_ linear             : Elapsed 0.010 ms (0.979 ms / 100)
  > BAD    0.975 -> 1.172 (+20.21%) [ +0.21%  +0.00%  +0.00% / +22.46% +20.31% +20.21%]     index_fill_ reverse            : Elapsed 0.010 ms (0.977 ms / 100)
  > Bad    0.980 -> 1.174 (+19.80%) [ +0.00%  +0.10%  +0.20% / +19.80% +20.41% +20.31%]     index_fill_ skip64             : Elapsed 0.010 ms (0.980 ms / 100)
  > Bad    0.982 -> 1.177 (+19.86%) [ +0.10%  +0.00%  +0.10% / +19.86% +20.67% +19.86%]     index_fill_ skip256            : Elapsed 0.010 ms (0.983 ms / 100)
  > Bad    0.975 -> 1.148 (+17.74%) [ +0.00%  +0.31%  +0.10% / +17.74% +19.28% +18.87%]     index_fill_ spread             : Elapsed 0.010 ms (0.975 ms / 100)
  > BAD    0.973 -> 1.169 (+20.14%) [ +0.31%  +0.10%  +0.00% / +20.14% +21.38% +21.38%]     index_fill_ strided 3          : Elapsed 0.010 ms (0.976 ms / 100)
  > BAD    0.973 -> 1.175 (+20.76%) [ +0.21%  +0.00%  +0.21% / +20.76% +20.76% +20.86%]     index_fill_ random             : Elapsed 0.010 ms (0.975 ms / 100)
  > BAD    0.970 -> 1.172 (+20.82%) [ +0.52%  +0.00%  +0.82% / +20.82% +21.34% +21.13%]     index_fill_ random_sorted      : Elapsed 0.010 ms (0.975 ms / 100)

Can be reproduced with this:

    B = torch.cuda.FloatTensor(2000, 4)
    idxs = torch.cuda.LongTensor(np.zeros(40))
    B.index_fill_(1, idxs, 1.0)

Another major case is the following.  Here the source and destination have
different strides, so we basically choose between "source-friendly" (base) and
"destination-friendly" (exp) access pattern.

Depending on the pattern, duration changes between roughly -10% and 15%.

  >   B = [15, 50, 250] (stride (50, 1, 750)) A = [15, 150, 250] (stride (1, 3750, 15)) dim = 1
  > Good   6.881 -> 6.183 (-10.14%) [ +0.15%  +0.00%  +0.13% / -10.01% -10.14%  -9.94%]     index_select const             : Elapsed 0.069 ms (6.891 ms / 100)
  > bad    7.890 -> 8.590 ( +8.87%) [ +0.39%  +0.06%  +0.00% /  +9.68%  +8.87%  +9.06%]     index_select wrap              : Elapsed 0.079 ms (7.921 ms / 100)
  > bad    7.885 -> 8.591 ( +8.95%) [ +0.36%  +0.27%  +0.00% /  +9.68%  +9.03%  +8.95%]     index_select linear            : Elapsed 0.079 ms (7.913 ms / 100)
  > bad    7.910 -> 8.621 ( +8.99%) [ +0.14%  +0.00%  +0.13% /  +9.22%  +8.99%  +9.12%]     index_select reverse           : Elapsed 0.079 ms (7.921 ms / 100)
  > good   6.896 -> 6.226 ( -9.72%) [ +0.13%  +0.09%  +0.00% /  -8.82%  -9.54%  -9.72%]     index_select skip64            : Elapsed 0.069 ms (6.905 ms / 100)
  > good   6.863 -> 6.183 ( -9.91%) [ +0.35%  +0.06%  +0.00% /  -9.49%  -9.91%  -9.73%]     index_select skip256           : Elapsed 0.069 ms (6.887 ms / 100)
  > Bad    7.925 -> 9.070 (+14.45%) [ +0.09%  +0.00%  +0.14% / +14.80% +14.66% +14.45%]     index_select spread            : Elapsed 0.079 ms (7.932 ms / 100)
  > Bad    7.910 -> 9.079 (+14.78%) [ +0.00%  +0.18%  +0.32% / +15.15% +14.78% +14.82%]     index_select strided 3         : Elapsed 0.079 ms (7.910 ms / 100)
  > good   8.019 -> 7.446 ( -7.15%) [ +0.02%  +0.00%  +0.15% /  -6.12%  -7.13%  -7.15%]     index_select strided 5         : Elapsed 0.080 ms (8.021 ms / 100)
  > Bad    8.031 -> 9.126 (+13.63%) [ +0.00%  +0.17%  +0.16% / +13.63% +14.59% +14.46%]     index_select strided 7         : Elapsed 0.080 ms (8.031 ms / 100)
  > Bad    7.917 -> 8.929 (+12.78%) [ +0.19%  +0.05%  +0.00% / +12.78% +16.36% +16.43%]     index_select strided 8         : Elapsed 0.079 ms (7.932 ms / 100)
  > Bad    7.942 -> 9.094 (+14.51%) [ +0.26%  +0.00%  +0.28% / +14.51% +14.54% +14.62%]     index_select strided 16        : Elapsed 0.080 ms (7.963 ms / 100)
  > Bad    7.945 -> 9.038 (+13.76%) [ +0.26%  +0.19%  +0.00% / +14.66% +13.76% +13.81%]     index_select strided 64        : Elapsed 0.080 ms (7.966 ms / 100)
  > good   6.923 -> 6.313 ( -8.81%) [ +0.27%  +0.00%  +0.01% /  -8.81%  -8.06%  -7.96%]     index_select strided 100       : Elapsed 0.069 ms (6.942 ms / 100)
  >        7.947 -> 8.188 ( +3.03%) [ +0.48%  +0.00%  +0.13% /  +6.37%  +3.36%  +3.03%]     index_select random            : Elapsed 0.080 ms (7.985 ms / 100)
  > bad    7.805 -> 8.209 ( +5.18%) [ +0.03%  +0.08%  +0.00% /  +8.03%  +5.18%  +5.24%]     index_select random_sorted     : Elapsed 0.078 ms (7.807 ms / 100)
  > Bad    8.005 -> 8.855 (+10.62%) [ +0.10%  +0.00%  +0.01% / +15.80% +10.82% +10.62%]     index_select perm              : Elapsed 0.080 ms (8.013 ms / 100)
  > Bad    7.967 -> 8.873 (+11.37%) [ +0.40%  +0.14%  +0.00% / +16.33% +11.37% +11.42%]     index_select perm_sorted       : Elapsed 0.080 ms (7.999 ms / 100)

Can be reproduced with this:

    B = torch.cuda.FloatTensor(250, 15, 50)
    B.transpose_(0, 2)
    B.transpose_(0, 1)
    print('B.size = {} stride = {}'.format(B.size(), B.stride()))
    # B.size = torch.Size([15, 50, 250]) stride = (50, 1, 750)

    A = torch.cuda.FloatTensor(150, 250, 15)
    A.transpose_(0, 1)
    A.transpose_(0, 2)
    print('A.size = {} stride = {}'.format(A.size(), A.stride()))
    # A.size = torch.Size([15, 150, 250]) stride = (1, 3750, 15)

    # index_select const: 69 -> 62 us
    idxs = np.zeros(50, dtype=np.int)
    A.index_select(1, torch.cuda.LongTensor(idxs), out=B)

    # index_select strided 5: 80 -> 74 us
    idxs = (np.arange(50) * 5) % 150
    A.index_select(1, torch.cuda.LongTensor(idxs), out=B)

    # index_select strided 16: 79 -> 91 us
    idxs = (np.arange(50) * 16) % 150
    A.index_select(1, torch.cuda.LongTensor(idxs), out=B)

    # index_select perm: 80 -> 89 us
    idxs = np.random.permutation(150)[:50]
    A.index_select(1, torch.cuda.LongTensor(idxs), out=B)

There are a few more isolated cases of "Bad" (>10%), but they don't seem as
interesting.

Examples of clear wins happen when "index stride" is small in both source and
destination:

  B = [512, 256] (stride (256, 1)) A = [512, 255] (stride (255, 1)) dim = 1
    ---> Scores between -9 .. -15% for various patterns

  B = [50, 250, 15] (stride (3750, 1, 250)) A = [50, 150, 15] (stride (2250, 1, 150)) dim = 1
    ---> Scores between -9 .. -18%

  B = [20, 40, 50, 256] (stride (2000, 50, 1, 40000)) A = [20, 40, 100, 256] (stride (40, 1, 800, 80000)) dim = 2
    ---> Scores between -58 .. -92% (i.e., 2x -- 13x speedup)
