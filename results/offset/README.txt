Benchmarks for using OffsetInfo: run on GTX 1080 and Ryzen 1600X.

clean.*.log : ran ptwise2.py on PyTorch master branch at 4ae0579
incr.*.log : run with OffsetInfo change.

SCORES.txt : comparison of the runs.
(Each side is run three times: we compare the best values (i.e., shortest
duration) of each side.)

Most cases are neutral, but some cases show >20% performance improvement.
Here's an example showing ~20% improvement (see ptwise3.py).

    A = torch.cuda.FloatTensor(1000, 256).fill_(1.0)
    B = torch.cuda.FloatTensor(128).fill_(1.01)
    A = A[:, :128]
    A.pow_(B)  # Improves from ~7.6 to ~6.0 usec

Unfortunately, a few cases show 5-6% performance hit, though some may be random
fluctuation.  (Performance routinely changes >5% even for the same version.)
Here is an example where performance "reliably" degrades ~5%:

    A = torch.cuda.IntTensor(2048, 2048)
    A = A[:, :2000]
    B = torch.cuda.IntTensor(2000).fill_(10)
    A.remainder_(B)  # Changes from ~180 to 189 usec.

Statistics:
  bad   :     4 (>5%)
        :  1399
  good  :   131 (5-10%)
  Good  :   120 (10-20%)
  GOOD  :    74 (>20%)
