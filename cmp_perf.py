#!/usr/bin/env python
#
# Compare the relative durations of test cases.

import re
import sys

file1, file2 = sys.argv[1:]
file1, file2 = open(file1), open(file2)

while True:
    line1 = file1.readline()
    line2 = file2.readline()

    assert (line1 == '') == (line2 == '')
    if line1 == '': break

    line1 = line1.rstrip('\n')
    line2 = line2.rstrip('\n')

    pat = r'Elapsed.*\(([0-9.]+) ms'
    dur1 = re.search(pat, line1)
    dur2 = re.search(pat, line2)

    if dur1 is None:
        assert dur1 == dur2
        print(line1)
    else:
        assert dur2 is not None
        dur1 = dur1.group(1)
        dur2 = dur2.group(1)
        diff = (float(dur2) - float(dur1)) / float(dur1)

        important = abs(diff) > 0.1

        print('%5s  %7s -> %7s (%+6.2f%%) %s' %
              ('!!!!!' if important else '     ',
               dur1, dur2, diff * 100.0, line1))
