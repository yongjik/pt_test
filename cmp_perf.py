#!/usr/bin/env python
#
# Compare the relative durations of test cases.

import re
import sys

files = sys.argv[1:]
assert len(files) >= 2
files = [open(f) for f in files]

while True:
    lines = [f.readline() for f in files]

    is_eof = sum(line == '' for line in lines)
    assert is_eof in (0, len(files)), lines
    if is_eof: break

    lines = [line.rstrip('\n') for line in lines]

    pat = r'Elapsed.*\(([0-9.]+) ms'
    durations = [re.search(pat, line) for line in lines]

    assert sum(dur is None for dur in durations) in (0, len(files)), lines

    if durations[0] is None:
        print(lines[0])
        continue

    durations = [dur.group(1) for dur in durations]
    base = float(durations[0])
    diffs = [(float(dur) - base) / base for dur in durations[1:]]
    important = any(abs(diff) > 0.1 for diff in diffs)

    print('%5s  %7s -> %s (%s) %s' %
          ('!!!!!' if important else '     ',
           base,
           ' '.join('%7s' % dur for dur in durations[1:]),
           ' '.join('%+6.2f%%' % (diff * 100.0) for diff in diffs),
           lines[0]))
