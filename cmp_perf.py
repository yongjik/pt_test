#!/usr/bin/env python
#
# Compare the relative durations of test cases.

import argparse
import re
import sys

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--base', type=str, nargs='+')
arg_parser.add_argument('--exp', type=str, nargs='+')

argv = arg_parser.parse_args()

files = argv.base + argv.exp
base_cnt = len(argv.base)
assert len(files) >= 2
files = [open(f) for f in files]

stat = {}

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
    dur_vals = [float(d) for d in durations]

    base_min = min(dur_vals[:base_cnt])
    exp_min = min(dur_vals[base_cnt:])

    # Relative difference between the best base/exp times.
    best_diff = (exp_min - base_min) / base_min

    if best_diff >= 0.2:
        verdict = '0BAD'
    elif best_diff >= 0.1:
        verdict = '1Bad'
    elif best_diff >= 0.05:
        verdict = '2bad'
    elif best_diff <= -0.2:
        verdict = '6GOOD'
    elif best_diff <= -0.1:
        verdict = '5Good'
    elif best_diff <= -0.05:
        verdict = '4good'
    else:
        verdict = '3'

    stat[verdict] = stat.get(verdict, 0) + 1

    important = abs(best_diff) > 0.05

    diffs = [(d - base_min) / base_min for d in dur_vals]
    #diff_strs = ['%s=%+6.2f%%' % (d, diff * 100.0) for d, diff in zip(durations, diffs)]
    diff_strs = ['%+6.2f%%' % (diff * 100.0) for diff in diffs]

    print('%-5s  %.3f -> %.3f (%+6.2f%%) [%s / %s] %s' %
          (verdict[1:],
           base_min, exp_min, best_diff * 100.0,
           ' '.join(diff_strs[:base_cnt]),
           ' '.join(diff_strs[base_cnt:]),
           lines[0]))

sys.stderr.write('=====\n')
for key in sorted(stat):
    sys.stderr.write('  %-5s : %5d\n' % (key[1:], stat[key]))
