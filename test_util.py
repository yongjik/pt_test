from __future__ import print_function
import time
import traceback

import torch

rep_count = 100

def try_cmd(cmd_id, cb):
    try:
        cb()
        return True
    except:
        print('\n========== Error in ', cmd_id)
        traceback.print_exc()
        return False

def time_cuda(*args):
    cb = args[-1]
    assert callable(cb)

    torch.cuda.synchronize()
    T0 = time.time()
    for _ in range(rep_count):
        cb()
    torch.cuda.synchronize()
    T1 = time.time()

    duration = (T1 - T0) / rep_count * 1000.0

    if len(args) >= 2:
        print('    %-30s : Elapsed %.3f ms' % (' '.join(args[:-1]), duration))
    else:
        print('    Elapsed %.3f ms' % duration)
    return duration
