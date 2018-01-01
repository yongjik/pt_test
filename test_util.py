from __future__ import print_function
import hashlib
import math
import struct
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

    total_duration = (T1 - T0) * 1000.0
    duration = total_duration / rep_count

    if len(args) >= 2:
        print('    %-30s : Elapsed %.3f ms (%.3f ms / %d)' %
              (' '.join(args[:-1]), duration, total_duration, rep_count))
    else:
        print('    Elapsed %.3f ms (%.3f ms / %d)' %
              (duration, total_duration, rep_count))
    return duration

# Helper function for generatic a stable pseudorandom value in [0.0, 1.0): it is
# used to designate a predefined portion of the training data for testing.
def stable_pseudorandom(s):
    h = hashlib.md5()
    h.update(bytearray('~~Random prefix~~' + s, 'utf-8'))

    val, = struct.unpack_from('Q', h.digest())
    val *= 1e-15
    val -= math.floor(val)

    assert 0.0 <= val < 1.0
    return val
