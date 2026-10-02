#!/usr/bin/env python3
"""Eight independent CPU processes test the FMA helper; no Torch/HPU imports."""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import struct


def bits(value):
    return struct.unpack('I', struct.pack('f', value))[0]


_barrier = None


def initialize(barrier):
    global _barrier
    _barrier = barrier


def worker(arguments):
    path, rank = arguments
    _barrier.wait(timeout=30)
    library = ctypes.CDLL(path)
    function = library.gk_fp32_oracle_fma
    pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [pointer] * 4 + [ctypes.c_uint64]
    function.restype = ctypes.c_int
    math = ctypes.CDLL('libm.so.6')
    math.fegetround.restype = ctypes.c_int
    math.fmaf.argtypes = [ctypes.c_float] * 3
    math.fmaf.restype = ctypes.c_float
    assert math.fegetround() == 0  # glibc FE_TONEAREST on this target.
    cases = [(1 + 2 ** -23, 1 - 2 ** -23, -1), (2 ** -126, .5, 0),
             (-0., 1., -0.), (1., 2., 3.), (2 ** 24, 1., 1.)]
    # Reproducible finite operands spanning signs, exponents and mantissas.
    for i in range(1024):
        a = (1 + ((i * 17 + rank) % 128) / 128) * 2.0 ** ((i % 61) - 30)
        b = (1 + ((i * 31 + rank) % 128) / 128) * 2.0 ** (((i // 61) % 31) - 15)
        c = (-1 if i & 1 else 1) * 2.0 ** ((i % 41) - 20)
        cases.append((a, b, c))
    arrays = [(ctypes.c_float * len(cases))(*(row[j] for row in cases)) for j in range(3)]
    output = (ctypes.c_float * len(cases))()
    assert function(*arrays, output, len(cases)) == 0
    expected = [math.fmaf(*row) for row in cases]
    assert [bits(v) for v in output] == [bits(v) for v in expected]
    assert output[0] == -2 ** -46 and bits(output[1]) == 0x00400000
    return dict(worker=rank, pid=os.getpid(), fe_round=math.fegetround(), cases=len(cases), all_bits_equal=True,
                fma_cancellation_bits=bits(output[0]), gradual_underflow_bits=bits(output[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    path = str(args.library.resolve(strict=True))
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=context,
                                                initializer=initialize, initargs=(context.Barrier(8),)) as pool:
        rows = list(pool.map(worker, [(path, rank) for rank in range(8)]))
    assert len({r['pid'] for r in rows}) == 8
    result = dict(status='PASS_CPU_ONLY', library_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), workers=rows,
                  scope='eight CPU subprocesses, not HPU TP8 workers; production helper rechecks FE_TONEAREST on every call',
                  imported_torch_or_hpu=False, device_acquired=False)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
