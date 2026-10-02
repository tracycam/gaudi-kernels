"""Device gate for Python ABI, plan roundtrip, fingerprints and busy eviction."""
import argparse
import array
import ctypes as C
import json
import os
import shutil
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--library', type=Path, required=True)
p.add_argument('--recipe', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--supplied', action='store_true')
a = p.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from gaudi_kernels.graph import Graph, Cache, GraphError, TEMP

sdk = C.CDLL('/usr/lib/habanalabs/libSynapse.so')
sdk.synInitialize.argtypes = []
sdk.synDeviceAcquireByModuleId.argtypes = [C.POINTER(C.c_uint32), C.c_uint32]
sdk.synDeviceRelease.argtypes = [C.c_uint32]
assert sdk.synInitialize() == 0
device = C.c_uint32()
assert sdk.synDeviceAcquireByModuleId(C.byref(device), int(os.environ['GAUDI_KERNELS_MODULE_ID'])) == 0
sdk.synDeviceMalloc.argtypes = [C.c_uint32, C.c_uint64, C.c_uint64, C.c_uint32, C.POINTER(C.c_uint64)]
sdk.synDeviceFree.argtypes = [C.c_uint32, C.c_uint64, C.c_uint32]
live_arenas, release_errors = set(), []
class Arena:
    def __init__(self, bytes):
        pointer=C.c_uint64()
        assert sdk.synDeviceMalloc(device,bytes,0,0,C.byref(pointer))==0
        self.address,self.bytes=pointer.value,bytes;live_arenas.add(self.address)
    def __del__(self):
        status=sdk.synDeviceFree(device,self.address,0)
        if status:release_errors.append(status)
        live_arenas.remove(self.address)
def allocator(bytes):
    owner=Arena(bytes)
    return owner.address,owner.bytes,owner
pool_allocator=allocator if a.supplied else None
N, size = 6144, 24576
g = Graph(a.library, a.output / 'built', device=device.value)
g.static_input('input', bytes=size).static_output('output', bytes=size)
g.buffer('a', bytes=size, role=TEMP).buffer('b', bytes=size, role=TEMP)
g.recipe_load('one', a.recipe)
g.launch('one', {'v0': ('input', 0), 'v1': ('a', 0)})
for i in range(1, 7):
    src, dst = ('a', 'b') if i % 2 else ('b', 'a')
    g.launch('one', {'v0': (src, 0), 'v1': (dst, 0)})
g.launch('one', {'v0': ('a', 0), 'v1': ('output', 0)}).instantiate(bucket=[1, 8],pool_allocator=pool_allocator)
g.save(a.output / 'plan')
g.close()
records = []
cache = Cache(max_graphs=1)
for phase in range(2):
    g = cache.get(phase, lambda: Graph.load(a.library, a.output / 'plan', a.output / f'loaded-{phase}', device=device.value,pool_allocator=pool_allocator))
    flights = []
    for mutation in range(4):
        x = array.array('f', ((i * 17 + mutation * 7 + phase * 19) % 131 / 32 for i in range(N)))
        expected = array.array('f', (v + .25 for v in x)).tobytes()
        flight = g.replay({'input': x})
        flights.append((flight, expected))
        x[:] = array.array('f', [12345] * N)
    try:
        cache.get('busy', lambda: (_ for _ in ()).throw(AssertionError('builder must not run')))
        raise AssertionError('busy eviction accepted')
    except GraphError as e:
        assert e.status == 8
    for mutation, (f, expected) in enumerate(flights):
        assert f.wait().query()
        assert f.result('output') == expected
        f.release()
        records.append(dict(stage='python_roundtrip', phase=phase, mutation=mutation, checked=N, bad=0))
    g.report(a.output / f'loaded-{phase}.json')
g.close()
for kind in ('fingerprint', 'recipe'):
    directory = a.output / f'bad-{kind}'
    shutil.copytree(a.output / 'plan', directory)
    if kind == 'fingerprint':
        manifest = json.loads((directory / 'plan.json').read_text())
        manifest['fingerprint']['driver'] = 'deliberate-mismatch'
        (directory / 'plan.json').write_text(json.dumps(manifest))
    else:
        path = directory / 'recipe-0.bin'
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    try:
        Graph.load(a.library, directory, a.output / f'rejected-{kind}', device=device.value,pool_allocator=pool_allocator)
        raise AssertionError('invalid plan accepted')
    except ValueError as e:
        records.append(dict(stage='python_rejection', kind=kind, reason=str(e)))
assert not live_arenas and not release_errors, (live_arenas,release_errors)
assert sdk.synDeviceRelease(device) == 0
assert sdk.synDestroy() == 0
(a.output / 'python.json').write_text(json.dumps(records, indent=2) + '\n')
