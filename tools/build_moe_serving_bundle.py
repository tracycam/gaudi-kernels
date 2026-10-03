"""Package the frozen batch provider and new expert provider into one SDK slot."""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_SHA = 'e9447e75356498df226b7d7daa3c2dac3f20efbab32834fc030000e61bdb68e2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--expert-build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'building', 'TPC_compiler_invoked': False, 'inputs': {}}
    try:
        assert digest(args.batch) == BATCH_SHA, 'qualified batch provider required'
        build = json.loads((args.expert_build/'build.json').read_text())
        for name in ('libgaudi_expert_partition_tpc.so', 'gaudi_expert_partition.so'):
            assert digest(args.expert_build/name) == build['files_sha256'][name]
        for name, source in [('provider_batch.so',args.batch),
                             ('provider_expert.so',args.expert_build/'libgaudi_expert_partition_tpc.so'),
                             ('gaudi_expert_partition.so',args.expert_build/'gaudi_expert_partition.so')]:
            shutil.copy2(source,args.output/name)
            report['inputs'][name] = {'sha256':digest(source),'path':str(source.resolve())}
        command = ['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs',
                   '-I/usr/include/habanalabs',str(ROOT/'csrc/host/moe_serving_bundle.cpp'),
                   '-ldl','-o',str(args.output/'libgaudi_moe_serving_tpc.so')]
        report['command'] = command
        with (args.output/'build.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=log,check=True,timeout=90)
        class Guid(ctypes.Structure):
            _fields_=[('name',ctypes.c_char*64),('properties',ctypes.c_ubyte*16)]
        handle=ctypes.CDLL(str((args.output/'libgaudi_moe_serving_tpc.so').resolve()))
        fn=handle.GetKernelGuids
        fn.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(Guid)]
        count=ctypes.c_uint32(0)
        assert fn(3,ctypes.byref(count),None)==0 and count.value==19
        items=(Guid*20)();ctypes.memset(ctypes.addressof(items),0x5a,ctypes.sizeof(items))
        count.value=19;assert fn(3,ctypes.byref(count),items)==0
        assert bytes(items[19])==b'Z'*80, 'capacity overflow'
        names=[bytes(x.name).decode() for x in items[:19]]
        assert len(set(names))==19
        report.update(state='built_cpu_registry_verified',guids=names,expert_source_commit=build['source_commit'])
    finally:
        report['files_sha256']={p.name:digest(p) for p in args.output.iterdir() if p.is_file() and p.name!='build.json'}
        (args.output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
