"""One TPC database for small-M metadata, consumers and K8 decode.

Reuses the already qualified metadata/consumer object bytes. This avoids the
Synapse 1.24.1 external-library limit without changing their kernel GUIDs/ELFs.
The legacy core decoder is intentionally absent: this database is stream-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--route-build', type=Path, required=True)
p.add_argument('--stream-build', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
api = ('GetKernelGuids', 'InstantiateTpcKernel', 'GetShapeInference', 'GetSupportedDataLayout', 'GetLibVersion')
names = ['metadata_glue.o', 'tiles_glue.o']+[n+'_x86.o' for n in ('count', 'prefix', 'inverse', 'row_map', 'gather', 'gate', 'combine')]
identity = json.loads((root/'source-identity.json').read_text())
report = dict(source_commit=identity['git_commit'], input_sha256={}, commands=[], status='BUILDING')
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
for name in names:
    src = a.route_build/name
    shutil.copy2(src, out/name)
    report['input_sha256'][str(src)] = digest(src)
for name in ('stream_decode_x86.o', 'stream_decode.o', 'stream_decode.s', 'stream_decode.dis'):
    src = a.stream_build/name
    shutil.copy2(src, out/name)
    report['input_sha256'][str(src)] = digest(src)
wrapper = (root/'csrc/host/route_bundle_glue.cpp').read_text()
wrapper = wrapper.replace('DECLARE(core) DECLARE(metadata) DECLARE(tiles)', 'DECLARE(metadata) DECLARE(tiles) DECLARE(stream)')
wrapper = wrapper.replace('decltype(&gk_bundle_core_', 'decltype(&gk_bundle_metadata_')
wrapper = wrapper.replace('PROVIDER(core,7),PROVIDER(metadata,4),PROVIDER(tiles,3)', 'PROVIDER(metadata,4),PROVIDER(tiles,3),PROVIDER(stream,1)')
wrapper = wrapper.replace('std::array<GuidInfo,14>', 'std::array<GuidInfo,8>').replace('std::array<unsigned,14>', 'std::array<unsigned,8>')
(out/'stream_bundle.cpp').write_text(wrapper)
(out/'exports.map').write_text('{global: '+'; '.join(api)+'; local: *;};\n')
commands = [
    ['g++', '-O2', '-std=c++17', '-fPIC', '-fvisibility=hidden', '-I/usr/lib/habanatools/include',
     *['-D'+x+'=gk_bundle_stream_'+x for x in api], '-c', str(root/'csrc/host/moe_stream_glue.cpp'), '-o', 'stream_glue.o'],
    ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-Wl,--version-script=exports.map',
     '-I/usr/lib/habanatools/include', 'stream_bundle.cpp', *names, 'stream_glue.o', 'stream_decode_x86.o', '-o', 'libgaudi_moe_stream_bundle_tpc.so']]
try:
    with (out/'build.log').open('w') as log:
        for cmd in commands:
            report['commands'].append(cmd)
            subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    report.update(status='BUILT_NOT_DEVICE_QUALIFIED', library_sha256=digest(out/'libgaudi_moe_stream_bundle_tpc.so'))
finally:
    report['files_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    (out/'build.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
