"""Build/resolve an exact-runtime-pinned view helper, without HPU tensors.

Never changes the installed bridge. Pin arguments must come from a recorded
inspection of the intended Lazy library, not a different version's headers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--runtime-library', type=Path, required=True)
p.add_argument('--runtime-sha256', required=True)
p.add_argument('--runtime-build-id', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = a.output_dir.resolve()
out.mkdir(parents=True, exist_ok=False)
identity = root / 'source-identity.json'
commit = (json.loads(identity.read_text())['git_commit'] if identity.exists() and not (root/'.git').exists()
          else subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
sources = ['csrc/torch/experimental_view_safe.cpp', 'csrc/torch/experimental_view_safe_probe.cpp']
declared = sources + ['csrc/torch/experimental_view_safe.hpp', 'tools/build_view_safe_experiment.py']
meta = {'commit': commit, 'state': 'building', 'scope': 'CPU build/link/symbol resolution only',
        'source_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in declared},
        'device_tensor_created': False, 'device_acquire_called': False}
def save():
    (out/'build.json').write_text(json.dumps(meta, indent=2)+'\n')
save()
try:
    library = a.runtime_library.resolve(strict=True)
    actual_sha = hashlib.sha256(library.read_bytes()).hexdigest()
    notes = subprocess.check_output(['readelf', '-n', str(library)], text=True)
    match = re.search(r'Build ID: ([0-9a-f]+)', notes)
    if actual_sha != a.runtime_sha256 or not match or match[1] != a.runtime_build_id:
        raise RuntimeError('recorded runtime fingerprint mismatch before importing Torch')
    meta['runtime'] = {'path': str(library), 'sha256': actual_sha, 'build_id': match[1]}
    pin = ('#pragma once\nnamespace gaudi_kernels::experimental {\n' +
           ''.join('inline constexpr const char* '+name+' = '+json.dumps(value)+';\n' for name, value in (
               ('kViewSafeRuntimePath', str(library)), ('kViewSafeRuntimeSha256', actual_sha),
               ('kViewSafeRuntimeBuildId', match[1]))) + '}\n')
    (out/'view_safe_runtime_pin.h').write_text(pin)
    meta['pin_header_sha256'] = hashlib.sha256(pin.encode()).hexdigest()
    os.environ['PT_HPU_LAZY_MODE'] = '1'
    os.environ.setdefault('MAX_JOBS', '1')
    import torch
    import habana_frameworks.torch as ht
    from torch.utils.cpp_extension import load
    h = Path(ht.__file__).parent
    meta['torch'] = torch.__version__
    save()
    load(name='gaudi_view_safe_experiment', sources=[str(root/n) for n in sources],
         extra_include_paths=[str(h/'include'), str(out)], extra_cflags=['-O2'],
         extra_ldflags=[f'-L{h}/lib', f'-Wl,-rpath,{h}/lib', '-lhabana_pytorch_plugin', '-ldl'],
         build_directory=str(out), is_python_module=False, verbose=True)
    meta['runtime_resolution'] = torch.ops.gaudi_view_safe.runtime_pin()
    # Resolution and pass-through of CPU values exercise no HPU handler calls.
    cpu = [torch.arange(8, dtype=t) for t in (torch.int32, torch.int64, torch.float32, torch.bfloat16)]
    outputs = torch.ops.gaudi_view_safe.normalize_list(cpu)
    assert all(torch.equal(x, y) and x.dtype == y.dtype and x.data_ptr() == y.data_ptr()
               for x, y in zip(cpu, outputs))
    meta['cpu_list_pass_through'] = 'PASS: values/dtype/address unchanged; handler was not called for CPU tensors'
    meta['state'] = 'BUILT_RESOLVED_CPU_ONLY_NOT_DEVICE_VALIDATED'
    meta['binary_sha256'] = hashlib.sha256((out/'gaudi_view_safe_experiment.so').read_bytes()).hexdigest()
except Exception as error:
    meta.update(state='failed', error=str(error))
    raise
finally:
    save()
print(json.dumps(meta, indent=2))
