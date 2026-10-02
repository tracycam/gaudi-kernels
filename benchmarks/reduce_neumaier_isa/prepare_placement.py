"""Copy exact sealed assets, or validate Meta registration in target Torch (CPU only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument('--assets', type=Path, required=True)
p.add_argument('--sealed-remote', type=Path)
p.add_argument('--verify-target-meta', action='store_true')
a = p.parse_args()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if a.sealed_remote:
    a.assets.mkdir(parents=True, exist_ok=False)
    sources = {'old-torch.so': 'runtime/block_fp8_reduce_experiment.so',
               'old-tpc.so': 'runtime/libblock_fp8_reduce_experiment.so',
               'hand-torch.so': 'builds/torch/gk_reduce_neumaier_isa_torch.so',
               'hand-tpc.so': 'builds/tpc/libreduce_neumaier_isa.so',
               'block-torch.so': 'runtime/gaudi_block_fp8_torch.so',
               'block-tpc.so': 'runtime/libgaudi_block_fp8_framework_tpc.so',
               'counter.so': 'runtime/libcount_launch.so',
               **{name+'.pt': 'inputs/isa-public/'+name+'.pt' for name in ('qkv-measured-m1', 'single-group-m16')}}
    for name, relative in sources.items():
        shutil.copy2(a.sealed_remote / relative, a.assets / name)
    manifest = dict(device_accessed=False, original_asset_paths=sources,
                    source_archive='artifacts/builds/reduce-neumaier-isa-public/device-f1fe27e-c94fddc/remote',
                    files_sha256={name: sha(a.assets/name) for name in sources})
    (a.assets / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
manifest = json.loads((a.assets/'manifest.json').read_text())
for name, digest in manifest['files_sha256'].items():
    assert sha(a.assets/name) == digest, name
if a.verify_target_meta:
    # These sealed bridges were built for the production lazy runtime. Loading
    # them in eager mode can register both backends before any device operation.
    os.environ['PT_HPU_LAZY_MODE'] = '1'
    import torch
    import habana_frameworks.torch.core  # registration only; all tensors stay on CPU/Meta
    for name in ('old-torch.so', 'hand-torch.so', 'block-torch.so'):
        torch.ops.load_library(str((a.assets/name).resolve()))
    records = []
    for name in ('qkv-measured-m1', 'single-group-m16'):
        f = torch.load(a.assets/(name+'.pt'), map_location='cpu', weights_only=False)
        m, n, k = f['shape']
        x = torch.empty((m,k), dtype=torch.bfloat16, device='meta')
        w = torch.empty(f['weight_bits'].shape, dtype=torch.float8_e4m3fn, device='meta')
        sw = torch.empty(f['weight_scales'].shape, dtype=torch.float32, device='meta')
        bias = torch.empty((n,), dtype=torch.float32, device='meta')
        q, sa = torch.ops.gaudi_block_fp8.quant(x)
        assert tuple(q.shape) == tuple(f['samples'][0]['quant_bits'].shape)
        assert tuple(sa.shape) == tuple(f['samples'][0]['activation_scales'].shape)
        part = torch.ops.gaudi_block_fp8.batch_mm(q,w)
        for variant, op in [('old',torch.ops.gk_reduce_experiment.neumaier),('hand',torch.ops.gk_reduce_isa.handschedule)]:
            y = op(part,sa,sw,bias)
            assert y.device.type == 'meta' and tuple(y.shape) == (m,n) and y.dtype == torch.bfloat16
            records.append(dict(case=name,variant=variant,output_shape=list(y.shape)))
    report = dict(device_accessed=False,torch_version=torch.__version__,meta_checks=records,
                  asset_manifest_sha256=sha(a.assets/'manifest.json'),all_pass=True)
    (a.assets/'target-meta.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
else:
    print(json.dumps(dict(device_accessed=False,files_verified=len(manifest['files_sha256']))))
