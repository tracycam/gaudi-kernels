#!/usr/bin/env python3
"""Build a binding manifest from a frozen control; never modify source files."""
import argparse
import hashlib
import json
from pathlib import Path

from gaudi_kernels.engine import EngineConfig
from gaudi_kernels.engine.selection import PolicySelection

GROUPS = {'GK_BLOCK_FP8': 'block_fp8', 'GK_BLOCK_REDUCE_ISA': 'reduce_isa',
          'GK_NORM': 'norm', 'GK_NORM_GRID24': 'norm_grid24', 'GK_SWA': 'swa',
          'GK_SWA_AV': 'swa_av', 'GK_QKV_POST': 'qkv_post', 'GK_ROUTER_POST': 'router_post',
          'GK_MXFP4_FOLDED': 'folded', 'GK_MXFP4_SCALE_TAIL': 'scale_tail', 'GK_MXFP4_DOWN': 'down'}


def prepare(control, output, run_dir, replay_library, layers=70):
    baseline = json.loads((control / 'baseline.json').read_text())
    env = baseline['environment']
    leaf = control / 'runtime/executor/executor'
    entries = {}
    def add(name, path):
        path = Path(path).resolve(strict=True)
        entries[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for prefix, group in GROUPS.items():
        for kind in ('TPC', 'TORCH'):
            key = prefix + '_' + kind + '_LIBRARY'
            # The control was mirrored locally but its metadata names remote
            # paths. On the device host the paths resolve without any rewrite.
            if key in env: add(group + '.' + kind.lower(), env[key])
    for name, path in [('native.torch', leaf/'torch-build/native_mxfp4_ops.so'),
                       ('native.tpc', leaf/'tpc/libnative_tpc.so'),
                       ('batch.torch', leaf.parent/'ops-build/unified_batch_ops.so'),
                       ('batch.tpc', leaf.parent/'tpc/libbatch_tpc.so'),
                       ('precision.torch', leaf.parent/'precision-ops-build/precision_ops.so'),
                       ('precision.tpc', leaf.parent/'precision-tpc/libprecision_tpc.so'),
                       ('tensor_hold.host', leaf/'libe1_tensor_hold.so'),
                       ('replay.host', replay_library),
                       ('vendor.tpc', Path('/usr/lib/habanalabs/libtpc_kernels.so'))]: add(name, path)
    if env.get('GK_FP32_ORACLE_LIBRARY'):
        add('cpu_oracle.host', env['GK_FP32_ORACLE_LIBRARY'])
        add('cpu_oracle_build.host', env.get('GK_FP32_ORACLE_BUILD_JSON',str(Path(env['GK_FP32_ORACLE_LIBRARY']).with_name('build.json'))))
    # Match the qualified compiler database search order exactly. Use observed
    # inventory instead of inferring executed GUIDs from enabled flags.
    inventory = json.loads((leaf/'typed-70g/runtime-inventory/rank0.json').read_text())
    order = [r['path'] for r in inventory['tpc_libraries']]
    by_path = {entry['path']: name for name, entry in entries.items() if name.endswith('.tpc')}
    if set(order) != set(by_path): raise ValueError('Control TPC registration coverage differs')
    artifacts = {by_path[path]: entries[by_path[path]] for path in order}
    artifacts.update({name: entry for name, entry in entries.items() if not name.endswith('.tpc')})
    initial = EngineConfig.from_dict(json.loads((Path(__file__).resolve().parents[2]/'config/70g-reference.json').read_text()))
    document = {'schema_version': 1, 'startup': {'run_dir': str(run_dir), 'layers': layers,
        'keep_cpu_oracle': True, 'same_input_audit': True, 'layer_diagnostics': True,
        'inventory': True, 'profile_config': env.get('E1_PROFILE_CONFIG')}, 'artifacts': artifacts, 'selection': PolicySelection(initial).to_dict()}
    output.write_text(json.dumps(document, indent=2)+'\n')
    return document


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--replay-library',type=Path,required=True)
    p.add_argument('--layers',type=int,default=70)
    a=p.parse_args();prepare(a.control,a.out,a.run_dir,a.replay_library,a.layers)
