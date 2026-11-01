"""Build model-owned providers into one library for Synapse's 15-library limit."""
import argparse
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--tpc-clang', default=shutil.which('tpc-clang') or '/opt/habanalabs/bin/tpc-clang')
parser.add_argument('--glue-include', type=Path, default=Path('/usr/include/habanalabs'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
objects = []
kernels = [
    ('prefill_softmax', 'full_mme_attention/prefill_softmax.c'),
    ('swa_mme_gather', 'swa_prefill_mme/gather.c'),
    ('swa_mme_softmax', 'swa_prefill_mme/softmax.c'),
    ('router_post', 'moe_prefill/router_post.c'),
    ('gate', 'mimo_moe_gate_fp32/main.c'),
    ('combine', 'mxfp4_moe/legacy/combine_f32.c'),
    ('swiglu', 'dense_prefill/swiglu.c'),
    ('residual_rmsnorm', 'dense_fp32/residual_rmsnorm.c'),
    ('linear', 'dense_prefill/linear.c'),
    ('dense_weight_split', 'dense_prefill/weight_split.c'),
    ('mimo_embedding', 'mimo_step/embedding.c'),
    ('mimo_argmax_parts', 'mimo_step/argmax_parts.c'),
    ('mimo_argmax_finish', 'mimo_step/argmax_finish.c'),
    ('reduce_compensated', 'block_fp8_reduce_isa/round_bf16_compensated.c'),
]
for name, source in kernels:
    subprocess.run([args.tpc_clang, '-O2', '-ffp-contract=off', '-mcpu=gaudi2', '-c',
                    str(root / 'csrc/tpc' / source), '-o', name + '.o'], cwd=out, check=True)
    subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                    name + '.o', name + '_x86.o'], cwd=out, check=True)
    objects.append(name + '_x86.o')
partition = ('prefix', 'prefix_split', 'inverse', 'gate', 'combine', 'decode', 'sparse_map', 'counted_gather', 'count')
for name in partition:
    source = root / 'csrc/tpc/moe_partition' / (name + '.c')
    obj = 'pc_' + name + '.o'
    flags = ['-DGK_ROUTE_DECODE_SCALE_BITS=1'] if name == 'decode' else []
    subprocess.run([args.tpc_clang, '-O2', '-ffp-contract=off', '-mcpu=gaudi2', *flags,
                    '-c', str(source), '-o', obj], cwd=out, check=True)
    subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
                    obj, 'pc_' + name + '_x86.o'], cwd=out, check=True)
    objects.append('pc_' + name + '_x86.o')
symbols = ('GetKernelGuids', 'InstantiateTpcKernel', 'GetShapeInference',
           'GetSupportedDataLayout', 'GetLibVersion')
for prefix, glue in [('swa_mme', 'swa_prefill_mme_glue.cpp'), ('full', 'full_mme_prefill_softmax_glue.cpp'),
                     ('moe', 'moe_prefill_glue.cpp'), ('dense', 'dense_prefill_glue.cpp'),
                     ('dense_split', 'dense_weight_split_glue.cpp'),
                     ('embedding', 'mimo_embedding_glue.cpp'), ('argmax', 'mimo_argmax_glue.cpp'),
                     ('qkv', 'block_fp8_reduce_compensated_glue.cpp'),
                     ('part', 'moe_partition_glue.cpp')]:
    obj = prefix + '_glue.o'
    subprocess.run(['g++', '-O2', '-std=c++17', '-fPIC', f'-I{args.glue_include}',
                    *[f'-D{symbol}={prefix}_{symbol}' for symbol in symbols],
                    *(['-DGAUDIRT_PARTITION_PREFILL_ONLY'] if prefix == 'part' else []), '-c',
                    str(root / 'csrc/host' / glue), '-o', obj], cwd=out, check=True)
    objects.append(obj)
subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs',
                '-Wl,-z,noexecstack', f'-I{args.glue_include}',
                str(root / 'csrc/host/mimo_prefill_bundle.cpp'), *objects,
                '-o', 'libgaudi_mimo_prefill_bundle_tpc.so'], cwd=out, check=True)
