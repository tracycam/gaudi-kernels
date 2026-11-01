"""Build the two M=1 expert TPC kernels without loading a device or bridge."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tpc-clang', required=True)
    parser.add_argument('--glue-include', default='/usr/include/habanalabs')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir.resolve()
    sources = (
        ('mimo_compact_gate', 'csrc/tpc/mimo_moe_gate_fp32/main.c'),
        ('mimo_combine_f32', 'csrc/tpc/mxfp4_moe/legacy/combine_f32.c'),
    )
    for name, path in sources:
        subprocess.run([args.tpc_clang, '-O2', '-mcpu=gaudi2', '-c', str(root / path),
                        '-o', name + '.o'], cwd=output, check=True)
        subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B',
                        'i386:x86-64', name + '.o', name + '_x86.o'],
                       cwd=output, check=True)
    subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs',
                    '-Wl,-z,noexecstack', '-I' + args.glue_include,
                    str(root / 'csrc/host/mimo_moe_fp32_provider_glue.cpp'),
                    *(name + '_x86.o' for name, _ in sources),
                    '-o', 'libgaudi_mimo_moe_fp32_provider_tpc.so'],
                   cwd=output, check=True)


if __name__ == '__main__':
    main()
