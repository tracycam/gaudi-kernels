"""Build the MiMo M=1 greedy argmax TPC library without acquiring a device."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--glue-include', default='/usr/lib/habanatools/include')
    parser.add_argument('--tpc-clang', default=shutil.which('tpc-clang') or '/opt/habanalabs/bin/tpc-clang')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir.resolve()
    objects = []
    for name in ('parts', 'finish', 'batch_parts', 'batch_finish'):
        obj = f'mimo_argmax_{name}.o'
        host_obj = f'mimo_argmax_{name}_x86.o'
        subprocess.run([args.tpc_clang, '-O2', '-mcpu=gaudi2', '-c',
                        str(root / f'csrc/tpc/mimo_step/argmax_{name}.c'), '-o', obj],
                       cwd=output, check=True)
        subprocess.run(['objcopy', '-I', 'binary', '-O', 'elf64-x86-64',
                        '-B', 'i386:x86-64', obj, host_obj], cwd=output, check=True)
        objects.append(host_obj)
    subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs',
                    '-Wl,-z,noexecstack', '-I' + args.glue_include,
                    str(root / 'csrc/host/mimo_argmax_glue.cpp'), *objects,
                    '-o', 'libgaudi_mimo_argmax_tpc.so'], cwd=output, check=True)


if __name__ == '__main__':
    main()
