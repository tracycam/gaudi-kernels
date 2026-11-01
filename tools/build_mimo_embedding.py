"""Build the MiMo M=1 embedding TPC library without acquiring a device."""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--glue-include', default='/usr/lib/habanatools/include')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir.resolve()
    commands = [
        ['/usr/bin/tpc-clang', '-O2', '-mcpu=gaudi2', '-c',
         str(root / 'csrc/tpc/mimo_step/embedding.c'), '-o', 'mimo_embedding.o'],
        ['objcopy', '-I', 'binary', '-O', 'elf64-x86-64', '-B', 'i386:x86-64',
         'mimo_embedding.o', 'mimo_embedding_x86.o'],
        ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-Wl,-z,defs', '-Wl,-z,noexecstack',
         '-I' + args.glue_include, str(root / 'csrc/host/mimo_embedding_glue.cpp'),
         'mimo_embedding_x86.o', '-o', 'libgaudi_mimo_embedding_tpc.so'],
    ]
    for command in commands:
        subprocess.run(command, cwd=output, check=True)


if __name__ == '__main__':
    main()
