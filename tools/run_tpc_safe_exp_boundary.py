"""Compile and run the Gaudi2 MoE gate's 750-boundary regression."""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaudirt-source', type=Path, required=True)
    parser.add_argument('--gaudirt-build', type=Path, required=True)
    parser.add_argument('--provider', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    binary = args.output_dir.resolve() / 'tpc_safe_exp_boundary'
    subprocess.run(['g++', '-O2', '-std=c++20',
                    '-I' + str((args.gaudirt_source / 'include').resolve()),
                    '-I/usr/include/habanalabs',
                    str(root / 'tests/tpc_safe_exp_boundary.cpp'),
                    str((args.gaudirt_build / 'libgaudirt.a').resolve()),
                    '-L/usr/lib/habanalabs', '-Wl,-rpath,/usr/lib/habanalabs',
                    '-lSynapse', '-lhcl', '-lpthread', '-o', str(binary)], check=True)
    subprocess.run(['hl-smi'], check=True)
    if not Path('/dev/accel/accel0').exists():
        raise RuntimeError('Gaudi module 0 device node is missing')
    users = subprocess.run(['fuser', '/dev/accel/accel0'], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL).returncode
    if users == 0:
        raise RuntimeError('Gaudi module 0 is in use')
    if users != 1:
        raise RuntimeError('unable to inspect Gaudi module 0 ownership')
    env = {'PATH': os.defpath, 'LD_LIBRARY_PATH': '/usr/lib/habanalabs',
           'HABANA_LOGS': str(args.output_dir.resolve()),
           'GC_KERNEL_PATH': str(args.provider.resolve()) + ':/usr/lib/habanalabs/libtpc_kernels.so'}
    subprocess.run([str(binary)], env=env, check=True)


if __name__ == '__main__':
    main()
