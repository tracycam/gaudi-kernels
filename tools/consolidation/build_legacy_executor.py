#!/usr/bin/env python3
"""Build the repository-owned recorder shim; TPC ELFs remain byte-identical."""
import argparse
from pathlib import Path
import subprocess

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sdk-include',type=Path,default=Path('/usr/include/habanalabs'))
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[2]/'csrc/legacy_executor/e1_replay.cpp'
    command=['g++','-std=c++17','-O2','-shared','-fPIC','-pthread',
             '-I'+str(a.sdk_include),str(source),'-ldl','-o',str(a.out)]
    subprocess.run(command,check=True)
    symbols=subprocess.check_output(['nm','-D',str(a.out)],text=True)
    if any('getenv' in line for line in symbols.splitlines()):raise RuntimeError('Built runtime still reads environment')
    if 'e1_configure_options' not in symbols:raise RuntimeError('Explicit startup ABI absent')
