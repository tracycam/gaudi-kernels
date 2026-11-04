"""Build the generic dense TPC kernels as one Synapse provider."""

import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tpc-clang", default=shutil.which("tpc-clang") or "/opt/habanalabs/bin/tpc-clang")
    parser.add_argument("--glue-include", default="/usr/include/habanalabs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([args.tpc_clang, "-O2", "-ffp-contract=off", "-mcpu=gaudi2", "-c",
                    str(root / "csrc/tpc/generic_dense/add_bf16.c"), "-o", "add_bf16.o"],
                   cwd=out, check=True)
    subprocess.run(["objcopy", "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64",
                    "add_bf16.o", "add_bf16_x86.o"], cwd=out, check=True)
    subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-Wl,-z,defs",
                    "-Wl,-z,noexecstack", "-I" + args.glue_include,
                    str(root / "csrc/host/generic_dense_glue.cpp"), "add_bf16_x86.o",
                    "-o", "libgaudi_kernels_bundle.so"], cwd=out, check=True)


if __name__ == "__main__":
    main()
