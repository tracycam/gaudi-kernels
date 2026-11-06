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
    objects = []
    for name, source in (("add_bf16", "generic_dense/add_bf16.c"),
                         ("fused_add_rms_norm_bf16", "generic_dense/fused_add_rms_norm_bf16.c"),
                         ("rms_norm_bf16", "generic_dense/rms_norm_bf16.c"),
                         ("silu_and_mul_bf16", "generic_dense/silu_and_mul_bf16.c"),
                         ("kv_store_bf16", "generic_dense/kv_store_bf16.c"),
                         ("rotary_bf16", "generic_dense/rotary_bf16.c"),
                         ("paged_metadata_i32", "generic_dense/paged_metadata_i32.c"),
                         ("token_indices_i32", "generic_dense/token_indices_i32.c"),
                         ("cast_f32_to_bf16", "generic_dense/cast_f32_to_bf16.c"),
                         ("index_put_i64", "generic_dense/index_put_i64.c")):
        subprocess.run([args.tpc_clang, "-O2", "-ffp-contract=off", "-mcpu=gaudi2", "-c",
                        str(root / f"csrc/tpc/{source}"), "-o", f"{name}.o"],
                       cwd=out, check=True)
        subprocess.run(["objcopy", "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64",
                        f"{name}.o", f"{name}_x86.o"], cwd=out, check=True)
        objects.append(f"{name}_x86.o")
    subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-Wl,-z,defs",
                    "-Wl,-z,noexecstack", "-I" + args.glue_include,
                    str(root / "csrc/host/generic_dense_glue.cpp"), *objects,
                    "-o", "libgaudi_kernels_bundle.so"], cwd=out, check=True)


if __name__ == "__main__":
    main()
