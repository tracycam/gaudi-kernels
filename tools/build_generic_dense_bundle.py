"""Build the generic dense TPC kernels as one Synapse provider."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def verify_tpc_abi(obj, source, expected):
    # Tensor arguments determine TPC descriptor slots. An ELF and Glue with
    # different input counts write through unbound slots; verify both the
    # embedded source and the declared contract for the particular kernel.
    with tempfile.TemporaryDirectory() as temp:
        embedded = Path(temp) / "source.c"
        copied = Path(temp) / "object.o"
        subprocess.run(["objcopy", "--dump-section", f".source={embedded}", str(obj), str(copied)],
                       check=True)
        data = embedded.read_bytes()
    if data != source.read_bytes():
        raise RuntimeError("TPC object does not contain the requested source")
    match = re.search(rb"\bvoid\s+main\s*\((.*?)\)\s*\{", data, re.DOTALL)
    args = [b" ".join(arg.split()) for arg in match.group(1).split(b",")] if match else []
    if args != expected:
        raise RuntimeError("TPC ELF tensor arguments do not match its Glue contract")


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
                         ("kv_store_head_major_bf16", "generic_dense/kv_store_head_major_bf16.c"),
                         ("kv_store_head_major64_bf16", "generic_dense/kv_store_head_major64_bf16.c"),
                         ("rotary_bf16", "generic_dense/rotary_bf16.c"),
                         ("paged_metadata_i32", "generic_dense/paged_metadata_i32.c"),
                         ("paged_alloc_i32", "generic_dense/paged_alloc_i32.c"),
                         ("paged_free_i32", "generic_dense/paged_free_i32.c"),
                         ("token_indices_i32", "generic_dense/token_indices_i32.c"),
                         ("cast_f32_to_bf16", "generic_dense/cast_f32_to_bf16.c"),
                         ("attn_tile_softmax_bf16", "generic_dense/attn_tile_softmax_bf16.c"),
                         ("attn_tile_softmax_packed_bf16", "generic_dense/attn_tile_softmax_bf16.c"),
                         ("attn_pack_stats_f32", "generic_dense/attn_pack_stats_f32.c"),
                         ("attn_direct_softmax_bf16", "generic_dense/attn_direct_softmax_bf16.c"),
                         ("attn_tile_indices_i32", "generic_dense/attn_tile_indices_i32.c"),
                         ("attn_decode_exp_f32", "generic_dense/attn_decode_exp_f32.c"),
                         ("attn_decode_normalize_bf16", "generic_dense/attn_decode_normalize_bf16.c"),
                         ("attn_online_merge_f32", "generic_dense/attn_online_merge_f32.c"),
                         ("attn_online_finalize_bf16", "generic_dense/attn_online_finalize_bf16.c"),
                         ("index_put_i64", "generic_dense/index_put_i64.c"),
                         ("argmax_parts_bf16", "generic_dense/argmax_parts_bf16.c"),
                         ("argmax_parts_f32", "generic_dense/argmax_parts_f32.c"),
                         ("argmax_finish_i32", "generic_dense/argmax_finish_i32.c"),
                         ("fp8_block_quant", "block_fp8_framework/activation_native.c"),
                         ("fp8_block_reduce", "block_fp8_framework/reduce_bias.c"),
                         ("fp8_token_quant", "block_fp8_framework/activation_token.c"),
                         ("fp8_channel_finish", "block_fp8_framework/channel_finish.c")):
        flags = []
        if name == "fp8_block_reduce":
            flags = ["-DROWS=4", "-DOUTPUT_BF16=1"]
        elif name == "attn_tile_softmax_packed_bf16":
            flags = ["-DPACKED_STATS=1"]
        subprocess.run([args.tpc_clang, "-O2", "-ffp-contract=off", "-mcpu=gaudi2", "-c",
                        *flags, str(root / f"csrc/tpc/{source}"), "-o", f"{name}.o"],
                       cwd=out, check=True)
        kv_prefix = [b"tensor key", b"tensor value", b"tensor locations", b"tensor metadata"]
        contracts = {
            "kv_store_bf16": kv_prefix + [b"tensor key_pool", b"tensor value_pool"],
            "kv_store_head_major_bf16": kv_prefix + [b"tensor old_pool", b"tensor pool"],
            "kv_store_head_major64_bf16": kv_prefix + [b"tensor old_pool", b"tensor pool"],
            "fp8_block_quant": [b"tensor input", b"tensor output", b"tensor scales"],
            "fp8_block_reduce": [b"tensor partial", b"tensor activation_scales", b"tensor weight_scales",
                                 b"tensor bias", b"tensor output"],
            "fp8_token_quant": [b"tensor input", b"tensor output", b"tensor scales"],
            "fp8_channel_finish": [b"tensor partial", b"tensor activation_scales", b"tensor weight_scales",
                                   b"tensor bias", b"tensor output"],
        }
        if name in contracts:
            verify_tpc_abi(out / f"{name}.o", root / f"csrc/tpc/{source}", contracts[name])
        subprocess.run(["objcopy", "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64",
                        f"{name}.o", f"{name}_x86.o"], cwd=out, check=True)
        objects.append(f"{name}_x86.o")
    subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-Wl,-z,defs",
                    "-Wl,-z,noexecstack", "-I" + args.glue_include,
                    str(root / "csrc/host/generic_dense_glue.cpp"), *objects,
                    "-o", "libgaudi_kernels_bundle.so"], cwd=out, check=True)


if __name__ == "__main__":
    main()
