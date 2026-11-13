# gaudi-kernels

Zero-legacy-bridge Gaudi2 Synapse TPC kernel provider (`libgaudi_kernels_bundle.so`,
40 generic GUIDs) and pure-CPU numerical reference / load-time codec helpers paired with
[`tracycam/GaudiRT`](https://github.com/tracycam/GaudiRT). Licensed under
[Apache-2.0](LICENSE).

## Layout

- `csrc/tpc/common/tpc_math_safe.h`: shared pure-VPU LUT-free TPC math primitives.
- `csrc/tpc/generic_dense/`: generic Dense / GQA PagedAttention (`d ∈ {64, 128, 256}`, SWA, softcap) / Norm (`weight_offset`) / RoPE / Ewise / Argmax / Vocab Sampler TPC kernels.
- `csrc/tpc/block_fp8_framework/`: generic `128x128` Block-FP8 small-bucket and single-pass SRAM `1K FP16` MME TPC kernels.
- `csrc/host/`: table-driven Synapse Glue provider (`generic_dense_glue.cpp`, `generic_attention_glue.hpp`, `generic_fp8_glue.hpp`).
- `python/gaudi_kernels/`: pure-CPU OCP-to-Gaudi2 E4M3 half-range codec (`fp8_codec.py`) and FP8 numerical reference (`block_fp8_oracle.py`).
- `tools/build_generic_dense_bundle.py`: single-provider `.so` builder with embedded TPC ELF ABI verification.

## Build & CPU Tests

```bash
PYTHONPATH=python python3 -m unittest discover -s tests -p "test_*.py" -v
python3 tools/build_generic_dense_bundle.py --output-dir build/bundle
```
