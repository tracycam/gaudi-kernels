# Single-component W4A8 numerical comparison

`moe_accuracy.py` isolates activation quantization on SHA-pinned real checkpoint
MXFP4 weights and actual routes, using CPU FP32 matrix products. It changes
neither the weight quantization nor the BF16 nonlinear boundaries.

```sh
python3 benchmarks/mxfp4_w4a8/moe_accuracy.py \
  --asset artifacts/builds/route-wide/sealed-device-d \
  --vllm ../deps/vllm \
  --out artifacts/builds/w4a8-quality-new --tokens 512
```

`metrics_vs_gpu_w4a8` compares against the pinned vLLM K32 MXFP8 reference.
`metrics_vs_w4a16` reports the separate quantization difference; it is not an
implementation-error threshold. The native E4M3 row-scale policy is a candidate,
not an assertion of exact GPU-contract equivalence. No device performance or
model quality acceptance follows from this CPU benchmark.

The reusable CPU reference for ragged complete MoE is
`gaudi_kernels.mxfp4_reference`. To compare its K32 quantizer against an explicit
local GPU reference source without importing GPU runtime code:

```sh
PYTHONPATH=python python3 benchmarks/mxfp4_w4a8/check_quantizer_reference.py \
  --vllm ../deps/vllm --out /tmp/mxfp8-reference-new.json
```

This check records source identity and compares FP8 bytes/E8M0 scales. It does
not test native Gaudi FP8 representability, a device quantizer or performance.
See [dual-path plan](../../docs/MXFP4_DUAL_PATH_PLAN.md) for the scale-streaming
and full-chain work still required.
