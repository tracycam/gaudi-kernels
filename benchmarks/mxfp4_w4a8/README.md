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
