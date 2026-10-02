# Native FP8 MME contract probe

This is a value-format diagnostic, not a quantizer or a throughput benchmark.
It keeps eight experts, one actual activation value per input element, native
FP8 operands, and FP32 MME output. No TPC kernel library is required.

Build without acquiring a device:

```sh
python3 benchmarks/fp8_mme_contract/build.py --output artifacts/builds/fp8-contract-new
```

The executable accepts `a` / `b` for E5M2 in both operands, varying A / B;
`a44` / `b44` for native E4M3 in both operands; and `a4` / `b4` for mixed
formats as negative interface controls. E4M3 tests include the native finite
range through ±240 and its subnormals; E5M2 tests cover normals and zeros.

Run only via `tools/run_device_probe.py` with the assigned module and current
committed source identity. `GK_FP8_CONTRACT_COMPILE_ONLY=1` stops after graph
compilation. Format results do not establish full-GEMM accuracy or model quality.
