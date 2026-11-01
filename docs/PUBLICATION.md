# Public source provenance

This initial public snapshot is exported from development commit
`a9998944bfa63ff10b4c1d85be53bc3f29d5ce08`, dated 2026-10-02. That identifier
refers to the locally retained development repository, not an ancestor reachable
in this new public repository. The development plugin was at
`9e62e34b`; its adapter interface diff is included separately.

The public repository starts with fresh Git history. It includes project source,
assembly, numerical references, CPU tests, benchmark/build drivers, configuration
examples, and public-facing documentation. Kernel arithmetic and device source
were not changed as part of publication. SSH benchmark drivers now take an
explicit `--host` and optional `--port` instead of a private machine address.

The release omits the prior Git history, private deployment/asset manifests,
raw experimental evidence, business prompts and outputs, model weights,
compiled objects/libraries, and proprietary toolchains. They remain in local
archives; publication did not delete or overwrite them. SHA-pinned deployment
libraries are not implied to be included just because their expected hashes
appear in source.

Some historical scripts and reference code deliberately remain as research
assets. Paths to omitted fixtures are not promises of a ready-to-run benchmark.
Public status documentation supersedes the historical catalog for integration
readiness. This repository does not republish the private business corpus.

Apache-2.0 was selected for the public project. Third-party code retains its
original attribution and licensing; project licensing does not relicense the
Gaudi SDK, runtime, model weights, or separately installed dependencies.

---

## Phase 4.2 kernel bundle & safe-math update (2026-10-31)

This update exports verified kernel snapshot `d23e585` (`Add compensated QKV
midpoint rounding for model TPC bundle`), paired with the companion
zero-legacy-bridge C++ runtime repository
[`tracycam/GaudiRT`](https://github.com/tracycam/GaudiRT) (`8fc3b3b`):

- **Hardware-safe TPC special functions (`csrc/tpc/common/tpc_math_safe.h`)**:
  exact 3-instruction piecewise linear-interpolation workaround for the 750
  `v_exp_f32` LUT boundary discontinuities discovered on Gaudi2 TPC hardware.
- **Compensated QKV BF16 midpoint rounding (`csrc/tpc/block_fp8_reduce_isa/round_bf16_compensated.c`)**:
  Neumaier-compensated Block-FP8 reduction rounding that eliminates double-rounding
  bias at BF16 midpoints during 70-layer prefill and decode.
- **Consolidated TPC provider bundles (`csrc/host/mimo_device_bundle_glue.cpp`, `csrc/host/mimo_prefill_bundle.cpp`)**:
  merges custom decode and `M=32` prefill TPC kernels within Synapse's per-process
  `GC_KERNEL_PATH` shared-library ceiling.
- **Device-resident token step kernels (`csrc/tpc/mimo_step/`, `csrc/tpc/dense_prefill/`, `csrc/tpc/moe_prefill/`, `csrc/tpc/full_mme_attention/`)**:
  `M=32` static-chunk prefill kernels, device embedding, and greedy argmax kernels.

Dated internal machine/NIC operational notes and out-of-tree raw benchmark
receipts remain omitted in accordance with project sanitization rules.
