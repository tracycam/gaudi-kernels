# Generated SRAM feed control

This is **not an MXFP4 kernel or effective-bandwidth result**. It deliberately
omits packed-weight loads, decoding, activation quantization and scale handling.
One TPC producer generates BF16 or single-component native E4M3 into transient
SRAM; a real same-format MME performs per-expert M=2 GEMM with FP32 output.
N=512, K=6144, E=8 or 32. All full outputs must match an exact finite oracle.

The control asks how much room the current producer/consumer implementation has
before adding omitted work. Separately addressed expert seed regions prevent
compiler common-subexpression elimination from faking multiple weight tiles.
Physical graphs must still prove SRAM placement, no DMA and exact weight-volume
coverage. A slow control is not a universal hardware impossibility proof.

Build with `build.py --output BUILD`, then use `tools/run_device_probe.py` on the
assigned module and `benchmarks/mxfp4_pipeline/launch.py --buffers 1
--compiler-sram --audit --compile-only -- BUILD/feed bf16 8` for the first audit.
Only after inspecting placement run without `--compile-only`; use a separate
`--profile` process for engine times. FP8 here is native E4M3 on both operands,
without precision reconstruction or extra M rows. Original MXFP4-equivalent
bytes are a target normalization only; no MXFP4 bytes are read by this control.

Optional `shared-a` concatenates experts along N and shares the two activation
rows. This is legal only for experts with identical activation rows (and row
order); it is not a replacement for arbitrary independent per-expert routes.
It changes the generated SRAM layout and output ordering without adding weight
elements. Use the separate full-output and placement audit for this experiment.

`contention` isolates resident MME reads, a reusable two-slot pipeline, a real
data-fenced serial control, SRAM address padding and separate section objects.
Its arguments are `bf16|fp8 resident|pipeline|serial E chunk slots padding_bytes
[split-sections]`. The footprint is bounded to 16 MiB. Use `--hard-control` in
the shared launcher, but verify physical ordering: ordinary control edges were
optimized away in an earlier experiment. The serial kernel now actually reads
its predecessor's FP32 output and predicates stores on a non-NaN value.

For bus samples, nest `/usr/bin/python3 benchmarks/mxfp4_feed_control/bmon.py --
BUILD/contention ...` after the shared launcher's `--profile --`. The launcher
uses `execv`, so nested executables need a path. `analyze_contention.py` audits
placement, whole binary outputs and trace ordering. HBW bus samples do not by
themselves account for all SRAM traffic or measure bank conflicts.
