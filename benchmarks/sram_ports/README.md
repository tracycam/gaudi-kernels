# SRAM port measurements

This is a synthetic endpoint experiment, not an MXFP4 kernel or model TPS result.
See [the measured report](../../docs/SRAM-PORTS-MEASURED-20260928.md).

`build.py --output BUILD` produces TPC ELF, assembly, disassembly, a kernel library
and the native Synapse probe. It does not acquire a device. Run through the
explicit-module `tools/run_device_probe.py` wrapper, with an exported committed
source identity, and the graph-auditing `benchmarks/mxfp4_pipeline/launch.py`.
`sram_ports/launch.py` disables CSE only for that probe process. It can configure
separate BMON/SPMU captures with `GK_PORT_BMON=1`.

Arguments are:

```
probe read|write|copy|mme|mix-read|mix-write|mix-copy \
  rows repeats N M K batch mme_nodes a|b|ab|out bf16|fp8
```

TPC traffic uses 24 tasks, 64 uint32 lanes per row, eight explicitly independent
loads per unrolled body. Read tests sum every loaded value; write/copy tests
change their final value with the repeat count. All outputs are checked exactly.
`GK_PORT_INTERLEAVE=1` changes the task/row SRAM layout in isolated TPC tests.
`GK_PORT_TASKS=1..96` varies the task count; count actual engines in the trace,
not from the task count alone. `--write-unroll 32` builds a store scheduling
control. `GK_PORT_FLAT=1` collapses isolated-write tensor coordinates to two
dimensions while preserving byte layout. All candidates retain exact output gates.

`scaling.py --module 5` runs the ABBA store task/layout/unroll sweep;
`--extra` probes the within-D0 anomaly and working-set sizes. See
[the current MME feed report](../../docs/MME-FEED-TARGET-20260928.md) for results.

MME tests place A, B, both, or the FP32 output in explicit RMW SRAM. Inputs are
initialized each recipe. Repeated GEMMs survive physical graph validation; the
whole output is checked. `out` uses one MME, varying output size to inspect the
write path. Mixed write/copy tests retain their SRAM output until a final TPC
drain reads both that output and the last MME result, forming a real join.

Use complete synEvent time for the main rates. For paired measurements change
only the amount of repeated work, run short/long/long/short in separate processes,
and divide the byte difference by the complete-time difference. Never divide
store bytes by TPC SPU halt time. Profile start markers must be inspected before
using an MME interval: some are writeback milestones rather than read starts.

`analyze.py ROOT... --output measurements.json` checks physical placement, whole
saved outputs and overlap. `summarize.py measurements.json --output summary.json`
derives the r5 paired rates and labels profile-only cross-checks separately.
Sources/builds and SHA manifests must be retained locally. None of these probes
establishes an absolute physical peak or a complete SRAM bus-transaction counter.
