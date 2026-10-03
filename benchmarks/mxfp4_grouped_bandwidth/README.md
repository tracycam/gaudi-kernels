# Grouped MXFP4 bandwidth experiment

Private experiments for **8–384 distinct active experts** on one TP shard.
M is the number of activation rows **per expert**. No production policy is
changed. Results and limitations: [report](../../docs/MXFP4-GROUPED-BANDWIDTH-20260929.md).

The probe now also admits E1..7 and MME M1..4096 for new coverage experiments,
with a512MiB activation/output bound. Those are admitted input sizes, not
measured performance or model qualification. TPC M remains1/2/4 with its
matching compiled specialization. Historical results remain8..384 experts.

- Original MXFP4 bytes + E8M0 scales, BF16 activations, FP32 accumulators/output.
- The kernel receives device expert IDs; all experts share one task grid.
- N256 and losslessly interleaved N512 layouts keep exactly 17/32 bytes/weight.
- TPC tasks decode a weight once and reuse it across their M rows. Split-K
  tasks have disjoint K32 ranges. Partial output sums are not expanded weights.
- The MME comparator uses compiler-managed SRAM decoded tiles and FP32 output.
  Actual compiled graphs are audited; a logical two-node graph can become many
  physical decoder/MME nodes.
- Literal GP control/prefetch support **N512, K6144, M1, split3 only**. Other
  shapes fail explicitly. The pinned reference ELF is the actual qualified
  production instruction schedule, not an unexecuted C template.
- The folded-scale arithmetic has a bounded fixture qualification, not a full
  BF16/E8M0-domain or full-model quality certificate. Do not promote by default.

Build in a fresh source snapshot with a matching Gaudi2 SDK:

```bash
python3 benchmarks/mxfp4_grouped_bandwidth/build.py \
  --output builds/grouped-m1-prefetch --rows 1 --literal prefetch
python3 tools/run_device_probe.py --module 5 --case grouped-e8-unique \
  --git-commit "$(git rev-parse HEAD)" \
  --kernel-library builds/grouped-m1-prefetch/libgrouped_tpc.so -- \
  python3 benchmarks/mxfp4_pipeline/launch.py --buffers 1 --compiler-sram --audit -- \
  "$PWD/builds/grouped-m1-prefetch/probe" tpc 8 32 512 6144 1 3 4 random
```

Native probe arguments are `mode E pool N K M splits banks pattern`.
The pool must equal `E*banks`, at most384. Each bank selects E unique experts;
banks are disjoint. E8 GP uses four banks, E8 down eight banks, to rotate more
than48MiB of original weights. Full poisoned-output checks precede timing.
Timing includes the complete projection and any reduction node. Five sample
means are retained, no slow samples dropped; profiled cases are excluded from
latency comparisons. Event timings still include boundaries between replays.

`--literal control` uses the qualified247-packet GP body, `prefetch` uses the
corrected222-packet body. `safe`, `--lookup-x2`, scalar `pair`/`inc`, and short
vector windows are retained **experimental/rejected controls**, not recommended
implementations. Build success or `device_verified=False` in static metadata
is not a correctness result; consult the sealed per-case output and report.

```bash
python3 benchmarks/mxfp4_grouped_bandwidth/analyze.py \
  artifacts/builds/mxfp4-grouped-bandwidth \
  --output evidence/mxfp4-grouped-bandwidth/summary.json
```

The analyzer requires a completed local seal, audits actual graph placement,
checks expert-ID uniqueness, distinguishes failed/compile-only cases, and
verifies ABBA input/reference/output hashes. Its byte proof concerns source
requests and storage ownership, **not physical HBM bus transactions**. The
actual binaries/disassembly, inputs, outputs, graphs, traces, and failed
candidates remain in the local asset archive.
# Aligned decoder scheduling result

The subsequent opt-in `GK_GROUPED_RING_SLOTS=2` host graph uses independently
named two-expert decode/MME nodes. Compiled graphs use **three** SRAM addresses;
the requested control edges are not a guarantee. ABBA E8/E16/E32 M2 latency
improves a further 7.5%/14.8%/19.7%, with identical decoder instructions and
bitwise outputs. See [the ring report](../../docs/MME-FEED-RING-20260928.md).
`ring_analyze.py` checks actual placement, exact expert coverage, outputs and traces.
The `GK_GROUPED_RING_DATA_PREFILL` experiment is rejected: its extra weight
consumer moves expanded weights to DRAM. Do not use it as a performance candidate.

The opt-in `build.py --rows 2 --decode-canonical-zero --decode-window 8`
specializes the grouped decoder for its validated K%32==0 domain. It preserves
the original weight storage and the existing finite-MAC baseline. Paired
E8/E32 M2 measurements improve complete projection latency by 15.5%/14.5%;
this does not reach the 3.2 TB/s feed target. See
[the report](../../docs/MME-FEED-TARGET-20260928.md) and `window_analyze.py`.
