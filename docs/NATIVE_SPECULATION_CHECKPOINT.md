# Native speculative cycle checkpoint — 2026-10-03

Numerical correctness remains required. Precision selection is deferred:
legitimate FP32 addition-tree and rounding differences do not block functional
work. Index/layout/mask/scale errors, nonfinite arithmetic and stale storage
are implementation failures. No FP64 agreement requirement is introduced.

The actual70-layer target and five-layer DFlash now compose through device
token tensors. A complete fixed-cohort cycle can be recorded once and replayed
without intermediate Python model/verifier execution. It still contains
314–315 C++ SDK submissions; this is not one hardware launch or speculative
serving admission. See [operator and ownership evidence](PACKED_TARGET_CHECKPOINT.md).

Two actual wiring errors were corrected: retaining the original SWA query
producer at the Lazy CustomOp boundary, and selecting DFlash denoiser positions
1 onward rather than treating its known anchor as the first draft. Real4K
smokes now accept drafts; the earlier zero-acceptance result is invalid for
judging the checkpoint.

## Fixed page bindings

V70, source `8c7a7e2`, compared legacy/static bindings in one resident TP8
process. Every case ran16 cycles, including one eager warmup, one capture and
14 advancing replays. No route snapshot hooks were installed. All eight ranks
passed live-KV, context-position, finite/output-count checks and released
recorded storage. All integer proposal/target/emitted witnesses matched between
binding variants on every rank and every step. Weights/arithmetic were unchanged.

| Requests,queries/request | Binding | Mean emitted/request/cycle | Median cycle,ms | Binding drain,ms | Replay with completion,ms |
|---|---|---|---|---|---|
|1,8|legacy|4.6875|44.63|6.29|36.23|
|1,8|static|4.6875|44.31|0.071|42.65|
|3,8|legacy|4.0833|78.79|10.72|65.59|
|3,8|static|4.0833|58.69|0.069|56.83|

Numbers are rank0 medians from a single sequential pair per cohort, not ABBA,
an engine union, or service TPS. Medians of separate stages need not sum to the
median cycle. Removing changing-length history construction eliminates the
binding drain; overall single-request timing did not improve. Three-request
cycle latency improved about25.5% in this diagnostic. Replica votes/delivery
remain in the cycle; answer completion/EOS and HTTP have not been certified.
Do not turn mean acceptance divided by a median diagnostic time into a serving
TPS claim. The qualified no-MTP service baseline remains approximately90–92TPS.

`PagedKVSession.reserve` owns candidate prefixes without creating device
history arrays. `BoundKVSession` retains request-private fixed-capacity page
maps and updates only query positions, candidate mappings and live lengths.
Out-of-capacity padded slots point to safe slot0 and are masked before
arithmetic. CPU tests independently compare these bindings to variable views,
including rejected prefixes and NaN padding. These private arenas are not the
vLLM allocator's COW/async lease implementation.

## Next implementation priorities

1. Attribute the40–60ms replay body by actual engine/recipe/operator intervals.
   Fixed bindings do not explain that remaining time. SDK enqueue currently
   takes approximately5–7ms and overlaps device work; never add it again to
   the completion span. Compare shared QKV/attention/MoE policies with the
   qualified native baseline, including all multirow fallbacks.
2. Move repeated position/page generation and anchor/budget feedback into the
   device cycle. Use accepted counts to advance a device cursor, with candidate
   storage reserved for the whole bounded horizon. Keep an output/commit log
   on device. Promote only completed verified prefixes at delivery. Restoring
   a cursor cannot restore overwritten live KV.
3. Connect qualified QKV postprocessing to actual owned KV buffers and fuse
   full-attention work. Packed metadata currently fails some ordinary decode
   fusion guards; do not merely weaken those guards while still pointing at
   another allocator's cache. Treat metadata/layout support as an explicit
   operator contract.
4. Keep small-M MoE on TPC. Actual U70 data shows M<=2 for94.7% of B1,T4 active
   experts and75.4% of B3,T8 experts. First extend/test the efficient compact
   path for12/16/24 rows and audit GP/down assembly, register spills and issue
   slots. Use actual expert M to admit MME only after a complete-chain win.
5. Integrate completed KV reservations, cancellation and request-cohort changes
   with the existing vLLM scheduler/allocator. Then measure full answers,
   accepted-token arrival, conditional acceptance and real single/multi-stream
   throughput.170TPS and80–100TPS/request remain goals, not results.

Every followup separates functional correctness, precision measurements and
performance. Original checkpoint storage width/block scales remain unchanged.
Source archives, failures, successful runs, private inputs and full witnesses
are retained locally; source-only changes stay on the public feature branch.

## Device body and row coverage followup

Direct Synapse captures now contain physical TPC/MME/DMA execution events for
an actual completed70-layer DFlash cycle. NIC has metadata only: its activity
is **unobserved**, not zero. Engine occupancy is unioned across cores/lanes.
Reused DMA context IDs and differing begin/end SP categories are handled by
queue-order descriptor pairing. Operator-family unions can overlap; do not
sum them into wall time.

For B1,T8, the broadcast-GQA capture spans26.98ms: TPC14.75ms, MME3.36ms,
compute union17.83ms and compute gaps9.15ms. GP5.33ms and down3.36ms are the
largest TPC families. B3,T8 spans57.61ms: compute union41.35ms and gaps16.27ms.
The old broadcast MoE `nm_gemv` family occupies22.60ms; block-FP8 decoding
occupies2.92ms. Profiling is intrusive and these windows are not service TPS.

The exact loaded GP ELF/.text hashes were checked against the qualified pins,
then disassembled with the TPC triple. Its K32 hardware-loop body contains247
logical VLIW packets,215 occupied VPU slots,128 BF16 FP32-accumulating MACs,
eight FP32 scale MACs and32 activation shuffles. The four accumulator chains
recur at six-packet distance. Embedded C source/basic-block labels are stale
after previous manual ISA edits; optimization must use the actual binary.
The initial248 count included the nonrepeated LOOP delay NOP at0xd20. Pinned
TPC LLVM `7f67b4e2` inserts that delay slot separately; the repeated body starts
at0xd40. The occupied-slot and MAC counts are unchanged.

Folding GQA query heads into M is functionally sound on the tested cohorts:
all eight ranks' proposals, target-next IDs, emitted IDs and accepted lengths
match the broadcast version. It is kept explicit/experimental. The median of
the **maximum rank duration per synchronized step** was40.56→41.01ms for B1
and77.26→83.37ms for B3. Rank0 alone misleadingly suggested a gain; neither
cohort timing nor physical compute unions establishes one.
The unprofiled cohorts also performed full-KV CPU snapshots before the narrow
timer, without aligning rank readiness. Early ranks can measure waits for
peer diagnostic copies. Later source explicitly aligns after snapshots and
records audit costs; prior timing remains diagnostic evidence, not a clean
performance adjudication or service throughput.

Unchanged compact GP/gate/vector-down/combine ELFs were tested directly on
loaded two-layer model expert weights at8/12/16/24 rows, on all eight ranks.
All128 state checks passed final-consumer equality, finite arithmetic, row
permutation and zero routing. Complete-chain ABBA favors12 rows slightly;
16 is roughly tied and24 loses. Only12 rows is admitted to a **scoped private
cycle experiment**; production configuration and automatic serving row sets
remain restricted to the prior qualified rows.

The independent GP reference initially exposed a reference-layout mistake.
Each128 FP32 partial words stores even64 BF16 lanes followed by odd64; gate
consumers already decode this ABI. A row-major comparison without that
permutation produced a false large-error finding. Separately materialized GP
and graph GP match bytewise; correcting the reference layout gives maximum
relative L2 about9.35e-8 over512 columns on all ranks. No MAC or precision
change was required. Original evidence and the corrected interpretation are
both retained. Legal FP32 addition-tree rounding remains accepted; precision
selection is deferred as instructed.

The complete B3,T4 compact12 ABBA followup (`700c905`) passed four16-cycle
runs on all ranks. Every integer witness matches every variant and mean
emitted/request/cycle is3.0. Each captured cycle still has314 SDK submissions.
Unaligned maximum-rank cycle medians were61.42/59.35/57.52/57.46ms in ABBA
order. The apparent roughly1.7% difference is smaller than timing drift and
is not promoted as a speedup. The separately aligned repeat also exposes
pre/post live-KV audit costs and captures physical engine intervals.

Compact12 is **route-batched GEMV**, not grouped expert-M GEMM. GP tasks are
`rows*topk*3` K-split tasks; down tasks are `rows*topk*12` output tiles. They
retain original packed weight storage but can reread the same expert for
different routes. Removing activation prep/replication does not establish
unique-expert HBM reuse or zero physical read amplification. Production
row dispatch stays unchanged until the consumer-chain evidence supports it.
