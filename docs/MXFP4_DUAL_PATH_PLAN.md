# Ragged expert MoE: W4A16 and MXFP8 W4A8

User objective (2026-10-03): optimize complete MoE for variable expert count
and arbitrary per-expert M, retaining both W4A16 and K32 MXFP8 W4A8. This is
C6 of UNIFIED_EXECUTION_PLAN, not a replacement for the serving/MTP objectives.
Accept legitimate FP32 accumulation differences; precision selection is deferred.

## Current gap

The canonical model uses route-batched TPC GEMV even for large total token count.
`compact` is not grouped expert-M GEMM. Experimental BF16 grouped MME exists
outside this default. Existing metadata has T<=513, topk<=8, E<=384 limits and
the projection benchmark formerly rejected every M other than1/2/4 even on MME.
W4A8 has CPU accuracy experiments, not an integrated MXFP8 device backend.

## Interface and execution

- Input: BF16 valid token rows, device expert IDs/routing weights, original
  MXFP4 bytes plus K32 E8M0 scales, an explicit activation policy and workspace.
- Device counts/offsets/permutation describe arbitrary ragged expert groups.
  E_active, per-expert M, total tokens and top-k are separate quantities.
  Zero-count experts do no arithmetic. Changing counts must not require CPU
  readback, recapture or per-expert host dispatch.
- Output: GP/gate/down with the chosen numerical boundaries, scatter/combine
  to original token order and FP32 local reduction. Distributed reduction
  belongs to the caller; record it separately when timing the whole layer.
- Shape-specialized TPC and grouped MME backends share the interface. The
  choice must minimize whole-call latency under contention and workspace
  limits, not sum independent isolated-engine minima. GP and down may differ.
- Large E uses bounded work waves. Large M uses M/N/K tiles and finite live
  FP32 accumulators. Inactive graph slots must not consume stale/NaN operands.
  A masked combine alone does not prove an empty MME node is skipped.

One universal instruction stream is not the goal: small calls cannot saturate
all engines and launch cost dominates some shapes. The target is the best
measured complete-chain latency within each shape/resource regime, with a
documented fallback for unqualified shapes rather than a false global optimum.

## Two numerical policies

W4A16: BF16 activations; original MXFP4 weights/scales; FP32 accumulation.
Small M may compute in TPC; larger M decodes bounded BF16 SRAM tiles for MME.
Large tiles with unrepresentable BF16-scaled weights require an explicit safe
path; arbitrary scale bytes are not silently accepted by a normal-only decoder.

W4A8: quantize GP input and post-SiLU/up down input independently to K32
E4M3FN values and E8M0 scales, with FP32 scale computation. Retain the same
nonlinear/combination boundaries. Native Gaudi FP8 encoding adaptation is a
separate stage: neither raw OCP byte compatibility nor lossless halving at
subnormal boundaries may be assumed. No two-component FP8 detour is planned.

GP activation quantization is shared: quantize each original token row once,
then route/gather the codes and scales to its experts. Do not quantize the same
input eight times. Down quantization follows each expert's nonlinear output and
cannot in general be shared between different experts. Routing weights remain
FP32 at the weighted combine boundary.

For each row m, output n and K32 block g:

    Y[m,n] = sum_g (sA[m,g] * sW[n,g] * dot(qA[m,g], qW[n,g]))

Ordinary FP8 MME lacks this block-scale operand interface. Implement either
streamed block partials plus scale/FP32 accumulation, or a proven representation
that preserves the chosen contract. Never replace this with one row/column
scale silently. Never materialize all G*M*N partials in HBM. Report additional
native-representation error separately from the intended MXFP8 quantization
error. The supplied CPU reference does not mandate a bitwise reduction tree.

Before building the FP8 full chain, measure the scale-accumulator throughput.
For E8/M1024/N512/K6144 there are805,306,368 K32 output partials, or3GiB if
materialized as FP32. Streaming removes that HBM allocation but not the scale
arithmetic or producer/consumer dependencies. Since BF16 already measures
387TFLOPS for this projection, raw FP8 MME peak alone cannot predict a gain.
Benchmark quantization, native-format adaptation and streamed accumulation
together; retain both user-visible policies without asserting W4A8 always wins.

## Storage and bandwidth accounting

Persistent prepack may permute weight bytes/scales without increasing their
17/32 bytes per parameter for aligned K. No persistent expanded weight copy.
Decode to SRAM, reuse within an expert tile, overlap producer/consumer where
the actual compiled graph permits it. SRAM allocation, repeated decoder
instances, physical overlap and HBM traffic need evidence, not source inference.

Unique active weight bytes / whole-chain time is an effective rate, not physical
HBM bandwidth. Large M can make this rate small while compute efficiency is high.
Report useful FLOPs, padded/executed work, temporary peak and actual rereads.
With finite SRAM and arbitrarily large M, a universal exactly-once physical
weight-read guarantee is not credible; measure and minimize amplification.

## Implementation order and acceptance

1. Reusable CPU FP32 reference for canonical packed weights, ragged routes,
   both policies, tail handling and full nonlinear/weighted combine. Add
   adversarial varying-block scales, masked/zero experts and row permutations.
   This is an oracle, never a host-routing runtime implementation.
2. Extend existing BF16 MME projection experiments beyond M4, retaining
   poison/finite/numerical checks, rotating weights and compiled SRAM audits.
   Measure actual GP/down shapes; then connect grouped MME for large expert M
   in the canonical model, keeping small-M production TPC as the comparator.
3. Implement device compact route metadata and ragged work descriptors without
   padding every expert to global M. Calibrate GP/down crossover tables using
   full consumer-chain costs, including preparation and combination.
4. Add single-component MXFP8 FP8-MME scale streaming and activation quantizer.
   Compare against the same-policy FP32 reference, then compare W4A8 vs W4A16
   as an explicit precision choice. Do not block on FP64 equality.
5. Joint task/tiling optimization: register/issue-slot audit for TPC, decoder
   store/MME feed overlap, bounded accumulator lifetime, load balancing and
   empty-work cost. Reject expanded weights in DRAM and unmeasured speedups.
6. Integrate both policies through vLLM/packed execution with runtime bindings,
   then measure prefill and actual MTP/output-arrival throughput and memory.

Matrix: E_active=1,2,4,8,16,20,32,64,128,384 and actual model E;
M_e=0,1,2,3,4,5,8,16,32,64,128,256,512,1024,2048 plus tails. Include
mixed M1/2/3, one hot expert with many cold experts, all experts, zero slots,
changing counts in one recipe and real model routes. Include GP/down separately
and the complete chain. Cover larger dimensions by bounded tiling, not by
claiming an untested monolithic allocation supports infinity.

Report median/p95 complete latency, effective unique-weight rate, useful
TFLOPS, engine overlap, padded-work ratio, peak allocated/reserved bytes and
numerical differences. Pair with the current production chain on identical
inputs; cold/rotating and hot-cache tests are distinct. Native graph capture
does not imply one SDK launch. Serving promotion requires observed full-chain
benefit and correct ownership, without reinstating obsolete FP64 gates.

## Initial checkpoint

`mxfp4_reference.py` implements step1 as a CPU oracle, including nonaligned K.
The grouped projection probe now admits MME M1..4096 and E1..384 with explicit
memory bounds; TPC remains limited by the actual compiled specialization.
Neither change installs a new production kernel or proves a performance gain.

Module5 projection probes of the existing K8 BF16 decoder/MME path completed:

| Projection | Active experts | M per expert | N / K | Median event us | Useful TFLOPS |
|---|---:|---:|---|---:|---:|
| GP | 3 | 3 | 512 / 6144 | 20.205 | 2.80 |
| GP | 8 | 32 | 512 / 6144 | 36.148 | 44.56 |
| GP | 8 | 1024 | 512 / 6144 | 133.128 | 387.14 |
| Down | 8 | 1024 | 6144 / 256 | 130.679 | 197.20 |

Each median is over five event sample means, not per-launch percentiles. All
poisoned outputs and full numerical comparisons passed. These are synthetic
finite normal-scale projections at actual TP-local dimensions, not full MoE,
checkpoint quality, a newly optimized binary, or prefill/model speedups. The
E8 cases rotate51MiB original weights; E3 rotates19.125MiB and is not a cold-HBM
claim. M1024 here means1024 rows for EACH active expert; total prefill tokens
must not be confused with this expert-local M.

The E8 compiled graphs each contain four decoder/four MME nodes. All expanded
weights reside in SRAM; address union is24MiB for GP and12MiB for down. This
proves placement/coverage, not physical HBM transaction counts or engine overlap.
Down materializes192MiB FP32 output per replay, against6.375MiB original
weights/scales. Bounded down-to-combine consumption therefore belongs in the
optimization scope; isolate physical stalls before calling it the sole bottleneck.

The shared CPU MXFP8 quantizer matches the supplied local vLLM reference's
codes/scales on five shapes through M1024 (reference SHA256
`03a16184c905b3a5f0f4aa52736cefeb0528d97fa3d79a85da08d10b68b86c09`).
122 CPU tests pass. FP8 MME scale streaming and runtime expert-M dispatch are
still pending. Local source/binary/graph/output evidence is retained in
`mxfp4-dual-path-20261003`; no production default has changed.

A subsequent same-input GP E8/M32 ABBA compared compiler slicing (A) with the
existing two-slot manual scheduling request (B), without changing decoder ELF:
A1/A2=35.513/35.531us; B1/B2=31.392/31.381us. Mean arm medians improve
35.522→31.387us, a11.64% projection latency reduction in this sequence. All
four output-bank hashes match in all arms. These are separate-process probes,
not a resident-model ABBA or a universal scheduling policy.

Both graphs have four decoder/four MME nodes and SRAM-only expanded weights.
The actual decoded address union rises24→36MiB: the compiler uses three
weight buffers despite the requested two slots. Hence this result trades SRAM
capacity for lower observed latency; it does not prove physical overlap without
an engine trace and must be retested within GP/gate/down/combine lifetimes.


## Device checkpoint

See [expert-M dispatch implementation and measurements](MOE_EXPERT_DISPATCH_CHECKPOINT.md).
Changing device counts now select unique expert buckets and masked TPC fallback;
complete T512/T513 W4A16 chains have measured wins. Static unused MME capacity
still executes. Invalid intermediate rows may contain stale/NaN values only
under the explicitly tested mask-before-read consumer contract; that is not
proof of skipping MME arithmetic. The model default is not yet switched.
