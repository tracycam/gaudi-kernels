# Unified execution implementation plan

Status: actual packed model/scheduler consumers and advancing70-layer target recorder
tested; full-cycle recording in device qualification. See [current evidence and remaining work](PACKED_TARGET_CHECKPOINT.md).
Device/model completion is recorded separately.
Baseline: public source commit `656b6f7`, development model baseline `a999894`.

## Objective and ownership

Provide a reusable Gaudi backend for vLLM. The scheduler and KV allocator remain
in vLLM; the adapter encodes scheduled work; the model describes its layers and
precision; kernels implement operators; the runtime owns plans and lifetimes.
Prefill, decode, and verification share packed token input, with request-specific
attention visibility and distinct output/commit rules. Kernels and graph modes
may specialize by shape without creating separate model implementations.

Final service goals: single-stream MTP >=170 committed tokens/s and two/three
streams >=80 tokens/s each, targeting100. First certify4K real-context greedy,
then sampling and16K/32K regression;128K prefill is reported separately. All
cycle stages count, including drafting, verification, acceptance, communication,
necessary host work and actual delivery. Raw performance and private prompts
remain outside plugin patches and outside public experiment payloads.

## Checkpoints

| ID | Implement | Evidence required | Current status |
|---|---|---|---|
| C1 | Packed TokenBatch, output ownership, capability/capacity checks, vLLM scheduler adapter | [1,4,128] occupies133 valid rows, ragged/reordered/page-boundary checks, no private model dimensions | Actual two-layer mixed scheduler consumption tested; default production admission pending |
| C2 | KV view/write staging and explicit computed/committed/emitted results | No live KV overwrite, prefix-only commit, reject/abort/EOS/budget ownership, continued attention after rejection | Exclusive arena prefixes and ordinary allocator borrowing tested; speculative allocator promotion/COW/events pending |
| C3 | Target-only mixed layer and model execution | Every request/query compared; padding excluded; regular and sliding-window causal visibility | 70-layer structural checks passed; historical floating thresholds reported separately; serving admission pending |
| C4 | Native packed target runtime | Replay same capacity without request-ID recompilation, validate-before-mutation, partial failure invalidates plan, clean teardown | Advancing70-layer target recording tested; full-cycle/dynamic service qualification incomplete |
| C5 | Complete NextN cycle, then DFlash using same target/KV interface | Actual accepted prefixes and full answers; all rejection paths; full-cycle TPS and conditional acceptance | 70-layer target features and real eager DFlash cycle connected; proposal alignment corrected; recording/quality/TPS pending |
| C6 | Expert-M dispatch, tiled KV reuse, fusion and explicit layer plans | Complete consuming-chain A/B, real route distributions, cold working set, physical traffic where observable | Not complete |
| C7 | Public rebuild and HTTP service | Documented dependencies/artifact generation, pinned plugin interface, concurrency/cancellation/mixed arrival, resource plateau and exit | Not complete |

## Contracts

- `TokenBatch`: valid flattened tokens/positions, cumulative query offsets,
  request ownership, output selection, and separate capacity padding. Ordinary
  [1,4,128] must not become384 input rows by per-request padding.
- `KVView`: existing visible state and storage identity supplied by the
  allocator. Candidate writes use reserved non-aliasing storage. Changing a
  length does not restore an overwritten sliding-window ring.
- `StepResult`: computed input counts, committed input counts, emitted output
  IDs, and output ownership are distinct. Prefill chunks may emit zero tokens;
  verification can compute many rows and commit only a prefix.
- `ExecutionPlan`: typed capacities, numerical/layout contract, workspace and
  actual recipe/collective dependencies. No hidden model constants or policy
  environment reads in generic protocol code.

Do not remove production T1/speculation guards before packed target and KV
state gates pass. Protocol code and CPU/device semantic probes are useful
milestones, but do not themselves admit full-model native verification.

## Validation

Metadata, codec bits, output provenance and KV ownership require exact checks.
Floating arithmetic uses FP32 accumulation/reference; legitimate accumulation-tree
differences are accepted. Precision selection and prior quality thresholds are
explicitly deferred by the user. Preserve these measurements without using them
as a substitute for diagnosing wrong indexing, layouts, masks or scale math.
Do not require FP64 agreement or use aggregate relative error alone to excuse
mask/scale/near-zero bugs. Complete answers and task quality are a separate gate
from teacher-forced layer/logit diagnostics.

Test zero/partial/full acceptance, EOS and output budget, request reorder/removal,
shared prefixes,127/128/129 boundaries, SWA wrap, ragged capacities and allocation
failure. CPU substitute tests and standalone device probes must be labelled as
such, with source/ABI identity and every request/query covered.

## Traffic and performance accounting

Keep checkpoint weight width and block-scale semantics. Loading-time reordering
is allowed and versioned; persistent BF16/FP32-expanded weight caches are not an
acceptable default. Record original storage, algorithmic tile reads, physical
HBM/SRAM traffic, activation/permutation traffic, partial sums, padding/empty
slots and workspace independently. Unobservable traffic stays unmeasured.

MoE dispatch is based on actual per-expert M, not total tokens. Preserve the
qualified small-M TPC path; use grouped MME only where the complete consumer chain
wins. Sweep representative active experts8/16/20/32/128/384 and M1/2/3/4/8/16/32/
128/512/1024, including mixed counts. Small shapes need latency criteria rather
than a universal percentage of advertised compute peaks.

For average committed tokens tau and full cycle t_ms, TPS=1000*tau/t_ms.170TPS
with tau2.4 requires t_ms<=14.12; tau must be measured. Multi-stream results are
per request, not aggregate totals. Report prefill stalls, TTFT, output-group
arrival gaps, recaptures and percentile latency alongside steady decode.

Count Python calls, FFI calls, SDK submissions, recipe/collective calls and
device nodes separately. One FFI invocation is not one hardware launch. Physical
engine intervals use unions, not sums across cores or lanes. Quantify overlap
only with a dependency explanation and corresponding device timeline.

## Delivery discipline

Implement in feature branches with small reviewable commits. Preserve all local
sources and success/failure assets. Public files contain no private host, corpus,
deployment path or payload. Never stop a foreign workload. Record device/NUMA,
source revision, library SHA, test scope and remaining blockers for every run.
Retain the qualified recorder baseline until the explicit layer/model producer
has its own gates; do not rename recording as explicit construction.
