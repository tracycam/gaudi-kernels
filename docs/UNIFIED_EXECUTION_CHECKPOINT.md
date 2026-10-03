# First unified token protocol checkpoint

Branch: `feat/unified-token-protocol`. The implementation sequence and final
service acceptance criteria remain in [the plan](UNIFIED_EXECUTION_PLAN.md).
No new model throughput or native T>1 admission is claimed by this checkpoint.

## Implemented

- `engine/token_batch.py`: model-independent request/query spans, packed tokens,
  bounded trailing capacity, explicit selected-output ownership, and separate
  computed/committed/emitted result counts. No Torch/HPU import or private shape
  constants in the protocol.
- `engine/kv_plan.py`: allocator lease identity, paged visible ranges, candidate
  write alias checks, stale-state rejection, and query-prefix commit descriptors.
  Shared prefix pages are readable; candidate writes cannot overwrite any live
  range or each other.
- `serving/executor/scheduled_tokens.py`: reads actual scheduled request counts,
  CPU input storage, positions, and draft IDs. Rejects incomplete request sets,
  mismatched draft ownership, overflow and missing input rows.
- Owned runner source audit: observes that packed representation before legacy
  prefill/decode separation. Disabled outside diagnostics; bounded512 records
  with explicit dropped-record reporting. It does not yet redirect the model
  forward to a packed backend.
- Minimal target semantic probe: one shared QKV projection and one shared output
  projection for133 valid rows. Per-request FP32 attention tests GQA, causal
  masking and sliding-window visibility. Projection-call counts describe source
  operations, not hardware launches or measured HBM transactions.

## Tests and device evidence

83 local tests and27 subtests pass. Cases include [1,4,128], reordered ragged
requests, query/output provenance, malformed capacity, shared prefix aliasing,
SWA ring wrap, stale KV lease, partial-page visibility, and independent hidden
and attention dimensions.

The standalone module5 probe tested four combinations: hidden width32/48 with
attention width32, and full/SWA128 attention. Each used133 valid rows within
capacity144. Results matched the independent token-at-a-time BF16/FP32 CPU
reference exactly for these fixtures. Changing later verify queries left prior
queries and other requests unchanged; NaNs in11 padding rows had no effect.
Continuing decode after selecting0/2/4 **input-query** commits passed. A zero
input commit here describes abort/discard, not zero accepted drafts in the
complete MTP algorithm, where an anchor may still be committed.

A separate two-layer MiMo TP8 source-audit regression passed B2/B3: eight paired
arms produced identical token sequences;35 teacher-forced queries passed.
Each rank recorded162 scheduling batches, with no dropped entries and a maximum
of254 valid rows. The audit observed actual prefill/decode source state. This
is a two-layer adapter regression, not a70-layer quality gate, a mixed native
verification test, or a timing qualification.

Device source identities: `706db43` for the initial semantic probe, `4012d4b`
for the model source audit, and `8706a48` for the two-geometry semantic probe.
Original stdout, inputs/reference/output tensors, supervision records, source
archives, and failures if any are retained locally. All owned runs exited0,
with no surviving owned model processes; device postflight returned idle.

## Important incomplete boundaries

- Production `_prepare_inputs` still calls the existing plugin preparation;
  metadata audit is not compact execution or a performance improvement.
- Existing native T1/speculation guards remain intact.
- KV commit descriptors do not apply allocator page tables or wait for device
  events. Page promotion, shared-tail copy/COW, and ticket-gated lifetime are
  still needed before a real speculative transaction is safe.
- The target probe is a semantic reference with score materialization, KV
  concatenation and reference GQA expansion. Those are unsuitable as the final
  memory-efficient attention backend; physical read amplification is unmeasured.
- RoPE, MoE, actual block-FP8 projection and the complete decoder/model are not
  included in the standalone synthetic layer.
- Full NextN/DFlash cycles, native packed replay, large-M expert dispatch and
  service throughput remain subsequent checkpoints.

Next: connect packed rows and per-request paged visibility to a real decoder
layer, retain all selected logits, implement completed-event KV page/tail commit,
and only then admit native T>1. The current source audit provides a matched
adapter baseline for that change.
