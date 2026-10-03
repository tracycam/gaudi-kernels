# Packed model and advancing target replay — 2026-10-03

The repository now contains an actual packed model consumer, an experimental
consumer of vLLM scheduling/KV allocation, and a target recorder whose device
inputs can advance without recapturing. These extend the
[first protocol checkpoint](UNIFIED_EXECUTION_CHECKPOINT.md).
They do not yet enable speculative native serving or establish new service TPS.

## Shared model execution

`serving/executor/packed_target.py` invokes the loaded model once on the valid
flattened rows. Embedding, norm, block-FP8 QKV, RoPE, output projection, router,
MoE and the LM head are the actual model/operators. Attention retains separate
request visibility. `[1,4,128]` consumes133 rows, not384; unused capacity is
excluded before embedding. Only explicitly selected query rows reach the LM
head, with request/local-query provenance.

`packed_backend.py` registers a DiffKV attention extension before model loading.
Ordinary metadata delegates to the qualified backend. The experimental packed
metadata selects paged FP32 attention with tiled scores and GQA KV sharing.
KV is gathered inside each tile rather than materializing the complete history.
This Torch attention implementation remains a correctness path, not a fused
performance backend. Equal-K/V backend registration is not implemented yet.

Returned hidden states, logits and optional decoder features have independent
output lifetimes. A subsequent forward sharing the model cannot overwrite a
previous returned result. The current diagnostic implementation achieves this
with device clones; a future runtime can replace them with explicit output leases.

## KV and replay boundaries

`PackedKVSession` owns an exclusive bounded arena. Candidate writes never alias
live history; prefix commit preserves the accepted input slots and discards the
suffix. Execution synchronizes before commit/abort. This is not a production
allocator, shared-tail COW implementation, or asynchronous completion ticket.

`PackedInputBuffers` and `BoundKVSession` hold stable token, position, slot-map,
visibility and length tensor addresses. A fixed request/query extent is compiled;
the tensor values can change between replays. Each preparation returns a fresh
transaction, so committing a prior transaction after preparing a new one fails.
Padding is masked and sanitized before either attention matmul, including NaNs.
SWA bounds its gathered keys by window+query_length-1 rather than full capacity.

`packed_advance.py` records the actual target once, updates those buffers, then
replays three advancing steps. Prefix counts include zero and partial acceptance.
Input transfer, inspection and allocator updates are synchronous diagnostic work;
there is no intermediate CPU work inside the recorded target. There is still
CPU preparation between steps, and no device sampler/drafter cycle here.

`PagedKVSession` additionally provides request-private aligned physical pages.
Its optional native SWA consumer uses the existing pinned multi-query FP32 TPC
kernel for its qualified geometry; other shapes use tiled attention. It preserves
the qualified graph-owned reshape edges required by the Lazy CustomOp bridge.
Device qualification of this newly connected consumer is tracked separately.
No weight representation or MoE dispatch is changed by the page arena.

## Actual scheduler consumption

`scheduled_target.py` encodes actual scheduled counts and page tables, borrows
the existing layer KV tensors, and runs all valid rows through one shared model
forward. Decode rows can coexist with an unfinished prefill chunk. An incomplete
chunk emits no token; completed queries use the vLLM sampler and normal output
ownership. Validation precedes state mutation. A failed step invalidates the
experimental runner rather than silently falling back with possibly modified KV.

This path is explicitly enabled by an idle-worker diagnostic RPC. It admits only
synchronous plain TP ordinary execution. Speculative allocator transactions,
PP/DP, LoRA, KV transfer and asynchronous scheduling remain rejected.

## Drafter input interface

`PackedTargetExecutor(feature_layers=(...))` uses the model's auxiliary feature
interface. Decoder index i maps explicitly to boundary i+1: the residual-completed
decoder output before final model norm. Selected feature tensors remain on device.
The prior model capture configuration is restored even if execution fails.
`packed_features.py` independently hooks decoder outputs and checks that feature
capture preserves ordinary hidden states/logits. The interface is model-owned;
it does not contain a second hand-written MiMo decoder.

## Evidence and limits

Device scope: Gaudi2 TP8, pinned Synapse1.24.1/PyTorch HPU2.7 development artifacts.
Source changes are on `feat/unified-token-protocol`. All success/failure logs,
source archives, inputs, full logits and ownership records are retained locally;
private payloads and deployment manifests are not published.

| Run/source | Scope | Result |
|---|---|---|
| J / `a2c19ac` | Two real layers;13/133 packed rows and ordinary-serving teacher | All query, causal isolation, live-KV and rejected-prefix continuation checks passed on eight ranks |
| J / `a2c19ac` | Actual scheduler/allocator/sampler | Mixed `[1,1,510]` and `[1,1,259]` steps executed as512/261 valid rows;12 aligned teacher queries passed; unfinished prefill emitted nothing |
| J / `a2c19ac` | Static same-address target recorder | Three repeated replays byte-exact on eight ranks;20 SDK submissions per captured two-layer target |
| M / `b21f846` | Advancing target recorder,13 query rows | One capture, three mutated replays, changing token/position/KV inputs and partial commits; all eight ranks passed;13 SDK submissions |
| M / `b21f846` | Real decoder features, layers0/1 |13 rows per feature; independent boundary comparison exact; hidden/logits unchanged; capture policy restored |
| K70 / `f3a3968` |70-layer query and continuation check | Random ragged case passed; real-text query/ordinary-teacher checks passed, but continuation failed with maximum KL0.01359; not admitted |

The SDK counts describe different capture scopes, not a paired speedup. Neither
count means one recipe or one hardware launch. No new full-model TPS, physical
HBM traffic, MTP acceptance rate or complete-answer quality is claimed.

The70-layer real-text failure persists after freezing outputs before every
oracle invocation. Old KV bytes, prefix slot coverage and future isolation pass.
The first accepted-KV difference is at layer19; operator-boundary diagnostics
are tracing the preceding arithmetic. Output-buffer reuse is no longer a viable
explanation for that remaining difference. Do not classify it as harmless FP32
rounding or change the gate without identifying the originating operation.

At133 rows the existing QKV policy selects BF16 activation arithmetic, whereas
token-at-a-time execution selects block-A8. That is a deliberate policy difference,
not simply a different FP32 addition tree. Quantization contracts and numerical
comparisons must record the actual branch.

## Reproduction and next work

Use a separately qualified manifest with native serving disabled and the pinned
plugin/core interfaces. The launcher supplies the output directory:

```bash
python -m tools.validation.launch \
  --manifest "$GK_MANIFEST" \
  --plugin-source "$PLUGIN_SOURCE" \
  --vllm-source "$VLLM_SOURCE" \
  --benchmark-module tools.validation.executor.packed_model_probe -- \
  --layers 2 --model "$MODEL" --small-only --serving-teacher \
  --advancing-replay --target-features
```

The shell variables denote user-supplied qualified paths, not a public
downloadable deployment. The70-layer diagnostic selects `--layers 70` and can
use `--trace-layers 17 18 19 20`. Hardware runs must first
verify ownership/idle state; the owned `run_model` supervisor supplies those checks.

Next, identify and repair the70-layer continuation discrepancy, qualify the
connected native SWA consumer and advancing70-layer replay, then connect real
target features to the existing DFlash drafter and device greedy acceptance.
Allocator staging/COW and completion-safe promotion must precede serving
admission. Measure actual tau and full-cycle/per-request TPS only after those
boundaries work. Small-M TPC MoE remains qualified; large-M compact regressions
and the slower all-MME small-batch pipeline are not promoted.
