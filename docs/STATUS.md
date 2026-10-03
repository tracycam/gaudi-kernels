# Measured development status — 2026-10-03

These observations came from the local development system: a 70-layer MiMo
model, Gaudi2 TP8, and separately qualified runtime artifacts. Raw private
prompts, model weights, binary artifacts, and full traces are not part of this
source release. This table describes that system, not a fresh public-checkout
benchmark or a general model-quality certification.

| Scope | Observation |
|---|---|
| Single stream, no MTP, 4K prompt | Approximately 90–92 tokens/s; NUMA placement and compact command scanning did not establish a speedup |
| Two streams, no MTP, 4K prompts | 62.58–63.07 tokens/s per stream in the common decode window |
| Three streams, no MTP, 4K prompts | 60.03–61.61 tokens/s per stream in the common decode window |
| QKV postprocess fusion, native/native paired timing | B2 about 0.548 ms/step saved; B3 about 0.432 ms/step saved |
| Fusion numerical check | 35 aligned full-logit queries bitwise equal; tested generated token sequences equal |
| Multirow arithmetic vs prior policy | Maximum teacher-forced KL about 0.00422 (B2) and 0.00789 (B3); this comparison is not bitwise equality |
| Native automatic startup | Two-layer TP8 check, 12 native steps per rank before benchmark enable RPC |
| Full native MTP/DFlash cycle | Real eager functional cycle connected; corrected proposal alignment and full-cycle recording undergoing device tests; no qualified170 tokens/s result |

Subsequent [packed target work](PACKED_TARGET_CHECKPOINT.md) passed actual
two-layer mixed scheduling and70-layer native SWA, advancing target replay and
independent target-feature checks on eight ranks. Three changing target replays
were byte-exact against the same eager program. These are functional diagnostics,
not new serving performance. Historical floating thresholds are now reported
separately under the user's deferred-precision rule. Serving speculative
admission remains closed for allocator/lifetime/full-cycle work, rather than
requiring FP64 agreement. A DFlash proposal-position bug was found and fixed;
the earlier zero-acceptance smoke result is not a valid drafter assessment.

The multi-request generation test used a private 4K code-review corpus, greedy
sampling, and a 96-token output budget. The common decode window begins after
every request has returned four tokens. Per-request denominators end at each
request's own completion; summing those rates is not a valid aggregate measure.
The timing is at the engine client, not HTTP. Outputs hit the length budget, so
they do not establish complete-answer or EOS quality.

## Prefill interference

An earlier real-context native arm measured first-token arrival at 4.14/8.51 s
for B2 and 4.14/8.65/12.95 s for B3. These include queueing and scheduling; they
are not isolated prefill kernel times. In B3 the first request's own decode
window was about 10.87 tokens/s while other requests were still prefilling.
The common decode rates therefore must not be presented as full-session user
experience. There is no new 128K prefill result in this release.

## Remaining priorities

1. Native multi-token target verification with correct per-query ownership.
2. Full NextN/DFlash drafting, acceptance, EOS/budget handling, and safe KV
   commit/rollback. Restoring a length does not restore overwritten SWA data.
3. Large-M expert dispatch and mixed prefill/decode scheduling.
4. Fewer device-node and collective boundaries; current replay still performs
   roughly 305–311 SDK submissions per eligible multi-request step.
5. Complete-answer evaluation, HTTP service measurements, and graceful teardown.

Small-M production MoE remains on the hand-scheduled TPC path. A slower
decode-to-BF16/MME experiment is not promoted merely because it uses MME.
Dispatch should depend on each expert's actual M and measured complete-chain
cost, not just total batch size.
