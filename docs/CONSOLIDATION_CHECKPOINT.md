# Reusable runtime cleanup checkpoint

This is the first bounded cleanup toward reusable infrastructure, not completion
of native MoE integration or a universal model backend. The runtime change is
commit `cf3c887`; packaging fix `4e1bb19` changes no inference code.

## Changes

- Removed the failed per-layer PyTorch HPU Graph wrapper, its cache invalidation
  hooks and counters. Its committed source and local failure evidence remain.
- One MoE production dispatch entry remains. Small-row compact/broadcast paths
  retain their operator choices. Grouped GEMM remains explicitly opt-in and is
  not admitted by this checkpoint.
- MiMo expert weight/layout adaptation now lives in
  `serving/models/mimo_mxfp4.py`, with direct imports from the installer and
  diagnostics. No duplicate adapter or forwarding module was added.
- Removed unreachable historical bootstrap modes and MoE sorted/BF16-routing
  branches under the fixed production FP32 contract. Unsupported sorted requests
  still fail explicitly. Existing required casts and clones are retained.
- Serving Python code decreased5960→5873 lines. Benchmark/reference history was
  not deleted; decode-only reachability is insufficient to retire those assets.
- Fixed a real packaging omission: the wheel now includes the existing serving
  `startup/sitecustomize.py`. A fresh target installation imports catalog,
  configuration and the native graph interface without importing Torch.

## Verification

- 130 CPU tests and29 subtests pass.
- 70 representative metadata-only old/new MoE comparisons cover rows0/1/2/8/
  512/2048/4096, compact/broadcast, GP/down choices, masked routes, debug outputs
  and rejections. Operator calls, dtype conversion and materialization sequences
  match. These are not device arithmetic or memory-traffic measurements.
- All219 protected files—217 device/C++ sources plus artifact pins and engine
  configuration—are byte-identical to pre-cleanup commit `37be750`.
- The source inventory covers113 Python modules:88 in the conservative serving
  closure,2 in the owned-graph closure, with shared package initialization.
  No serving imports of benchmark/build/reference scripts were found. Files
  outside these roots can still be standalone APIs; they are not declared dead.
- The122-file wheel contains every inventoried Python source, the moved model
  adapter and bootstrap asset, and neither removed module. Installation was
  tested outside the source checkout.

## Real model regression

MiMo70 layers, TP8, the retained private4K corpus, greedy96-token continuations,
B2 and B3. Reused `native_batch_test.py`: bridge/native/native/bridge, same
resident model per batch. All returned token sequences match the bridge,
and every rank records native batch execution. Owned runner exit0; all eight
cards returned to768MiB/0% utilization.

| Native decode | Per-request rate, two native arms |
|---|---:|
| B2 | 62.64–63.27 tokens/s |
| B3 | 59.60–61.06 tokens/s |

These are actual engine-client arrivals in the common post-warmup window, ending
at each request's own completion. They exclude earlier prefill from the decode
rate; raw first-token arrivals and full intervals remain recorded. No new HTTP,
complete-answer, B1, MTP or large-prefill performance claim is made.

No material regression was observed against the previously documented roughly
63/60 tokens/s per-request range. This run compares bridge/native execution of
the cleaned source; it is **not** a same-process before/after-source ABBA and
cannot establish an exact zero change or a speedup. Device sources/pins and the
operator sequence identity provide independent evidence for the refactor.

Two initial invocations of the older numerical-policy benchmark failed argument
prechecks before model inference; their numerical gates were not weakened. The
existing execution-equivalence benchmark was then used for this unchanged-
arithmetic cleanup. The first package build exposed the omitted bootstrap and
is retained alongside its corrected build. A broader504-case metadata audit
completed comparisons but was interrupted during finalization; the bounded
70-case rerun exited0. All failures remain local.

## Next

Follow [REUSABLE_INFRASTRUCTURE.md](REUSABLE_INFRASTRUCTURE.md): reuse the existing
owned graph for the current MoE producer and device tensor/stream boundary;
validate production gains and2048/4096 prefill; then centralize operator
capabilities and deployment interfaces. A second model must reuse graph/runtime
and deployment code to demonstrate generality. Do not start another executor or
replace tuned kernels with a slower universal implementation.
