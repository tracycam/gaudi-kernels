# Reusable infrastructure contract

The objective is to abstract proven optimizations into capabilities that an
application can use and another model can reuse. A small explicit integration
is acceptable. Copying an executor, its upload loop or resource management for
each model is not. One unchanged-application wheel comparison measures framework
contribution; it is not a ban on useful public input/capture APIs.

| Responsibility | Owner | What an application supplies |
|---|---|---|
| Capture, execution plan, resources, submission | PyTorch bridge HPUGraph | A capturable tensor computation |
| Reusable inference input storage | bridge InputBufferCache | Names, tensors, destination device |
| Small independent input uploads | standard Torch foreach copy, bridge backend | Source/destination lists |
| Runtime admission/coverage | graph execution_info | No model private fields |
| Optimized computation | Kernel operators | Shape/layout/precision contract |
| Scheduling, sampling and KV state | vLLM | Actual request inputs and state lifecycle |
| Model adaptation | Model/operator integration | Weights, layouts, semantics |

## Implemented cleanup

- Production configuration has one executor, PyTorch, and rejects enabled replay
  flags and historical executor artifacts before loading libraries.
- The old recorder and independent native_graph runtime are removed from the
  installable package and preserved only in the excluded archive.
- Production workers/runner no longer install quality wrappers or recorder hooks.
  Intrusive diagnostics use a separate worker or explicit post-timing callable RPC.
- Input buffer ownership delegates to bridge InputBufferCache instead of copying
  a per-model allocation cache. It uses ordinary Torch storage/copy dependencies.
- MXFP4 routing preserves the existing FP32/shape contract through an explicit
  method adapter; it no longer rewrites source text during import.
- Operator policy installation is separated from numerical diagnostic code.

## Acceptance

1. Same application/config/operators/weights, alternate wheel: verify completed
   wall time and output, keeping integration changes out of the bridge delta.
2. Normal production worker: prove no diagnostic wrapper during timed inference,
   no external executor, actual prepared command coverage and full-model quality.
3. A different pretrained model: same bridge APIs, no MiMo/vLLM/private kernels or
   core change; compare graph to ordinary same-device output, record actual
   admission and completion. Full-forward parity is not KV decode or service TPS.
4. Installed distribution: no retired graph/recorder modules or code payloads;
   one execution owner, bounded inputs and clear fallback/lifecycle behavior.
5. Retain baseline, failures, source, builds and raw results locally with hashes.

The prior bridge-plus-serving combination (`47f28b972f9` / `a450e13`) achieved
91.02 B1 TPS. It is a regression reference, not evidence for cross-model or
unchanged-application gains. New cleanup/production/migration results must be
reported independently; interrupted cases remain pending.

## Remaining work

Extend verified input/capture contracts to prefill, larger batches, sampling and
speculative transactions through the same runtime. Resolve the plugin's
transitional operator-registration interface. Complete independent wheel and
second-model measurements. Keep specialized tuned compute schedules; general
execution does not require replacing them with a slow universal kernel.
