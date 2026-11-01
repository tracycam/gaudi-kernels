# Reuse the owned native graph for expert MoE

The user requires existing work to be reused, with new work becoming reusable
infrastructure. This is an implementation constraint, not an optional cleanup.
Raw successes and failures continue to be retained locally.

## Existing pieces to keep

- `graph.py` and `csrc/native_graph/`: owned recipe serialization/cloning,
  named external buffers with retained owners, dedicated allocator-supplied
  arenas, byte-range validation, completion tickets, cache and plan save/load.
- `csrc/ops/mxfp4_moe_grouped.*` and `mxfp4_moe_bucket.*`: explicit Synapse
  graph construction patterns. Their older route/layout/capacity contracts are
  not interchangeable with the current expert partitioner.
- `moe_expert_plan.py`, the current count/prefix/inverse kernels, K8 decoder,
  gate and combine: the measured original-storage W4A16 implementation.
- Existing small-M TPC dispatch, typed serving policy, vLLM worker factory,
  module locks, owned runners and artifact identity checks.

## Current evidence and correction

The independent expert backend passes changing routes at T512/2048/4096.
Its large-T complete operator chain is faster than route-broadcast. This has
not yet become a model speedup. A two-layer vLLM T2048 test passes conditioned
teacher logits but regresses prefill: the owned runner disables outer prefill
HPU graphs, so the large Python operation builder is re-entered on every call.
A per-layer `wrap_in_hpu_graph` experiment then fails at the owned reshape dtype
contract, including after moving route dtype conversion outside capture. The
exact unexpected dtype has not been isolated; it must not be called solved.
These experiments and their Git versions remain diagnostics, not acceptance.

The per-layer HPU Graph wrapper is not the architectural destination. Reuse the
owned native runtime and add the missing producer/integration boundaries.

## Missing boundaries, in implementation order

1. Explicitly append the **current** count/bucket/GP/gate/down/combine chain to
   a Synapse graph. Reuse the tested TPC ELFs and BF16 MME arithmetic. Compile
   a recipe once per admitted static shape. Counts remain device data.
2. Hand that recipe to the existing owned runtime; declare original packed
   weights/scales as external read-only owners. Use its workspace/arena and
   cache machinery. Do not build another memory manager or command recorder.
3. Add a device-to-device serving boundary for activation, route and output
   tensors, with framework-stream dependencies and lifetimes made explicit.
   Current `Graph.replay` stages GKG_INPUT through host memory and downloads
   GKG_OUTPUT. A model adapter must not use those staging roles for intermediate
   activations. EXTERNAL_READ/EXTERNAL_RW roles already avoid these copies, but
   fixed address ownership and cross-stream ordering still need an adapter.
4. Verify changed IDs, M distributions, tensor addresses and allocator churn,
   then two-layer and70-layer production A/B. Only then admit grouped shapes
   and2048/4096 chunks in a deployment manifest.

`Graph.capture()` explicitly does not qualify recording a Lazy Python forward:
its recorder expects synchronous submissions on one host thread and one stream.
Do not erase that restriction or assume a Python stream/event integer is a raw
Synapse handle. The installed bridge maps framework IDs to SDK resources.

The existing native replay still submits each recipe/collective through SDK
APIs. One Python/FFI call is not one hardware submission. The compute producer
must reduce recipe boundaries; replacing the host wrapper alone is insufficient.
