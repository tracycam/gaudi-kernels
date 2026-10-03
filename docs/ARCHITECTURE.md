# Current architecture

vLLM retains API/request ownership, scheduling, KV page allocation, and output
delivery. The Gaudi plugin provides the worker/model-runner construction point.
`gaudi_kernels.serving.worker.NativeHPUWorker` selects the repository-owned
`NativeHPUModelRunner` through that point.

`engine/` supplies typed policies, shape selection, immutable artifact bindings,
and runtime context. `serving/` provides startup, NUMA/device placement, model
adaptation, and the runner. MiMo-specific expert packing and installation live
in `serving/models/mimo_mxfp4.py`. `serving/executor/` manages capture, named input
transfers, request/query ownership, replay, and numerical diagnostics.

The qualified model path captures a real PyTorch/HPU graph execution and records
the resulting recipe, communication, and transfer sequence. Subsequent eligible
decode steps bind new inputs and invoke the native host runtime without a Python
model forward. Request order/shape changes drain the plan and trigger recapture.
Inputs are validated as a whole before mutation. A partially submitted failure
invalidates the plan; it cannot silently fall back and execute the step twice.

This integrated runtime lives in `csrc/legacy_executor/`. The separately developed
`csrc/native_graph/` owns explicit recipes, resources, and replay tickets, but its
complete decoder-layer producer has not replaced the integrated recorder.
Hundreds of SDK submissions remain inside one FFI call.

Kernel layers are separated into TPC source (`csrc/tpc`), host database metadata
(`csrc/host`), Synapse operation builders (`csrc/ops`), and PyTorch registration
(`csrc/torch`). MME is called through the installed Synapse stack. Python wrappers
carry format, layout, scale, and numerical contracts.

## Transitional boundaries

- Worker/runner construction is explicit; some model/operator installation hooks
  remain transitional.
- Six SWA-related GUIDs were packaged together; there is not one consolidated
  database containing every production kernel.
- Native multi-request T1 execution is tested for B2/B3. That is not native T>1
  verification or a complete speculative decoding transaction.
- Prefill still uses the older bridge path; large-M expert dispatch and
  prefill/decode scheduling interference remain work items.
- Current serving admission is synchronous, with one concurrent batch. Other
  settings, including native speculation, are rejected rather than presumed safe.

See [STATUS.md](STATUS.md) for tested scope and [PUBLICATION.md](PUBLICATION.md)
for the distinction between public source and private deployment artifacts.

Repository cleanup and the path from these components to reusable infrastructure
are tracked in [REUSABLE_INFRASTRUCTURE.md](REUSABLE_INFRASTRUCTURE.md).
