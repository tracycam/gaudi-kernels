# Current architecture

Production execution has one owner: the PyTorch Gaudi bridge. Applications use
HPUGraph / wrap_in_hpu_graph / wrap_in_hpu_graph_func. Compiled commands, tensor
bindings, allocator generations, streams, completion and failure handling live
inside that bridge. This package provides operators and serving integration.

`FrameworkHPUWorker` selects `FrameworkHPUModelRunner` through the plugin factory.
It needs no MiMo policy/artifact manifest. Applications configure graph scope
through `additional_config["hpu_graph"]`. `MimoHPUWorker` / `MimoHPUModelRunner`
add the reference model's operator installation and position adapter above it.
vLLM owns requests, scheduling, KV allocation and output delivery. The adapter
expresses tensor inputs and eligible capture scope. It does not submit SDK
commands or own another allocator. Input storage reuse delegates to the bridge's
public `InputBufferCache`; small batched uploads use ordinary `_foreach_copy_`.

`engine/` contains the reference MiMo operator policy and artifact manifest. It
is not the generic bridge execution configuration. MiMo expert packing remains
in `serving/models/mimo_mxfp4.py`; model/precision/kernel choices remain above the
runtime. Framework code does not inspect layers, experts, token positions or KV.

`serving/diagnostics/` is explicitly selected for validation. Production workers
have no experiment RPC methods or default quality instrumentation. Validation
can use vLLM callable RPCs out of band; numerical probes install after timing.
No diagnostic invocation is part of token production.

The old recorder, native_step, independent graph API and their SDK runtimes are
in `tools/archive/legacy_execution/`, excluded from wheels and source packages.
Production configuration rejects them, true replay flags and executor artifacts.
Their frozen source and evidence remain available for historical reproduction.

Kernel layers remain TPC (`csrc/tpc`), database glue (`csrc/host`), Synapse graph
builders (`csrc/ops`) and Torch registration (`csrc/torch`). These are compute
operators, not alternative whole-model executors.

## Supported integration and remaining boundaries

- Whole-step integration currently admits synchronous plain greedy unit-query
  decode; unsupported scheduling/sampling states keep their ordinary semantics.
- The bridge APIs are model-independent. Static inference, shape buckets and
  explicit tensor inputs do not imply all dynamic Python state is capturable.
- KV update/page-layout validation belongs to vLLM. It remains in its adapter.
- Operator registration uses a transitional import hook for the pinned plugin,
  installed explicitly by bootstrap, with once-per-module binding, support for
  already imported modules, and rejection after partial installation failure.
  Importing the registration module is inert. Source text substitution and
  legacy worker/runner hooks have been removed.
  A stable upstream registration interface is a separate integration task.
- The MiMo production bootstrap and default launcher require all 70 layers.
  Reduced-layer weight filtering lives only in the validation launcher; it is
  not an operator or bridge optimization. Generic workers do not use this
  MiMo manifest bootstrap and have no model layer-count requirement.
- Quality comparison is a diagnostic API; the production policy module does
  not carry teacher-forced logit checks. Historical validation imports retain
  a compatibility re-export of that same comparator.
- Prefill/MTP/KV rollback contracts and the optimized operator coverage remain
  distinct from framework graph execution. They have not acquired new quality
  or throughput guarantees merely from this cleanup.

See [BUILDING.md](BUILDING.md), [REUSABLE_INFRASTRUCTURE.md](REUSABLE_INFRASTRUCTURE.md)
and [BRIDGE_FRAMEWORK.md](BRIDGE_FRAMEWORK.md) for application interfaces. The local
delivery report distinguishes hardware-qualified revisions from offline changes.
