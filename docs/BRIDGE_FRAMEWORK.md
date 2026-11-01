# Application interfaces and model migration

The execution owner is the forked PyTorch Gaudi bridge. There is no second
gaudi-kernels graph runtime to install. Model compute/precision remains an
operator/backend responsibility; graph execution handles bindings, dependencies,
resources and submission.

## Existing Torch entry points

Use `habana_frameworks.torch.hpu.HPUGraph`, `wrap_in_hpu_graph`, or
`wrap_in_hpu_graph_func`. Supported plans select prepared adapters automatically.
Use ordinary Tensor operations for producers and consumers. Capture does not
turn Python request/KV state into automatically advancing device state.

The new `InputBufferCache` is a small inference storage utility. It allocates via
Torch and calls ordinary copy; it does not own a stream, allocator or executor.
Names plus shape/dtype identify buckets. The cache bounds retained entries and
uses LRU eviction; graph owners may retain their own references. Do not mutate
the returned Tensor's layout. Clear/rebind the application graph when replacing
captured module state, and supply new tensors to the wrapper rather than assuming
a manual fixed-binding graph follows a newly allocated input automatically.

```python
import torch
import habana_frameworks.torch as ht

with torch.inference_mode():
    model = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16).to("hpu").eval()
    inputs = ht.hpu.InputBufferCache("hpu", max_entries=8)
    run = ht.hpu.wrap_in_hpu_graph(model, disable_tensor_cache=False)
    for host_input in batches:  # Application-owned CPU tensors, shape [M, 64].
        x = inputs.copy("features", host_input)
        ht.core.mark_step()
        result = run(x)
        consume(result)
```

The example describes interfaces, not a new performance result. Use Lazy mode
and a matching wheel/SDK/Torch ABI. Source tensors of non-blocking transfers
remain valid until completion. `clear()` does not synchronize. Training is not
admitted by this input utility; unsupported graph modes retain framework behavior.

For several independent inputs, use the standard `_foreach_copy_` operator.
Bridge batching guards types, layout, size, aliases and asynchronous dependencies;
it does not merge arbitrary operations across observable effects. A manual graph
offers `execution_info()` for out-of-band plan coverage/rejection snapshots.
Measure completed work or actual output delivery; enqueue time is not throughput.

## vLLM integration

The compatible plugin must offer its worker/runner factory. Select the generic
worker with ordinary vLLM options:

```python
from vllm import LLM

llm = LLM(
    model=model_path,
    worker_cls="gaudi_kernels.serving.worker.FrameworkHPUWorker",
    async_scheduling=False,
    additional_config={"hpu_graph": {
        "scope": "step", "reuse_inputs": True, "max_input_buckets": 128,
    }},
)
```

No `GK_RUNTIME_MANIFEST`, MiMo class name, layer count or kernel pin is required
by this generic worker. It loads the normal model/backend through vLLM. Plain
synchronous greedy unit-query decode is the currently admitted step scope;
other requests retain the ordinary path. Input layout reuse compares the actual
request/page/layout state and rebuilds through the original builder on changes.
Sampling and KV semantics remain in vLLM.

`MimoHPUWorker` and its runner add the reference model's tuned operator policy,
packing and position binding. Its pinned manifest is an operator deployment
requirement, not a general graph requirement. Diagnostic workers and probes are
explicitly selected; they do not install on normal production workers.

Migration should reuse this generic worker and graph API. Add a model adapter
only for genuinely different weights/operators/state semantics. Never copy SDK
submission, resource or upload loops into each model. New shapes/modes need
functional and performance admission rather than a new executor.

## Evidence limits

The prior 91.02 TPS combination is MiMo-specific evidence, not a cross-model
guarantee. The first cleaned revision completed a 70-layer device run before the
remote SSH outage; complete raw evidence retrieval and later generic-worker,
input-layout guard and trained second-model hardware checks remain pending.
CPU checks and installable package checks are reported separately. Do not mark
these pending checks complete merely because the new interfaces exist.
