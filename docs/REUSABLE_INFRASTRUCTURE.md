# Consolidation contract and roadmap

The objective is to reuse optimized code when porting, tuning and deploying
another model, while retaining measured performance. Reuse existing work first;
new code must have a clear home and reusable boundary. Raw experiment assets,
including failed alternatives, remain local. Git history preserves removed code.

## One owner per responsibility

| Responsibility | Canonical source | What a new model supplies |
|---|---|---|
| Graph/resources/replay primitives | `graph.py`, `csrc/native_graph/` | Explicit graph producer and declared tensor owners |
| Optimized compute | `csrc/tpc/`, `csrc/ops/`, `csrc/host/` | Missing operator semantics/layouts, not another runtime |
| Tensor/framework boundary | `csrc/torch/`, serving integration | Valid tensor/stream/allocator contracts |
| Shape and numerical policy | `engine/` | Capability and precision requirements |
| Model adaptation | `serving/models/` | Weight/layout mapping, structural constraints and model-specific semantics |
| Scheduling and KV allocation | vLLM plus `serving/worker.py` and `runner.py` | Supported model adapter |
| Diagnostics and benchmarks | `serving/diagnostics/`, `tools/validation/`, `benchmarks/` | Fixtures and end-to-end regression cases |

These are ownership boundaries, not a claim that the existing code is already
model-generic. MiMo-specific dimensions and admission limits still exist in
several operators/configurations. Generalize these from actual second-model
requirements; do not replace tuned kernels with a slower universal loop.

## First cleanup

- Remove the newly introduced per-layer PyTorch HPU Graph cache, including its
  cache invalidation and capture counters. It failed model integration and is
  not the destination for the native backend. Keep its failure evidence.
- Keep one production MoE selection entry; preserve compact/broadcast dispatch
  and the explicitly opt-in grouped experiment.
- Move MiMo packing/weight/model adaptation to `serving/models/mimo_mxfp4.py`.
  Production imports point directly to it; no compatibility forwarding module.
- Remove unreachable historical modes, sorted execution and BF16-routing
  branches from the fixed FP32 production contract. Preserve explicit rejection
  of unsupported requests and the actual casts/materializations still required.
- Include the existing serving `startup/sitecustomize.py` in the Python wheel;
  package discovery alone omits this non-package startup directory, which the
  installed launcher needs.
- Preserve TPC/MME source, assembly, binary SHA pins, precision settings and the
  existing integrated replay implementation during this structural change.

The current fast model path lives in `csrc/legacy_executor/`. Its one-time
framework capture remains a dependency until explicit producers replace it.
Deleting it now would delete tested functionality. The newer owned runtime is
already in `csrc/native_graph/`; it is the destination, not a reason to introduce
another runtime. See [the concrete MoE integration gap](MOE_NATIVE_GRAPH_INTEGRATION.md).

## Verification and dependency inspection

```sh
PYTHONPATH=python python -m pytest tests -q
python tools/runtime_inventory.py --output /tmp/gaudi-runtime-inventory.json
```

The inventory identifies conservative Python dependency closures and checks
that serving does not import benchmark/build/reference scripts. It does not
infer device-kernel reachability or declare unreferenced modules safe to delete.
Kernel libraries can expose operators used only by prefill or diagnostics.

For arithmetic-preserving MoE entry changes, reuse
`tools/validation/compare_moe_entry.py` with the saved prior source. It compares
operator selection, casts, materializations and metadata at small/large rows.
It does not replace device numerical or end-to-end performance checks.

## Next checkpoints

1. **Native MoE producer and framework boundary.** Reuse current expert kernels
   with the owned graph runtime. Device-only intermediate handoff, explicit
   stream ordering, no per-layer Python operator construction on replay. Gate:
   route/address changes and allocator churn, then production A/B, including
   preserved small-batch latency and2048/4096 prefill.
2. **Shared operator contracts.** Centralize shape/layout/precision/workspace
   capability declarations. Preserve measured specialized schedules. Gate:
   every admitted shape has an explicit supported path or a documented fallback;
   selection must not read device counts back to CPU.
3. **Deployment closure.** Build versioned artifacts and dependency manifests
   directly from repository source. Replace remaining plugin source rewrites
   with explicit integration points. A narrow bridge patch is justified only
   by a demonstrated missing interface; no PyTorch core fork is required by
   this cleanup. Gate: clean-environment installation and reproducible serving.
4. **Second-model reuse.** Add a structurally different model through the adapter
   and missing operators. Gate: no modifications to graph ownership/replay or
   deployment machinery; real generation, prefill/decode and memory checks pass.
5. **Archive review.** Separate historical experiments after their source,
   build, license and fixture dependencies are recorded. Archive locally before
   removing anything; never delete solely because decode traces omit a GUID.

## Performance admission

A source cleanup is not a speedup claim. Record unchanged device sources and
binary pins, compare operation sequences, then compare actual token arrivals
and prefill completion on the same model/settings. Use paired measurements
where the architecture permits them, retaining cold work, KV transitions and
failures. Small differences inside run variance are inconclusive. A kernel
microbenchmark win cannot admit a slower model path.
