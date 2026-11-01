# Expert-M dispatch checkpoint — 2026-10-03

This implements device-selected W4A16 expert buckets and measures the complete
local MoE chain. It does **not** establish a universal minimum latency, enable
the model's default backend, or claim an end-to-end TPS improvement.

## Result

One shared plan for T=512, top8, 384 checkpoint experts now beats route-broadcast
on all four tested distributions. Each expert belongs to exactly one M bucket:
1–64 or 65–512. No expert's M is split into repeated fixed-M32 weight decodes.
The static capacities are384 and63 slots; unused MME capacity still executes.

| T512 route distribution | Active experts; per-expert M | Broadcast | Final plan | Latency reduction |
|---|---|---:|---:|---:|
| Saved checkpoint routes | 150; maximum98 | 6.790ms | 4.522ms | 33.4% |
| Eight hot experts | 8; all512 | 6.797ms | 3.642ms | 46.4% |
| Balanced32 | 32; all128 | 6.793ms | 3.958ms | 41.7% |
| Mixed | 384; four512, remaining5/6 | 6.794ms | 5.210ms | 23.3% |

T=513 also passed changing-route replay: checkpoint6.804→4.743ms,
hot8 6.805→3.915ms, balanced32 6.805→4.179ms, mixed6.806→5.473ms.
Its planning bound exceeds1GiB, so that experiment explicitly allows2GiB.
These are single-module complete producer/GP/gate/down/combine/consumer graphs,
with two same-process ABBA trials and fixed resident inputs. They exclude TP
communication and all other model layers. Timing is unprofiled HPU event time.

Distribution-specific plans can still be faster: an earlier T512 hot8 plan
with eight M512 slots measured1.659ms, and a T512 balanced32 plan measured2.819ms.
Those are separate runs, not a same-process comparison against the final plan.
A later64/128/512 three-band sweep measured4.651ms on checkpoint routes,
3.766ms hot8,3.889ms balanced32 and5.333ms mixed; it did not dominate the
two-band plan. Choosing those plans from device counts without paying for every static branch
remains work. Small-T production TPC kernels have not been replaced. The sweep
baseline is route-broadcast, not the faster compact/folded small-T production path.

## What changed

- `moe_expert_plan.py`: explicit M bands, expert-slot capacities, a conservative
  workspace bound, and optional bounded capacity with overflow retained on TPC.
- `moe_expert_dispatch.py`: one device histogram shared by M buckets; validated
  prefix/inverse maps partition actual experts, and TPC owns unselected routes.
  Route/count values never return to the CPU. Same graphs accept changed IDs,
  counts, token permutations, and restored inputs.
- The K8 decoder interleaves eight loads/lookups. It retains original
  HistoricalN512 MXFP4 bytes and E8M0 scales and produces BF16 SRAM fragments.
  This decoder requires the existing finite scale domain2..252; the model
  prepack contract is narrower8..244. It is not a full-domain MXFP4 reader.
- Empty decoder slots may skip stores. `sparse_map` writes only selected route
  cells; `counted_gather` checks valid rows **before** reading that map, and
  skips invalid activation rows. GP's gate masks before reading partials;
  combine never reads rows excluded by the validated inverse map.
- Combine shares each scalar route-index/routing-weight load across four
  independent64-lane vectors. Each output retains its original route-slot
  FP32 FMA order. It does not introduce a new quantization policy.

## Physical trace, then optimization

The first all-MME plan had real wasted padding traffic. Per-family interval
unions in the candidate recipe's first profiled replay changed as follows:

| Family | Before sparse padding / combine256 | After |
|---|---:|---:|
| Row mapping | 800.7us | 108.8us |
| Gather | 2166.4us | 1200.0us |
| Combine | 1129.8us | 397.1us |
| Decode | 2573.7us | 1702.6us |
| BatchGemm | 2152.0us | 2143.1us |

Families overlap and **must not be added**. The decoder arithmetic did not
change in this comparison; its union changes also reflect scheduling. The
unprofiled checkpoint sequence was6.358→5.260→4.522ms as padding traffic and
then repeated combine metadata reads were removed.

The compiled timed candidate has594 decoder outputs, all placed in SRAM.
Its workspace is815,562,752 bytes, versus201,326,848 for broadcast. There is no
persistent BF16/FP32 weight cache. Activation/partial-result workspace grows,
and physical HBM transactions have not been proved exactly1.000×. Original
weight storage width and logical expert reuse are not physical-counter evidence.

## Correctness and failed experiments

- All tested device counts, expert ownership, valid inverse targets, and
  selected row maps match a separate CPU oracle. Sparse padding cells are
  intentionally undefined and must never be read by a valid consumer.
- Changing-route, token-permutation, zero-routing, restored-input, and output
  consumer checks pass. Restoring the same input restores identical output.
- Writing NaNs instead of skipping padded activations/empty decoded experts
  preserves the complete output. The skip-vs-poison and old-vs-combine256
  comparisons cover28 saved outputs, all bitwise equal. This establishes
  masking for these cases; it does **not** mean empty MME work was skipped.
- Independent CPU FP32 K32 MoE references use unpacked canonical checkpoint
  bytes. For three sampled rows per distribution, final checkpoint relative
  L2 is2.41e-4 (0.0241%), max absolute error4.06e-8; hot/mixed relative L2 is
  below9e-8. Full-output candidate-vs-broadcast checkpoint relative L2 is
  about9e-5. Different FP32 accumulation trees can cross a BF16 gate rounding
  boundary. No FP64 or cross-tree bitwise gate was imposed. Full-model quality
  and a broader adversarial numerical campaign are not certified by this sample.
- Blind power-of-two buckets regress sparse inputs; capped buckets retain
  correctness but did not alone repair latency. A device-compacted cyclic TPC
  queue is bit-correct but slower and is not selected. These negative cases,
  failed builds, literal comparator assembly, and logs remain archived.
- 126 CPU tests /29 subtests pass. Running unscoped `pytest` additionally discovers
  historical benchmark files with duplicate module basenames and fails collection;
  the documented source-suite command is `pytest tests`.

## Use and remaining integration

After registering the separately built TPC providers and loading the corresponding
Torch custom-op libraries, the explicit experimental operator entry is:

```python
from gaudi_kernels.moe_expert_plan import ExpertPlan
from gaudi_kernels.moe_expert_dispatch import expert_moe

plan = ExpertPlan(min_mme_rows=1, row_caps=(64, 512))
y = expert_moe(
    x, ids, routing, gp, gs, down, ds, lut, directions,
    plan=plan, tpc=existing_tpc_moe,
    decoder="k8", empty_mode=1, padding_mode=1,
)
```

This example is for the measured T512 MiMo TP-local geometry: hidden6144,
GP N512/K6144, down N6144/K256, top8. Do not substitute another packing with the
same tensor shape. Original weights are not expanded in HBM. Plans and scratch
capacity are static; expert assignment within each plan is device data.

Remaining work, in order:

1. Assemble the new provider with existing libraries within the runtime's TPC
   library limit; expose a typed optional backend in the owned vLLM executor.
   Preserve the current small-T compact/folded path. No deploy-time source patch.
2. Run actual model prefill/verify with the same activation policy, allocator
   lifetime checks, and sampled logits/quality. Operator gains are not model TPS.
3. Select a small set of MME capacities and TPC fallbacks from measured total
   chain cost; include mixedM and actual expert counts. Avoid all empty branch
   costs and the failed cyclic scheduling strategy. No universal M threshold yet.
4. Reduce the remaining1199us gather union and the hundreds of decoder/MME
   fragments; measure physical memory traffic and actual scratch high-water use.
5. Extend routed metadata beyondT513 and other N/K shapes. W4A8 MXFP8 still has
   a CPU reference, not an optimized integrated device backend.

Raw weights, outputs, binaries and full traces remain local. The private archive
is `gaudi-kernels/artifacts/builds/moe-expert-dispatch-20261003/`, beside this public
source checkout. The saved fixture/runtime are reused by verified SHA from
`route-wide/sealed-device-d`; they are not distributed in the public repository.
