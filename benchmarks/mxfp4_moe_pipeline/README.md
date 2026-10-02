# Counted mixed-M MoE pipeline experiment

Native Synapse graph on one assigned Gaudi2 module. Each replay contains
gate/up, the established BF16/FP32 SiLU contract, down, and FP32 weighted combine.
`GK_CHAIN_GATHER=separate|fused` additionally executes token gather inside the
same timed graph. Router selection is supplied as input, not computed here.

The synthetic fixture has five tokens, eight distinct experts per token, twenty
active experts and forty valid expert rows: seven M1, six M2, seven M3. Gate/up
is N512/K6144 and down N6144/K256, representing one expert TP shard. Original
MXFP4 weights and their E8M0 scales are retained; BF16 expansion must remain in
SRAM. Two disjoint weight banks total 100.27 MB. No activation quantization.

Build on the matching SDK without acquiring a device:

```sh
python3 benchmarks/mxfp4_grouped_bandwidth/build.py --output decoder --rows 2 --decode-canonical-zero --decode-window 8
python3 benchmarks/mxfp4_mixed_m/build.py --output build --decoder decoder
python3 benchmarks/mxfp4_moe_pipeline/build.py --output chain
```

Run through `tools/run_device_probe.py` with the three libraries
`decoder/libgrouped_tpc.so`, `build/libmixed.so`, and `chain/libchain.so`.
Native arguments: `chain/probe stage|prefetch1|prefetch2|stream BANKS`.
Full commands, including module5 checks, are saved in the experiment assets.

- `stage`: all gate/up outputs precede one global SiLU, which makes down IDs
  available. This intentionally retains the stage-wide dependency baseline.
- `prefetch1/2`: one/two groups of down weights use original IDs independently
  of global SiLU. The trace, not this name, determines how early they run.
- `stream`: each expert pair has its own SiLU/down activation dependency;
  down weight decode uses original IDs. This allows some down decoding while
  later gate/up work remains, despite adding small SiLU nodes.

GP, SiLU output and down output remain persistent for independent per-operator
oracle checks in every arm. These diagnostic boundaries are counted and have
not been eliminated from the final product. Correctness downloads occur before
timing. Each timed `h.run()` is one native `synLaunch`, with no host intervention
between device nodes; repeated timing is a queue of graph replays, not a full
autoregressive model token loop.

All output elements are checked, including NaN-poisoned inactive activation
rows that must yield zero outputs. GP/down use normal FP32 MAC error bounds;
SiLU preserves BF16 gate/up rounding, FP32 evaluation, BF16 SiLU rounding, and
BF16 multiplication with up. Observed SiLU output is zero BF16 ULP from the
reference in this fixture. Combine matches the specified FP32 order exactly.
All accepted variants also match each other's intermediate and final bytes.

`stream` with `GK_CHAIN_GATHER=fused` is the measured combined candidate.
This is an explicit experimental graph, not an installed vLLM backend or
quality/performance acceptance on real model weights. The fast decoder retains
its existing aligned N/K and normal-scale contract. Supplied active token IDs
must be in range; counts must be 1..3 and expert IDs must reference the pool.

See [all optimization results](../../docs/MXFP4-ALL-OPTIMIZATIONS-20260929.md).
