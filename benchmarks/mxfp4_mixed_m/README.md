# Mixed M1/M2/M3 expert projection

Experimental native Synapse recipe, original MXFP4 weights, BF16 activations,
BF16 MME inputs, FP32 output/accumulation. One `synLaunch` runs device preparation,
all expert decode/MME nodes, and logical output assembly. This starts with
already-routed expert activations; router, token gather and weighted combine are
not included. No vLLM/model integration claim.

2026-09-29: `lean` removes unused ID output and fixed-capacity loads while
retaining runtime count validation. `GK_MIXED_UPSTREAM=separate|fused` adds actual
token gather in both timed arms; fused gather directly produces the final flat
zero-padded activation layout. Only use these controls with `lean`.
`GK_MIXED_SYNTHETIC_PRODUCER=1` is a diagnostic that writes constant BF16 weights;
it is not MXFP4 inference performance. `GK_TIMING_REPEATS=100` increases the
timing window; compare arms with identical repetition settings.

Decoder `--decode-schedule increment|pipeline` candidates change actual address
generation and prefetch scheduling. They remain experimental; source/ELF changes
alone do not establish performance benefit. See the
[complete follow-up](../../docs/MXFP4-ALL-OPTIMIZATIONS-20260929.md).

Build on the matching Gaudi2 SDK, without acquiring a device:

```sh
python3 benchmarks/mxfp4_grouped_bandwidth/build.py --output decoder --rows 2 --decode-canonical-zero --decode-window 8
python3 benchmarks/mxfp4_mixed_m/build.py --output build --decoder decoder
```

Native probe arguments are `POLICY N K n1 n2 n3 banks`. Use the module-checking
`tools/run_device_probe.py` runner and register both `decoder/libgrouped_tpc.so`
and `build/libmixed.so`. Compile/audit SRAM placement before replaying new shapes.
Exact recorded commands are retained with each experiment's local assets.

| Policy | Activation layout | Grouping | Runtime count limits |
|---|---|---|---|
| `exact` | Exact rows | One expert per MME | Cached capacities must fit |
| `pairs` | Pair's maximum M | Adjacent expert pairs | Cached capacities must fit |
| `bucket` | Exact rows, device stable sort by M | Pairs within equal-M buckets | Sorted bucket capacities must fit |
| `merge12` | M1/M2 share cap2; M3 cap3 | Pair within merged buckets | Sorted bucket capacities must fit |
| `padded` | Three rows per expert | Adjacent pairs | Any counts 1..3 |
| `bounded` | Three rows; no sort/prefix scan | Adjacent pairs | Any counts 1..3 |
| `independent` | Same as bounded; decode reads original IDs | Adjacent pairs | Any counts 1..3 |
| `lean` | Same as independent; prep has only activation/count inputs | Adjacent pairs | Any counts 1..3 |
| `fused` | Prep fused into each decode | Adjacent pairs | **Rejected: compiled expanded weights in DRAM** |

`independent` is the accepted candidate for the tested mixed-small-M shapes.
It preserves stable expert order and output slots `[expert,3,N]`. Downstream
consumers must use counts to ignore masked slots. No host count read or sorting
occurs between experts. Three static slots are a bounded recipe shape, not three
real tokens: unused activation rows become zero on device. Their source storage
is deliberately filled with NaNs during correctness tests.

`GK_MIXED_ROTATE_COUNTS=1` rotates every expert's count through 1,2,3 between
pre-uploaded banks, while preserving one compiled recipe. The host selects input
addresses before each full replay, as it also selects a new weight-ID bank;
there are no host calls between the operation's device nodes. This is not a
claim of a persistent device token loop.

Weights use the existing paired-nibble layout: low/high nibbles refer to N lanes
j and j+128. An UNPCK_8_TO_16 load yields byte lookup indices; BV16 paired LUT
decodes both nibbles. K8 independent loads/lookups are retained unchanged. Weights
expand only into SRAM. The decoder's fast path requires aligned N/K and its
existing finite normal scale contract (E8M0 2..252); this experiment samples
118..132 and does not extend that contract to all codes or arbitrary tails.

Analysis and full local integrity verification:

```sh
python3 benchmarks/mxfp4_mixed_m/analyze.py artifacts/builds/mxfp4-mixed-m-20260928/r{1,2,4,5} --output evidence/mxfp4-mixed-m/summary.json
python3 benchmarks/mxfp4_mixed_m/verify_local.py artifacts/builds/mxfp4-mixed-m-20260928/r{1,2,4,5} --output evidence/mxfp4-mixed-m/asset-validation.json
```

See [results and limitations](../../docs/MXFP4-MIXED-M-20260928.md).
