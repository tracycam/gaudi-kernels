# Build and validation boundaries

The Python package can be installed without an HPU. It installs interfaces and
the historical catalog, not prebuilt device extensions or model weights.

```bash
python3 -m pip install -e .
python3 -m gaudi_kernels --json
```

## CPU checks

Use Python 3.10+, pytest, NumPy, a CPU PyTorch build, and `g++` with C++17 support:

```bash
PYTHONPATH=python python3 -m pytest tests -q
```

Tests cover configuration, transfer binding, query ownership, placement logic,
capture gates, timing attribution, and the actual C ABI against an SDK substitute.
They do not execute on Gaudi hardware.

## Device compilation

Install a compatible Gaudi SDK and TPC compiler separately. A representative
build-only command is:

```bash
python3 tools/build_block_fp8.py \
  --output-dir artifacts/builds/block-fp8-local \
  --include-dir /usr/include/habanalabs \
  --tpc-compiler tpc-clang
```

The output directory must not already exist. This builds a TPC kernel database;
it does not acquire a card, launch a kernel, register every PyTorch operator, or
qualify an end-to-end deployment. Other builders have their own explicit inputs.
Benchmark drivers can acquire devices; use only devices assigned to the run.

Some assembly experiments require a historical ELF as a metadata container.
In particular, the `--literal` variant of
`benchmarks/mxfp4_grouped_bandwidth/build.py` requires a pinned `reference/gp.o`
that is deliberately not distributed. Its assembly source is included. The
source release does not claim all historical experiments can be rerun without
their archived assets.

## Model serving

The launcher requires a runtime manifest, a compatible development vLLM-Gaudi
plugin, a vLLM checkout, model weights, and separately qualified libraries.
Manifests bind actual files and SHA identities; the typed policy JSON examples
under `config/` alone are not deployment manifests. The interface patch under
`integrations/` is a reviewable part of the plugin work, not the entire model fork.

Native execution is enabled explicitly with `runtime.native_enabled=true` and
the native runner policy after model initialization. Current admission requires
synchronous scheduling and one concurrent batch; native speculation is not yet
admitted. Do not bypass artifact validation to make an unqualified build start.

`tools/verify_assets.py` and some historical consolidation tools expect private
artifact manifests and evidence. Those inputs are intentionally absent from the
public source release. Their failure without the inputs is not a device test.
