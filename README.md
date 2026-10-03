# gaudi-kernels

Gaudi2 LLM kernels, numerical contracts, native execution experiments, and a
vLLM integration layer. Licensed under [Apache-2.0](LICENSE).

This is a research source release. It contains working, shape-specific kernels
and an integrated multi-request decode implementation, alongside experimental
alternatives. It is **not a turnkey inference server**, a general replacement
for CUDA Graph, or a claim that every kernel beats the vendor implementation.

## What is here

- BF16, ordinary FP8, block-scaled FP8, and MXFP4 linear/GEMV/GEMM paths.
- TPC decode/compute, SRAM-to-MME graph builders, and numerical reference code.
- RMSNorm, residual, QKV postprocessing, RoPE/KV, SWA, and MoE building blocks.
- Typed execution policies, physical-device/NUMA binding, native capture/replay,
  and repository-owned vLLM worker/model-runner classes.
- CPU tests, assembly sources, benchmark drivers, and profiling analysis tools.

See [architecture](docs/ARCHITECTURE.md), [measured status and limitations](docs/STATUS.md),
[build instructions](docs/BUILDING.md), and [numerical contracts](docs/CONTRACTS.md).
The next implementation checkpoints and acceptance criteria are in the
[unified execution plan](docs/UNIFIED_EXECUTION_PLAN.md).

## Start without a Gaudi device

Python 3.10+ is required. From a source checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m gaudi_kernels
```

Installation and catalog inspection do not allocate a device. The catalog is a
historical, shape-specific inventory; it does not enable experimental backends.
The Python distribution is named `gaudi-llm-kernels`; its import is `gaudi_kernels`.

For CPU validation, install a CPU PyTorch build, NumPy, pytest, and a C++17 compiler:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pytest
PYTHONPATH=python python -m pytest tests -q
```

Device builds require a separately installed compatible Gaudi SDK, TPC compiler,
and PyTorch HPU stack. The qualified development stack was Gaudi2 / Synapse 1.24.1
/ PyTorch HPU 2.7; other combinations require validation. Proprietary SDKs,
compiled experiment libraries, model weights, private prompts, and raw traces
are not distributed here.

## vLLM integration

vLLM owns requests, scheduling, KV allocation, and output delivery. Our worker and
runner own the eligible native execution path. Single-step B2/B3 decode was tested
with a 70-layer MiMo model on TP8; full native multi-token verification and the
MTP/DFlash cycle remain unfinished.

The source expects additional interfaces in the development vLLM-Gaudi plugin.
An [interface patch](integrations/vllm-gaudi/README.md) is included for review;
it is not a complete patch against an arbitrary upstream release. Native serving
also requires separately built and qualified artifacts with matching SHA pins.
`pip install .` alone cannot reproduce the full-model deployment.

## Performance reporting

The latest local full-model results were approximately 90–92 tokens/s for one
non-speculative stream. With 4K contexts, two streams achieved about 63 tokens/s
each and three about 60–61 **after all requests entered decode**. Prefill caused
substantial earlier stalls. These are development-system observations, not
portable guarantees or HTTP serving benchmarks. See [scope and measurements](docs/STATUS.md).

One Python/FFI call is not one hardware launch: the integrated replay still
submits hundreds of SDK operations. Algorithmic bytes/time is not a measurement
of physical HBM transactions. Kernel speedups are not automatically model speedups.

## Source layout

```text
python/gaudi_kernels/engine/    Typed policies and artifact ownership
python/gaudi_kernels/serving/   Launcher, worker, runner, model adapter
  executor/                   Capture, bindings, replay, query state
csrc/tpc/                     TPC source kernels
csrc/host/                    Kernel database glue
csrc/ops/                     Synapse graph construction
csrc/torch/                   PyTorch custom-operator registration
csrc/legacy_executor/         Current integrated capture/replay runtime
csrc/native_graph/            Explicit graph/resource infrastructure
reference/                    Numerical and implementation references
benchmarks/                   Experimental probes and benchmark drivers
tools/                        Build, validation, and analysis utilities
tests/                        CPU and SDK-substitute tests
```

Historical benchmark scripts may require separately generated fixtures or local
artifacts. Their presence is not a support or correctness guarantee. The release
scope and omitted assets are documented in [publication provenance](docs/PUBLICATION.md).

## Contributing and licensing

See [CONTRIBUTING.md](CONTRIBUTING.md). Retain third-party notices, state precision
changes, and report failures along with successes. Hardware tests must run only
on devices assigned to the experiment.

Project-authored code is available under Apache-2.0. Existing third-party
copyrights and license notices remain applicable; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This is an independent project, not an Intel-endorsed product.

中文：这是 Gaudi2 内核与原生执行基础设施的研究源码仓库。已验证的模型路径和
仍未完成的 MTP、prefill、通用化工作均在状态文档中明确列出；不包含 SDK、权重、
私有业务输入或实验二进制，也不承诺克隆后即可一键运行完整模型。
