# Third-party notices

The native FP8 load/conversion pattern (`python/gaudi_kernels/fp8_codec.py`) is
derived from the Apache-2.0 licensed `1CatAI/1cat-vllm-gaudi` revision
`9f2b52ae3b3be7c4e3d081329b4e87b9d3bea096`. The upstream license is preserved
at `third_party/1cat/LICENSE`.

This repository does not bundle Intel's TPC compiler, Synapse, or MME runtime;
those are external host/device dependencies. Model checkpoints and evaluation
inputs are not included.
