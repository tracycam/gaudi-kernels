# Third-party notices

The native FP8 load/conversion pattern is derived from the Apache-2.0 licensed
1CatAI/1cat-vllm-gaudi revision `9f2b52ae3b3be7c4e3d081329b4e87b9d3bea096`.
The upstream license is preserved at `third_party/1cat/LICENSE`. Relevant source
comments and original experiment provenance remain intact.

`reference/production-quant/legacy_ops.py` and `candidate_ops.py` contain
vLLM-Gaudi reference/adapted code with the existing Intel Corporation
2024–2025 Apache-2.0 copyright header preserved. These are numerical audit
references, not a redistributed SDK. The candidate includes project changes
to quantization precision; it is not an unmodified upstream file.

`integrations/vllm-gaudi/native-runner-interface.patch` and the historical
interface diffs under `tools/consolidation/upstream/` contain changes to
Apache-2.0 vLLM-Gaudi sources. A copy of the upstream license is preserved at
`third_party/vllm-gaudi/LICENSE`. Upstream project:
https://github.com/vllm-project/vllm-gaudi

The catalog does not claim ownership of Intel's compiler, Synapse, MME runtime
or PyTorch bridge. These are external dependencies. Matching their runtime ABI
is part of backend qualification; this repository does not bundle those runtimes.

Model checkpoints and business evaluation inputs are not included and are not
licensed by the project LICENSE. Refer to their respective distributors.
