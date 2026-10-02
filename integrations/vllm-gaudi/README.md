# Native runner interface patch

`native-runner-interface.patch` contains the development plugin changes from
`9b158d73` to `9e62e34b`: an explicit runner factory, named input/output transfers,
adapter position handling, and per-query/ragged KV visibility fixes with tests.

It is provided for interface review and for the matching development baseline.
It is **not** a complete MiMo support patch against the latest upstream plugin;
the development baseline already contains additional model and operator work.
The public gaudi-kernels snapshot cannot replace those dependencies by itself.

The patch contains material from vLLM-Gaudi, licensed Apache-2.0. Existing
copyright headers are retained; see `third_party/vllm-gaudi/LICENSE` and
`THIRD_PARTY_NOTICES.md` at the repository root.
