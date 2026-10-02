# Public source provenance

This initial public snapshot is exported from development commit
`a9998944bfa63ff10b4c1d85be53bc3f29d5ce08`, dated 2026-10-02. That identifier
refers to the locally retained development repository, not an ancestor reachable
in this new public repository. The development plugin was at
`9e62e34b`; its adapter interface diff is included separately.

The public repository starts with fresh Git history. It includes project source,
assembly, numerical references, CPU tests, benchmark/build drivers, configuration
examples, and public-facing documentation. Kernel arithmetic and device source
were not changed as part of publication. SSH benchmark drivers now take an
explicit `--host` and optional `--port` instead of a private machine address.

The release omits the prior Git history, private deployment/asset manifests,
raw experimental evidence, business prompts and outputs, model weights,
compiled objects/libraries, and proprietary toolchains. They remain in local
archives; publication did not delete or overwrite them. SHA-pinned deployment
libraries are not implied to be included just because their expected hashes
appear in source.

Some historical scripts and reference code deliberately remain as research
assets. Paths to omitted fixtures are not promises of a ready-to-run benchmark.
Public status documentation supersedes the historical catalog for integration
readiness. This repository does not republish the private business corpus.

Apache-2.0 was selected for the public project. Third-party code retains its
original attribution and licensing; project licensing does not relicense the
Gaudi SDK, runtime, model weights, or separately installed dependencies.
