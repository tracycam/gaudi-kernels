# Contributing

Contributions are submitted under Apache-2.0. Preserve existing copyright,
license, and attribution notices. Describe the origin of reused code.

Include the relevant device, SDK/compiler versions, shapes, precision policy,
and graph mode with a kernel change. Run CPU tests before submitting; device
results need actual device validation, including tails and representative
numerical edge cases. Mark untested paths explicitly.

Compare the complete operation or consuming chain with the current baseline.
Report warmup, context length, batch and per-expert row counts, timing boundaries,
and unsuccessful trials. A faster isolated kernel may slow the full model.

Do not commit credentials, remote connection details, private prompts, model
weights, compiled SDK components, deployment manifests, or raw experiment logs.
Keep those locally. Tests should use synthetic input. Use separately installed
toolchains and build outputs under ignored directories.

For host-side tests:

```bash
PYTHONPATH=python python3 -m pytest tests -q
```

Some tests compile a small SDK substitute and require `g++`. Passing these tests
does not establish HPU numerical correctness, model quality, or performance.
