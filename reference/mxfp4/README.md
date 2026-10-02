# Historical MXFP4 source bundle

This is a model-specialized reference, not a newly qualified generic backend.
See `../../evidence/reports/mxfp4-unified-batch/README.md` for the historical
scope, successful checks, rejected full-batch numerical cases and regressions.

The build script creates an ELF from `inputs/template.c`, assembles the edited
`inputs/gemv.s`, then replaces the ELF `.text` section. The C template alone is
therefore not the final executable arithmetic. In the imported precision-fixed
assembly the final partial stores are FP32; copying only the template can
reintroduce an earlier BF16 boundary.

The current direct GP/down addressing and glue retain MiMo hidden6144/localI256
assumptions. Multiple token rows still execute TPC GEMV tasks. This does not
establish an efficient grouped MME GEMM implementation or new-host qualification.

Do not run this historical build in place: it regenerates source/assembly files.
Use an isolated experiment copy. Keep packing, template, assembly, glue and
build script together, and re-establish their output contract before promotion.
