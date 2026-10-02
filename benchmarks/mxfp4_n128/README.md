# N128 / adjacent-K packed TPC experiment

Private operator candidate only. Preserve original E2M1 codes and E8M0 scales,
BF16 activation, FP32 MAC/block-scale/partial output. Repack nibble pairs along
K at load time; each N128 task has one accumulator per row rather than two.
The weight storage remains exactly 17/32 bytes per parameter. This is a
logical owner/access argument, not a physical HBM transaction measurement.

`build.py --rows 2 --output builds/n128-m2` builds the candidate.
Add `--control` for the original N256 grouped implementation built from the
same committed source and independent fixture/oracle/timer. `--branch` emits
actual per-count code paths and initializes all inactive output rows, without
MACs for those rows; inspect the ELF to verify the compiler retains branches.
`--n256 --branch` isolates genuine count dispatch without the N128 repack.
The `mixed3` fixture uses 20 experts, 6/8/6 experts with M=1/2/3, respectively,
40 valid rows in a cap4 tensor. It is synthetic grouped input, not a recorded
model route distribution. The original `mixed` pattern also covers M4.

Use the existing exclusive module probe runner and pipeline launch wrapper.
Compare GP N512/K6144 and down N6144/K256 separately; count reduction and
rotating original owners >48 MiB. A projection win is only an intermediate
gate: full MoE versus production, including prep/gate/combine, is mandatory
before model performance testing. No model-default changes here.

All sources, ELF/ISA, fixture bytes, poisoned-output checks, failures and
unfiltered timings must be copied back to the local asset directory.
