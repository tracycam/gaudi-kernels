# Closed comparison

The user changed the objective to feeding MME at 3.2 TB/s. No further DMA
comparison is planned. The completed experiment is retained for reproducibility.
`evidence/mxfp4-dma-prefetch/comparison.json` contains paired full-recipe timings,
placement/traffic audits, output hashes and overlap evidence. Raw assets and
their SHA manifest are local under `artifacts/builds/mxfp4-dma-prefetch-20260928/r1`.

Staging did not provide a substantial improvement: gate E32/M2 improved 1.52%,
gate E8/M2 regressed 3.12%, and down E8/M2 regressed 5.35%. This compares the
explicit two-slot builder, not the faster compiler-managed grouped baseline.
The production default remains direct packed HBM reads by TPC.
