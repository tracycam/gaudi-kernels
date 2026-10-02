Literal production comparator from tpc-dev/precision-fix-20260925.
The direct_down.s executable body is not regenerated from direct_down.c: compile
the template, assemble the supplied source, replace ELF .text. It stores FP32.
The combine restores the historical even/odd row ordering while applying routing
weights in FP32 slot order. Weight packing is reference/mxfp4/executor/kernels/
packing.py (historical unpermuted N512), not the newer native-v2 lane permutation.
This baseline has exactly two compute nodes and no preparation/finish node.
Shape contract remains N6144,K256; it is a comparator, not a new generic API.
