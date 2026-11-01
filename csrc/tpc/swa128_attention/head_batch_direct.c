// Fixed two-entry-per-row metadata, with the same FP32 attention arithmetic.
#define BATCH_WINDOW 1
#define BATCH_DIRECT 1
#define AV_GROUP_HOIST 1
#include "../swa128_ilp/head_quad.c"
