// Four independent QK tokens and 16-token AV broadcast groups, exact FP32 order.
#define BATCH_WINDOW 1
#define AV_GROUP_HOIST 1
#include "../swa128_ilp/head_quad.c"
