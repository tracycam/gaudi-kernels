#pragma once
#include <synapse_api.h>
namespace gaudi_kernels::experimental::fp8_first_principles {
enum class Quantizer { Amax16, LutCorrected, LutSingle, LutSingle4 };
// Adds one node to the caller's graph. No initialization, acquisition or launch.
// x BF16[K,M], q nativeF8[K,M] with unit tensor scale/exponent bias7,
// scales F32[1,M], table F32[129,1]. Returned scales include native-half x2.
// table contents must match cpu_gate.py's certificate. Finite BF16 inputs only.
// q/scales can fan out to several MMEs if their liveness fits the chosen graph.
synStatus add_quantizer(synGraphHandle graph,Quantizer kind,synTensor x,
                       synTensor table,synTensor q,synTensor scales);
}
