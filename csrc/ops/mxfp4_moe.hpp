#pragma once
#include <synapse_api.h>
#include <string>
#include <vector>
namespace gaudi_kernels::mxfp4_moe {
enum class Engine { Direct512, Fused512, Fused256, Fused256C };
struct Plan { Engine engine=Engine::Direct512; bool caller_certifies_fast_arithmetic=false; bool allow_unqualified_experiments=false; };
struct Resources {std::vector<synTensor> tensors;unsigned nodes=0;};
// Graph-local down+FP32 routing combine, no acquire/compile/launch or host routing.
// FCD-first tensors: packed=[256,E*(N/512)*K], scales=[512,E*(N/512)*(K/32)],
// activation=[K,T*R] BF16, ids=[R,T] int32, routing=[R,T] FP32,
// lut=[512,1] BF16, output=[N,T] FP32. Native-v2 bytes only, N%512=K%32=0.
// Invalid IDs are masked to zero; slots are combined in ascending order with
// FP32 FMA, after each complete down dot. No routing quantization or atomics.
// Fast-arithmetic certification includes historical unscaled K32 accumulation,
// scaled sum, and routing-FMA normal/finite bounds. This is not exact fallback.
// Fused engines are experimental and require allow_unqualified_experiments;
// current measured candidates regress or fail, so no production auto-selection.
// Each route reloads its expert weights: this TPC path makes no M_e reuse claim.
Resources append_routed_down(synGraphHandle graph,synTensor packed,synTensor scales,
 synTensor activation,synTensor ids,synTensor routing,synTensor lut,synTensor output,
 int n,int k,int tokens,int routes,int experts,const Plan&plan={},
 const std::string&prefix="mxfp4_moe");
}
