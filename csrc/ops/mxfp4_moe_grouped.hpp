#pragma once
#include <synapse_api.h>
#include <cstdint>
#include <string>
#include <vector>
#include "mxfp4_moe_layout.hpp"
namespace gaudi_kernels::mxfp4_moe_grouped {
using WeightLayout=mxfp4_moe_layout::WeightLayout;
// Prototype: one E2/T2 linear+combine device smoke is qualified. GP/SiLU,
// broader batch shapes, capacity tradeoffs and model integration remain gates.
struct Plan {
 WeightLayout weight_layout=WeightLayout::Unspecified;
 int capacity=0; // 0 means T, never an inferred mean M_e.
 int expert_batch=4,n_tile=512;
 uint64_t decoded_sram_limit=12ull*1024*1024;
 bool unique_experts_per_token=false;
 bool activation_per_token=false; // GP shares X[K,T] across routes.
 bool gate_up_epilogue=false; // BF16[N/2,T*R], literal rounding boundaries.
 bool scales_2_252=false,caller_certifies_fast_arithmetic=false;
 // Smaller-than-T capacities are diagnostics, not a production fallback plan.
 bool allow_capacity_experiment=false;
};
struct Budget {
 int capacity=0,expert_batch=0,n_tile=0;
 uint64_t decoded_sram_bytes=0,grouped_activation_bytes=0;
 uint64_t fp32_expert_outputs_upper_bytes=0,original_weight_scale_bytes=0;
 uint64_t mme_row_slots=0;
 unsigned decode_nodes=0,mme_nodes=0,combine_nodes=0;
};
Budget budget(int n,int k,int tokens,int routes,int experts,const Plan&plan);
struct Resources {std::vector<synTensor> tensors;std::vector<synSectionHandle> sections;Budget budget;};
// FCD-first W=[256,K,E*N/512] raw uint8 native-v2, S=[512,K/32,E*N/512].
// A=[K,T*R] or [K,T] original BF16 per plan; IDs and routing=[R,T] int32/FP32; LUT=[512,1].
// Y=[N,T] FP32 combined, or [N/2,T*R] BF16 gate/up activation; mandatory overflow=[E,1] int32 are caller-owned outputs.
// All grouping and inverse metadata are device tensors. Invalid IDs are masked.
// CAP<T requires explicit experiment mode: overflowing tokens output NaN plus
// overflow flags, never an incomplete result advertised as a valid MoE output.
// No same-graph overflow fallback or framework/model integration is implemented.
Resources append(synGraphHandle graph,synTensor packed,synTensor scales,
 synTensor activation,synTensor ids,synTensor routing,synTensor lut,
 synTensor output,synTensor overflow,int n,int k,int tokens,int routes,int experts,
 const Plan&plan,const std::string&prefix="mxfp4_expert_batch");
}
