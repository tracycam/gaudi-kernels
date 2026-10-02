#pragma once
#include <synapse_api.h>
#include <cstdint>
#include <string>
#include <vector>
#include "mxfp4_moe_layout.hpp"
namespace gaudi_kernels::mxfp4_moe_bucket {
using WeightLayout=mxfp4_moe_layout::WeightLayout;
// Experimental graph, not production dispatch. No host read of counts.
struct Plan {
 WeightLayout weight_layout=WeightLayout::Unspecified;
 int expert_batch=2,n_tile=512;
 int scratch_buffers=1; // opt-in 2 allows, but does not prove, decode/MME overlap.
 uint64_t decoded_sram_limit=12ull*1024*1024;
 bool unique_experts_per_token=false,activation_per_token=false,gate_up_epilogue=false;
 bool scales_2_252=false,caller_certifies_fast_arithmetic=false;
};
struct Bucket {int lower,upper,expert_slots,slot_base,row_base;};
struct Budget {
 std::vector<Bucket> buckets;
 int expert_slots=0,row_slots=0,expert_batch=0,n_tile=0;
 uint64_t grouped_activation_bytes=0,fp32_output_bytes=0,decoded_sram_bytes=0,per_buffer_sram_bytes=0;
 unsigned decode_nodes=0,mme_nodes=0;
};
Budget budget(int n,int k,int tokens,int routes,int experts,const Plan&plan);
struct Resources {std::vector<synTensor> tensors;std::vector<synSectionHandle> sections;Budget budget;};
// Same aligned native-v2 byte layout as mxfp4_moe_grouped. W[256,K,E*N/512],
// S[512,K/32,E*N/512], A[K,T or T*R], IDs/routing[R,T], LUT[512,1].
// Counts, slot->expert map and route->flat-row inverse remain device tensors.
// Invalid expert IDs are masked. Unique top-k per token implies count<=T;
// any violation with count>T emits status[e]=1 and NaN for that expert's routes.
// status[E,1] must be a caller-visible output. No silent overflow fallback.
// Y[N,T] FP32 ordered combine, or BF16[N/2,T*R] literal GP boundary.
Resources append(synGraphHandle graph,synTensor packed,synTensor scales,
 synTensor activation,synTensor ids,synTensor routing,synTensor lut,
 synTensor output,synTensor status,int n,int k,int tokens,int routes,int experts,
 const Plan&plan,const std::string&prefix="mxfp4_bucket");
}
