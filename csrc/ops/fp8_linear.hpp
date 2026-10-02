#pragma once
#include <synapse_api.h>
#include <cstdint>
#include <string>
#include <vector>

namespace gaudi_kernels {
enum class FP8Activation { PerTokenE4M3, BF16 };
enum class FP8ExperimentalQuantizer { Existing, IntegerAmax, LutSingle, LutFour };
// Inputs: A BF16 [M,K], W prepared native FP8 [N,K], channel scales
// FP32 [N], bias FP32 [N] (zero for no bias), output BF16 [M,N].
// Synapse fastest dimension comes first. Finite E4M3FN weights only.
// The caller owns graph/device, persistent tensor allocation and launch.
struct FP8LinearSpec {
 uint32_t m,n,k;
 FP8Activation activation;
 uint64_t rmw_budget_bytes=0; // queried by caller; 0 uses compiler placement
 std::string prefix="fp8_linear";
 uint32_t decode_row_block=8; // measured scheduling candidate: 8 or 128
 bool weight_n_contiguous=false; // prepared physical [K,N], GEMM B transpose=false
 bool reused_scale_epilogue=false; // four rows share FP32 channel scale/bias vectors
 bool interleaved_epilogue=false;
 bool native_quant_conversion=false; // same arithmetic policy; native normal-range conversion
 bool force_intermediate_sram=false; // optional placement plan; fails if budget is insufficient
 FP8ExperimentalQuantizer experimental_quantizer=FP8ExperimentalQuantizer::Existing;
 // Caller-owned immutable FP32[129,1] input, required only by LUT candidates.
 synTensor reciprocal_table=nullptr;
};
struct FP8LinearBuild {
 std::vector<synTensor> intermediates;
 synSectionHandle rmw_section=nullptr;
 uint64_t rmw_bytes=0;
};
// One-time host preparation: output remains one byte per weight. E4M3FN
// magnitude >=16 subtracts one exponent; smaller magnitudes use RNE /2.
// Multiply the original per-channel/tensor scale by 2. Does not mutate input.
bool prepare_fp8_linear(const uint8_t* original,uint8_t* prepared,size_t count);
// Preferred per-channel adapter avoids rounding small weights in channels
// without OCP extended codes 120..126. Channel scale remains separable.
bool prepare_fp8_linear_channels(const uint8_t* original,uint8_t* prepared,
 const float* scales,float* prepared_scales,uint32_t n,uint32_t k);
void transpose_prepared_fp8_linear(const uint8_t* nk,uint8_t* kn,uint32_t n,uint32_t k);
synStatus add_fp8_linear(synGraphHandle graph,const FP8LinearSpec& spec,
 synTensor activation,synTensor weight,synTensor prepared_weight_scales,
 synTensor bias,synTensor output,FP8LinearBuild* build);
}
