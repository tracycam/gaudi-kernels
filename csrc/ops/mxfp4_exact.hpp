#pragma once
#include <synapse_api.h>
#include <string>
#include <vector>
namespace gaudi_kernels::mxfp4_exact {
enum class Layout { NativeN512V2=2, CompactV3=3 };
enum class Mode { ExactOnce, RepairFp32 };
// Finite BF16 A, E2M1, finite E8M0 0..254, optional finite FP32 bias.
// ExactOnce sums original products+bias exactly then rounds once, including
// FP32 subnormals. True overflow rounds to infinity. Invalid inputs yield qNaN.
// RepairFp32 copies a supplied equivalent FP32 fast result only under a proven
// no-FTZ/no-overflow guard; ordinary FP32 rounding is allowed in that mode.
// RepairFp32 rereads weights and is NOT a one-pass HBM solution.
// NativeN512V2 uses [256,Nblocks*Kprepared] packed_mxfp4 + [512,Nblocks*Kprepared/32]
// uint8 scales. CompactV3 uses flat uint8 blobs, Synapse [bytes,1], no padding.
// Activation/output are [K,M]/[N,M]. All coordinates/blob byte offsets fit int32.
// Only adds a node to the caller's graph. No acquire, compile, launch or sync.
void append(synGraphHandle graph,synTensor packed,synTensor scales,synTensor activation,
            synTensor output,synTensor bias,int n,int k,int m,Layout layout,
            Mode mode=Mode::ExactOnce,synTensor fast=nullptr,
            const std::string& prefix="mxfp4_integer");
struct PredispatchResources {std::vector<synTensor> tensors;int splits=0;};
// Minimal closed route: M1, N%512==0, K%32==0, native-v2 tensor descriptors.
// Aligned compact-v3 payloads have identical bytes, but the caller must create
// explicit native descriptor views; this function never silently reshapes/copies.
// scale_min/max are trusted load-time bounds over ALL supplied scales. Unknown
// bounds may be [0,254], which safely selects exact. No scale tensor pre-scan.
// M1 mapping MUST satisfy mapping[t]=t; an old dynamic/expert map is invalid.
// All splits share a flag. Unsafe: historical body reads no weights; exact
// finish alone reads each packed/scale byte once logically. No M>1 reuse claim.
PredispatchResources append_predispatch_native_m1(
 synGraphHandle graph,synTensor packed,synTensor scales,synTensor activation,
 synTensor lut,synTensor mapping,synTensor output,synTensor bias,int n,int k,
 int scale_min,int scale_max,int splits=0,bool force_exact=false,
 const std::string& prefix="mxfp4_predispatch");
}
