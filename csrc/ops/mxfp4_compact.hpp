#pragma once
#include <synapse_api.h>
#include <string>
#include <vector>
#include <cstdint>
namespace gaudi_kernels::mxfp4_compact {
// These append-only building blocks consume caller-owned views and SRAM scratch.
// All tensor dimensions below are FCD-first. A=[K,M], Y=[N,M], FP32 output.
// No device acquisition, compile, launch, runtime repacking or weight expansion
// in HBM. Caller owns sections and must chain dependencies before scratch reuse.
// All original compact segments must be partitioned once; use MME for M reuse.
enum class Kind { NativeBody, RawRows };
struct Tile {
 Kind kind=Kind::NativeBody;
 synTensor packed=nullptr,scales=nullptr,lut=nullptr,scratch=nullptr,activation=nullptr,output=nullptr;
 int n=0,k=0,m=0,native_blocks=0,block_offset=0,pair_offset=0;
 int row_k_tile=32;
 // Validated by compact-v3 loader (all original E8M0 codes in2..252).
 // This is separate from the arithmetic proof: exact BF16 decode alone does
 // not prevent an MME FP32 product/reduction from flushing a subnormal.
 bool scales_2_252=false;
 bool caller_certifies_mme_arithmetic=false;
};
struct Nodes { synNodeId decode=0,mme=0; };
Nodes append_decode_mme_tile(synGraphHandle graph,const Tile& tile,
                            const std::string& prefix="mxfp4_compact");
struct Group {
 synTensor packed=nullptr,scales=nullptr,activation=nullptr,output=nullptr,bias=nullptr;
 int n=0,k=0,m=0,layout_version=3;
 bool scales_2_252=false,caller_certifies_mme_arithmetic=false;
};
struct Plan {
 int native_n_tile=512;
 uint64_t scratch_limit=16ull*1024*1024;
 // 0 retains native_n_tile. Larger raw-row tiles can coalesce a short K tail
 // into fewer MME nodes without enlarging or rereading the original payload.
 int row_n_tile=0;
 int row_k_tile=32;
};
struct TilePlan { Kind kind; int n_begin,n_count,k_begin,k_count,block_offset,pair_offset; };
// Pure host planner, callable offline; no Synapse function or tensor needed.
std::vector<TilePlan> plan_tiles(int n,int k,const Plan& plan={});
struct Resources {
 std::vector<synTensor> tensors;
 std::vector<synSectionHandle> sections;
 uint64_t original_weight_scale_bytes=0,scratch_bytes=0;
 unsigned decode_nodes=0,mme_nodes=0;
};
// Flat uint8 compact-v3 blobs. Logical split/reshape nodes preserve original
// bytes; physical aliasing/placement must be audited on the target compiler.
// One shared scratch section is serialized across all tiles and groups. Bias
// is FP32, applied after the full-K FP32 sum, before any caller output cast.
Resources append_grouped(synGraphHandle graph,synTensor lut,
                         const std::vector<Group>& groups,const Plan& plan={},
                         const std::string& prefix="mxfp4_compact");
}
