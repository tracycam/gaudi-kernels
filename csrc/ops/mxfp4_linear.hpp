#pragma once
#include <synapse_api.h>
#include <cstdint>
#include <string>
#include <vector>
namespace gaudi_kernels::mxfp4 {
// The caller owns graph, persistent tensors, LUT, device and tensor lifetimes.
// Original weight storage only: [ceil(N/256), K,128] packed bytes and
// [ceil(N/256),ceil(K/32),256] E8M0 bytes in host-major notation.
// Validate scale codes [2,252] at packing/loading time. No activation conversion.
// Layout-v1 is N256; historical N512 tensors must be unpacked/repacked first.
// Tensor rank/dimensions/dtypes are checked at graph construction, without D2H.
struct Group { synTensor packed=nullptr,scales=nullptr,activation=nullptr,output=nullptr,bias=nullptr; int m=0,n=0,k=0; };
struct Plan { int n_tile=512; uint64_t scratch_limit=12ull*1024*1024; bool tpc_gemv=false; unsigned scratch_buffers=1; bool unified_scratch_section=false; bool prefetch_packed_to_sram=false; };
// scratch_limit bounds each decoded tile; scratch_buffers=2 requests two slots.
// Total requested RMW storage is limited to 16 MiB. Two slots permit overlap;
// the compiler may reorder whole slot chains, so throughput is shape dependent.
// Opt-in unified_scratch_section keeps disjoint slots in one RMW section, so
// the compiler can bundle producer/consumer pairs across slot boundaries.
// Experimental prefetch_packed_to_sram stages original packed bytes + E8M0 using
// DMA, with separate live ranges from decoded weights. Requires unified scratch,
// one full unpadded expert per tile (N%256=K%32=0, N<=n_tile), and an MME plan.
// Both packed and decoded buffers count against the aggregate 16 MiB budget.
struct Resources {
 std::vector<synTensor> tensors; std::vector<synSectionHandle> sections;
 uint64_t scratch_bytes=0,logical_weight_bytes=0; unsigned mme_nodes=0,decode_nodes=0,empty_groups=0,dma_nodes=0;
};
// Append all experts to ONE caller graph; no acquire/compile/launch/synchronize.
// Host-known plans only: changing group sizes requires a new graph plan.
Resources append_grouped(synGraphHandle graph,synTensor lut,const std::vector<Group>& groups,
                         const Plan& plan={},const std::string& prefix="mxfp4");
}
