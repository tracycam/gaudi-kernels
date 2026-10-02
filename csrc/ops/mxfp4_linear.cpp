#include "mxfp4_linear.hpp"
#include <synapse_common_types.hpp>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
namespace gaudi_kernels::mxfp4 {
static void ck(synStatus status,const char* context){if(status!=synSuccess)throw std::runtime_error(std::string(context)+": Synapse "+std::to_string(status));}
static void require_tensor(synTensor tensor,synDataType dtype,std::initializer_list<int> sizes,const char* label){
 synTensorGeometry geometry={};synDataType actual;
 ck(synTensorGetGeometry(tensor,&geometry,synGeometryMaxSizes),"tensor geometry");
 ck(synTensorGetDeviceDataType(tensor,&actual),"tensor dtype");
 if(actual!=dtype||geometry.dims!=sizes.size())throw std::invalid_argument(std::string(label)+": incompatible MXFP4 layout-v1 dtype/rank");
 unsigned dim=0;for(int size:sizes)if(geometry.sizes[dim++]!=unsigned(size))throw std::invalid_argument(std::string(label)+": incompatible MXFP4 layout-v1 shape");
}
Resources append_grouped(synGraphHandle graph,synTensor lut,const std::vector<Group>& groups,const Plan& p,const std::string& prefix){
 Resources r;
 uint64_t packed_stride=0,scale_stride=0;
 if(p.scratch_buffers<1||p.scratch_buffers>2)throw std::invalid_argument("MXFP4 scratch_buffers must be 1 or 2");
 if(p.prefetch_packed_to_sram&&(p.tpc_gemv||!p.unified_scratch_section))throw std::invalid_argument("packed prefetch requires unified SRAM MME plan");
 if(!graph||!lut||p.n_tile<256||p.n_tile%256||!p.scratch_limit)throw std::invalid_argument("bad MXFP4 graph/plan");
 require_tensor(lut,syn_type_bf16,{512,1},"byte LUT");
 for(const auto&g:groups){
  if(g.m<0||g.n<1||g.k<1)throw std::invalid_argument("bad MXFP4 dimensions");
  if(!g.m){++r.empty_groups;continue;}
  if(p.prefetch_packed_to_sram){
   if(g.n>p.n_tile||g.n%256||g.k%32)throw std::invalid_argument("packed prefetch requires one full unpadded expert per tile");
   packed_stride=std::max(packed_stride,uint64_t(g.n)*g.k/2);
   scale_stride=std::max(scale_stride,uint64_t(g.n)*(g.k/32));
  }
  if(!g.packed||!g.scales||!g.activation||!g.output)throw std::invalid_argument("missing MXFP4 tensor");
  require_tensor(g.packed,syn_type_uint8,{128,g.k,(g.n+255)/256},"packed weights");
  require_tensor(g.scales,syn_type_uint8,{256,(g.k+31)/32,(g.n+255)/256},"E8M0 scales");
  require_tensor(g.activation,syn_type_bf16,{g.k,g.m},"activation");
  require_tensor(g.output,syn_type_single,{g.n,g.m},"output");
  if(g.bias)require_tensor(g.bias,syn_type_single,{g.n,1},"bias");
  uint64_t scratch=uint64_t(std::min(p.n_tile,g.n))*g.k*2;
  if(!p.tpc_gemv&&scratch>p.scratch_limit)throw std::invalid_argument("MXFP4 decoded tile exceeds explicit SRAM budget");
  r.scratch_bytes=std::max(r.scratch_bytes,p.tpc_gemv?uint64_t(0):scratch);
 }
 const unsigned buffers=p.scratch_buffers;
 if(r.scratch_bytes+packed_stride+scale_stride>16ull*1024*1024/buffers)throw std::invalid_argument("MXFP4 aggregate scratch exceeds 16 MiB RMW");
 std::vector<synSectionHandle> scratch(buffers,nullptr);
 if(r.scratch_bytes)for(unsigned i=0;i<buffers;++i){auto&section=scratch[i];if(i&&p.unified_scratch_section){section=scratch[0];continue;}ck(synSectionCreate(&section,0,graph),"scratch section");ck(synSectionSetPersistent(section,false),"scratch nonpersistent");ck(synSectionSetRMW(section,true),"scratch SRAM");r.sections.push_back(section);}
 const uint64_t scratch_stride=r.scratch_bytes;
 const uint64_t packed_base=scratch_stride*buffers,scale_base=packed_base+packed_stride*buffers;
 r.scratch_bytes=(r.scratch_bytes+packed_stride+scale_stride)*buffers;
 auto tensor=[&](std::string name,synDataType dt,int d0,int d1,synSectionHandle section=nullptr,int d2=0,uint64_t section_offset=0){synTensorDescriptor d={};d.m_name=name.c_str();d.m_dataType=dt;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=d0;d.m_sizes[1]=d.m_minSizes[1]=d1;if(d2){d.m_dims=3;d.m_sizes[2]=d.m_minSizes[2]=d2;}synTensor t;ck(synTensorCreate(&t,&d,section,section_offset),"intermediate");r.tensors.push_back(t);return t;};
 std::vector<synNodeId> last_mme(buffers,0);
 std::vector<synNodeId> last_decode(buffers,0);
 std::vector<bool> have_previous(buffers,false);
 unsigned tile_index=0;
 for(size_t e=0;e<groups.size();++e){const auto&g=groups[e];if(!g.m)continue;std::string tag=prefix+"_e"+std::to_string(e);
  r.logical_weight_bytes+=uint64_t((g.n+255)/256)*128*g.k+uint64_t((g.n+255)/256)*256*((g.k+31)/32);
  synTensor raw=g.bias?tensor(tag+"_prebias",syn_type_single,g.n,g.m):g.output;
  if(p.tpc_gemv){
   int tasks=((g.n+255)/256)*g.m;
   int splits=std::min((g.k+31)/32,std::max(1,(24+tasks-1)/tasks));
   // Always give GEMV a 3D FP32 output. Each split owns disjoint K groups;
   // the reduction never rounds partials through BF16.
   synTensor partial=tensor(tag+"_partials_f32",syn_type_single,g.n,g.m,nullptr,splits);
   synTensor in[]={g.packed,g.scales,lut,g.activation};ck(synNodeCreate(graph,in,&partial,4,1,nullptr,0,"gk_mxfp4_gemv_f32_v1",(tag+"_gemv").c_str(),nullptr,nullptr),"gemv");
   ck(synNodeCreate(graph,&partial,&raw,1,1,nullptr,0,"gk_mxfp4_reduce_f32_v1",(tag+"_reduce").c_str(),nullptr,nullptr),"FP32 split reduction");
  }
  else{
   std::vector<synTensor> ys;
   for(int start=0;start<g.n;start+=p.n_tile){int n=std::min(p.n_tile,g.n-start);std::string suffix=tag+"_n"+std::to_string(start);
    synTensor d=tensor(suffix+"_decoded_sram",syn_type_bf16,n,g.k,scratch[tile_index%buffers],0,p.unified_scratch_section?(tile_index%buffers)*scratch_stride:0);
    synTensor y=g.n<=p.n_tile?raw:tensor(suffix+"_result",syn_type_single,n,g.m);ys.push_back(y);
    synTensor packed=g.packed,scales=g.scales;
    unsigned current_slot=tile_index%buffers;
    if(p.prefetch_packed_to_sram){
     packed=tensor(suffix+"_packed_sram",syn_type_uint8,128,g.k,scratch[0],g.n/256,packed_base+current_slot*packed_stride);
     scales=tensor(suffix+"_e8m0_sram",syn_type_uint8,256,g.k/32,scratch[0],g.n/256,scale_base+current_slot*scale_stride);
     synNodeId copies[2];
     ck(synNodeCreateWithId(graph,&g.packed,&packed,1,1,nullptr,0,"memcpy",(suffix+"_prefetch_packed").c_str(),&copies[0],nullptr,nullptr),"prefetch packed");
     ck(synNodeCreateWithId(graph,&g.scales,&scales,1,1,nullptr,0,"memcpy",(suffix+"_prefetch_scales").c_str(),&copies[1],nullptr,nullptr),"prefetch scales");
     if(have_previous[current_slot])ck(synNodeDependencySet(graph,&last_decode[current_slot],copies,1,2),"packed slot reuse after decode");
     r.dma_nodes+=2;
    }
    synTensor di[]={packed,scales,lut};int offset=start/256;synNodeId decode,mme;
    ck(synNodeCreateWithId(graph,di,&d,3,1,&offset,sizeof(offset),"gk_mxfp4_decode_bf16_v1",(suffix+"_decode").c_str(),&decode,nullptr,nullptr),"decode");
    synGEMMParams gp{false,false};synTensor mi[]={g.activation,d};
    ck(synNodeCreateWithId(graph,mi,&y,2,1,&gp,sizeof(gp),"gemm",(suffix+"_mme").c_str(),&mme,nullptr,nullptr),"MME");
    unsigned slot=tile_index++%buffers;
    if(have_previous[slot])ck(synNodeDependencySet(graph,&last_mme[slot],&decode,1,1),"scratch reuse dependency");
    last_mme[slot]=mme;last_decode[slot]=decode;have_previous[slot]=true;++r.mme_nodes;++r.decode_nodes;
   }
   if(ys.size()>1){synConcatenateParams cp{0};ck(synNodeCreate(graph,ys.data(),&raw,ys.size(),1,&cp,sizeof(cp),"concat",(tag+"_concat").c_str(),nullptr,nullptr),"concat");}
  }
  if(g.bias){synTensor bi[]={raw,g.bias};synTensor out=g.output;ck(synNodeCreate(graph,bi,&out,2,1,nullptr,0,"gk_mxfp4_bias_f32_v1",(tag+"_bias").c_str(),nullptr,nullptr),"bias");}
 }
 return r;
}
}
