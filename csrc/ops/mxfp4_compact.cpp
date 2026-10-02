#include "mxfp4_compact.hpp"
#include <synapse_common_types.hpp>
#include <initializer_list>
#include <stdexcept>
namespace gaudi_kernels::mxfp4_compact {
static void ck(synStatus s,const char* context){if(s!=synSuccess)throw std::runtime_error(std::string(context)+": Synapse "+std::to_string(s));}
static void require(synTensor t,synDataType dtype,std::initializer_list<int> sizes){
 if(!t)throw std::invalid_argument("missing compact tile tensor");
 synTensorGeometry geometry={};synDataType actual;
 ck(synTensorGetGeometry(t,&geometry,synGeometryMaxSizes),"geometry");
 ck(synTensorGetDeviceDataType(t,&actual),"dtype");
 if(actual!=dtype||geometry.dims!=sizes.size())throw std::invalid_argument("compact tile dtype/rank mismatch");
 unsigned d=0;for(int size:sizes)if(size<1||geometry.sizes[d++]!=unsigned(size))throw std::invalid_argument("compact tile shape mismatch");
}
Nodes append_decode_mme_tile(synGraphHandle graph,const Tile&t,const std::string&prefix){
 if(!graph||t.n<1||t.k<1||t.m<1)throw std::invalid_argument("bad compact decode/MME tile");
 if(t.kind!=Kind::NativeBody&&t.kind!=Kind::RawRows)throw std::invalid_argument("unknown compact tile kind");
 if(t.row_k_tile!=32&&t.row_k_tile!=256)throw std::invalid_argument("compact row K task must be32 or256");
 if(uint64_t(t.n)*t.m>INT32_MAX||uint64_t(t.k)*t.m>INT32_MAX)throw std::invalid_argument("compact tile activation/output elements must fit int32");
 if(!t.scales_2_252)throw std::invalid_argument("compact BF16 decoder requires E8M0 codes2..252; route extreme scales to exact compact reader");
 if(!t.caller_certifies_mme_arithmetic)throw std::invalid_argument("compact MME requires separate FP32 arithmetic certificate; scale eligibility is insufficient");
 require(t.lut,syn_type_bf16,{512,1});require(t.activation,syn_type_bf16,{t.k,t.m});require(t.output,syn_type_single,{t.n,t.m});
 int params[2]={t.block_offset,t.pair_offset};const char*guid;
 if(t.kind==Kind::NativeBody){
  if((t.n!=256&&t.n!=512)||t.k%32||t.native_blocks<1||t.block_offset<0||t.block_offset>=t.native_blocks||t.pair_offset<0||t.pair_offset>1||t.pair_offset+t.n/256>2)
    throw std::invalid_argument("bad compact native block/half");
  if(uint64_t(t.native_blocks)*t.k*256>INT32_MAX)throw std::invalid_argument("compact native source byte offsets must fit int32");
  require(t.packed,syn_type_uint8,{256,t.k,t.native_blocks});require(t.scales,syn_type_uint8,{512,t.k/32,t.native_blocks});
  require(t.scratch,syn_type_bf16,{t.n,t.k});guid="gk_mx4c3_native_s2_252_bf16";
 }else{
  if(t.k>=32&&t.k%32)throw std::invalid_argument("compact row view must be K32 aligned or final K1..31 tail");
  if(uint64_t(t.n)*((uint64_t(t.k)+1)/2)>INT32_MAX||uint64_t(t.n)*((uint64_t(t.k)+31)/32)>INT32_MAX)throw std::invalid_argument("compact row source byte offsets must fit int32");
  require(t.packed,syn_type_uint8,{int((uint64_t(t.k)+1)/2),t.n});require(t.scales,syn_type_uint8,{int((uint64_t(t.k)+31)/32),t.n});
  require(t.scratch,syn_type_bf16,{t.k,t.n});guid=t.row_k_tile==256?"gk_mx4c3_rows256_s2_252_bf16":"gk_mx4c3_rows_s2_252_bf16";
 }
 synSectionHandle section=nullptr;uint64_t offset=0;bool rmw=false,persistent=true;
 ck(synTensorGetSection(t.scratch,&section,&offset),"scratch section");
 if(!section)throw std::invalid_argument("compact decoded weight must have explicit SRAM section");
 ck(synSectionGetRMW(section,&rmw),"scratch RMW");ck(synSectionGetPersistent(section,&persistent),"scratch persistence");
 if(!rmw||persistent)throw std::invalid_argument("compact decoded weight must be nonpersistent RMW SRAM");
 if(uint64_t(t.n)*t.k*2>16ull*1024*1024)throw std::invalid_argument("compact decoded tile exceeds16MiB");
 Nodes nodes; synTensor di[]={t.packed,t.scales,t.lut};
 bool native=t.kind==Kind::NativeBody;
 ck(synNodeCreateWithId(graph,di,&t.scratch,3,1,native?params:nullptr,native?sizeof(params):0,guid,(prefix+"_decode").c_str(),&nodes.decode,nullptr,nullptr),"compact decode");
 synGEMMParams gp{false,!native};synTensor mi[]={t.activation,t.scratch};
 ck(synNodeCreateWithId(graph,mi,&t.output,2,1,&gp,sizeof(gp),"gemm",(prefix+"_mme").c_str(),&nodes.mme,nullptr,nullptr),"compact MME");
 return nodes;
}
}

#include <algorithm>
#include <limits>
namespace gaudi_kernels::mxfp4_compact {
std::vector<TilePlan> plan_tiles(int n,int k,const Plan&p){
 if(n<1||k<1||(p.native_n_tile!=256&&p.native_n_tile!=512)||(p.row_k_tile!=32&&p.row_k_tile!=256)||p.row_n_tile<0||p.scratch_limit<2||p.scratch_limit>16ull*1024*1024)
  throw std::invalid_argument("bad compact plan geometry/budget");
 if(uint64_t(n)*((uint64_t(k)+1)/2)>INT32_MAX||uint64_t(n)*((uint64_t(k)+31)/32)>INT32_MAX)
  throw std::invalid_argument("compact blob offsets must fit int32");
 int nf=n/512*512,kf=k/32*32;
 std::vector<TilePlan> result;
 if(nf&&kf){
  if(uint64_t(p.native_n_tile)*kf*2>p.scratch_limit)throw std::invalid_argument("compact native tile exceeds SRAM; select native_n_tile256 or smaller K");
  for(int start=0;start<nf;start+=p.native_n_tile)
   result.push_back({Kind::NativeBody,start,p.native_n_tile,0,kf,start/512,(start%512)/256});
 }
 auto rows=[&](int first,int count,int kb,int kc){
  if(!count||!kc)return;
  unsigned capacity=std::min<uint64_t>(p.row_n_tile?p.row_n_tile:p.native_n_tile,p.scratch_limit/(2ull*kc));
  if(!capacity)throw std::invalid_argument("compact row cannot fit SRAM");
  for(int start=first;start<first+count;){
   int chunk=std::min<int>(capacity,first+count-start);
   result.push_back({Kind::RawRows,start,chunk,kb,kc,0,0});start+=chunk;
  }
 };
 rows(nf,n-nf,0,kf);rows(0,n,kf,k-kf);
 return result;
}
Resources append_grouped(synGraphHandle graph,synTensor lut,const std::vector<Group>&groups,const Plan&p,const std::string&prefix){
 if(!graph)throw std::invalid_argument("missing compact graph");require(lut,syn_type_bf16,{512,1});
 Resources r;std::vector<std::vector<TilePlan>> plans;
 for(const auto&g:groups){
  if(g.layout_version!=3||g.m<0)throw std::invalid_argument("compact layout-v3 and nonnegative M required");
  auto tiles=plan_tiles(g.n,g.k,p);plans.push_back(tiles);
  if(!g.m)continue;
  if(!g.scales_2_252||!g.caller_certifies_mme_arithmetic)throw std::invalid_argument("compact fast route needs scale2..252 and separate arithmetic certificate");
  if(uint64_t(g.n)*g.m>INT32_MAX||uint64_t(g.k)*g.m>INT32_MAX)throw std::invalid_argument("compact activation/output tensor elements must fit int32");
  require(g.packed,syn_type_uint8,{int(uint64_t(g.n)*((uint64_t(g.k)+1)/2))});
  require(g.scales,syn_type_uint8,{int(uint64_t(g.n)*((uint64_t(g.k)+31)/32))});
  require(g.activation,syn_type_bf16,{g.k,g.m});require(g.output,syn_type_single,{g.n,g.m});if(g.bias)require(g.bias,syn_type_single,{g.n,1});
  for(const auto&t:tiles)r.scratch_bytes=std::max(r.scratch_bytes,uint64_t(t.n_count)*t.k_count*2);
 }
 synSectionHandle scratch=nullptr;
 if(r.scratch_bytes){ck(synSectionCreate(&scratch,0,graph),"compact scratch");ck(synSectionSetPersistent(scratch,false),"compact nonpersistent");ck(synSectionSetRMW(scratch,true),"compact SRAM");r.sections.push_back(scratch);}
 auto tensor=[&](const std::string&name,synDataType type,std::vector<int>dims,synSectionHandle section=nullptr){
  synTensorDescriptor d={};d.m_name=name.c_str();d.m_dataType=type;d.m_dims=dims.size();
  for(unsigned i=0;i<dims.size();++i)d.m_sizes[i]=d.m_minSizes[i]=dims[i];
  synTensor out;ck(synTensorCreate(&out,&d,section,0),"compact intermediate");r.tensors.push_back(out);return out;
 };
 auto node=[&](std::vector<synTensor>inputs,std::vector<synTensor>outputs,const void*params,unsigned bytes,const char*guid,const std::string&name){
  ck(synNodeCreate(graph,inputs.data(),outputs.data(),inputs.size(),outputs.size(),params,bytes,guid,name.c_str(),nullptr,nullptr),"compact graph node");
 };
 auto reshape=[&](synTensor input,synDataType type,std::vector<int>dims,const std::string&name){auto out=tensor(name,type,dims);node({input},{out},nullptr,0,"reshape",name+"_view");return out;};
 auto split=[&](synTensor input,synDataType type,std::vector<std::vector<int>>shapes,int axis,const std::string&name){
  if(shapes.size()==1)return std::vector<synTensor>{input};
  std::vector<synTensor>out;for(unsigned i=0;i<shapes.size();++i)out.push_back(tensor(name+"_"+std::to_string(i),type,shapes[i]));
  synSplitParams sp{unsigned(axis)};node({input},out,&sp,sizeof(sp),"split",name+"_view");return out;
 };
 synNodeId last_mme=0;bool previous=false;
 for(unsigned gi=0;gi<groups.size();++gi){const auto&g=groups[gi];if(!g.m)continue;
  const auto&tiles=plans[gi];std::string tag=prefix+"_g"+std::to_string(gi);
  int nf=g.n/512*512,kf=g.k/32*32,kt=g.k-kf,nt=g.n-nf;
  std::vector<int>kinds;std::vector<std::vector<int>>wshape,sshape;
  if(nf&&kf){kinds.push_back(0);wshape.push_back({nf*(kf/2)});sshape.push_back({nf*(kf/32)});}
  if(nt&&kf){kinds.push_back(1);wshape.push_back({nt*(kf/2)});sshape.push_back({nt*(kf/32)});}
  if(kt){kinds.push_back(2);wshape.push_back({g.n*((kt+1)/2)});sshape.push_back({g.n});}
  auto ws=split(g.packed,syn_type_uint8,wshape,0,tag+"_packed_regions");
  auto ss=split(g.scales,syn_type_uint8,sshape,0,tag+"_scale_regions");
  synTensor w[3]={},s[3]={};
  for(unsigned i=0;i<kinds.size();++i){int kind=kinds[i];
   std::vector<int>wd=kind==0?std::vector<int>{256,kf,nf/512}:kind==1?std::vector<int>{kf/2,nt}:std::vector<int>{(kt+1)/2,g.n};
   std::vector<int>sd=kind==0?std::vector<int>{512,kf/32,nf/512}:kind==1?std::vector<int>{kf/32,nt}:std::vector<int>{1,g.n};
   w[kind]=reshape(ws[i],syn_type_uint8,wd,tag+"_w_view"+std::to_string(kind));
   s[kind]=reshape(ss[i],syn_type_uint8,sd,tag+"_s_view"+std::to_string(kind));
  }
  synTensor xmain=g.activation,xtail=g.activation;
  if(kf&&kt){auto xs=split(g.activation,syn_type_bf16,{{kf,g.m},{kt,g.m}},0,tag+"_a_k");xmain=xs[0];xtail=xs[1];}
  // Split row segments by N only. Each view is consumed by exactly one decoder.
  std::vector<synTensor>rw[3],rs[3];unsigned row_index[3]={};
  for(int kind=1;kind<3;++kind)if(w[kind]){
   std::vector<std::vector<int>>wd,sd;
   for(const auto&t:tiles)if(t.kind==Kind::RawRows&&((t.k_begin>0||!kf)?2:1)==kind){wd.push_back({(t.k_count+1)/2,t.n_count});sd.push_back({(t.k_count+31)/32,t.n_count});}
   rw[kind]=split(w[kind],syn_type_uint8,wd,1,tag+"_row_w"+std::to_string(kind));
   rs[kind]=split(s[kind],syn_type_uint8,sd,1,tag+"_row_s"+std::to_string(kind));
  }
  std::vector<synTensor>ymain,ytail;
  for(unsigned ti=0;ti<tiles.size();++ti){const auto&t=tiles[ti];std::string name=tag+"_tile"+std::to_string(ti);
   bool tail=t.k_begin>0||!kf;int kind=t.kind==Kind::NativeBody?0:tail?2:1;
   Tile tile;tile.kind=t.kind;tile.n=t.n_count;tile.k=t.k_count;tile.m=g.m;tile.native_blocks=nf/512;tile.block_offset=t.block_offset;tile.pair_offset=t.pair_offset;
   if(kind==0){tile.packed=w[0];tile.scales=s[0];}else{unsigned i=row_index[kind]++;tile.packed=rw[kind][i];tile.scales=rs[kind][i];}
   tile.lut=lut;tile.activation=tail?xtail:xmain;tile.row_k_tile=p.row_k_tile;
   tile.scratch=tensor(name+"_decoded_sram",syn_type_bf16,kind==0?std::vector<int>{tile.n,tile.k}:std::vector<int>{tile.k,tile.n},scratch);
   tile.output=tensor(name+"_f32",syn_type_single,{tile.n,g.m});
   tile.scales_2_252=g.scales_2_252;tile.caller_certifies_mme_arithmetic=g.caller_certifies_mme_arithmetic;
   auto nodes=append_decode_mme_tile(graph,tile,name);
   if(previous)ck(synNodeDependencySet(graph,&last_mme,&nodes.decode,1,1),"compact scratch reuse");last_mme=nodes.mme;previous=true;
   (tail?ytail:ymain).push_back(tile.output);++r.decode_nodes;++r.mme_nodes;
  }
  auto concat=[&](const std::vector<synTensor>&ys,const std::string&name){if(ys.size()==1)return ys[0];auto out=tensor(name,syn_type_single,{g.n,g.m});synConcatenateParams cp{0};node(ys,{out},&cp,sizeof(cp),"concat",name+"_view");return out;};
  synTensor main=ymain.empty()?nullptr:concat(ymain,tag+"_main_f32"),tail=ytail.empty()?nullptr:concat(ytail,tag+"_tail_f32");
  if(main&&tail){auto full=tensor(tag+"_full_k_f32",syn_type_single,{g.n,g.m});node({main,tail},{full},nullptr,0,"add_fwd_f32",tag+"_sum_k_f32");main=full;}else if(!main)main=tail;
  if(g.bias)node({main,g.bias},{g.output},nullptr,0,"add_fwd_f32",tag+"_bias_f32");
  else node({main},{g.output},nullptr,0,"reshape",tag+"_output_view");
  r.original_weight_scale_bytes+=uint64_t(g.n)*((uint64_t(g.k)+1)/2)+uint64_t(g.n)*((uint64_t(g.k)+31)/32);
 }
 return r;
}
}
