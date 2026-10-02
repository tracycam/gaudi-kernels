#include "mxfp4_exact.hpp"
#include <cstdint>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
namespace gaudi_kernels::mxfp4_exact {
static void check(synStatus s,const char*what){if(s!=synSuccess)throw std::runtime_error(std::string(what)+": "+std::to_string(s));}
static void require(synTensor t,synDataType type,std::initializer_list<int64_t> dimensions){
 if(!t)throw std::invalid_argument("missing MXFP4 exact tensor");
 synTensorGeometry shape={};synDataType dtype;
 check(synTensorGetGeometry(t,&shape,synGeometryMaxSizes),"geometry");
 check(synTensorGetDeviceDataType(t,&dtype),"dtype");
 if(dtype!=type||shape.dims!=dimensions.size())throw std::invalid_argument("MXFP4 exact dtype/rank mismatch");
 unsigned i=0;for(auto size:dimensions){
  if(size<1||size>std::numeric_limits<int32_t>::max()||shape.sizes[i++]!=uint64_t(size))
   throw std::invalid_argument("MXFP4 exact shape/coordinate range mismatch");
 }
}
void append(synGraphHandle graph,synTensor packed,synTensor scales,synTensor activation,
            synTensor output,synTensor bias,int n,int k,int m,Layout layout,Mode mode,
            synTensor fast,const std::string&prefix){
 if(n<1||k<1||m<1)throw std::invalid_argument("positive MXFP4 exact dimensions required");
 if(layout==Layout::CompactV3){
  require(packed,syn_type_uint8,{int64_t(n)*((int64_t(k)+1)/2),1});
  require(scales,syn_type_uint8,{int64_t(n)*((int64_t(k)+31)/32),1});
 }else if(layout==Layout::NativeN512V2){
  int64_t blocks=(int64_t(n)+511)/512,prepared_k=(int64_t(k)+31)/32*32;
  if(blocks*prepared_k*256>std::numeric_limits<int32_t>::max())throw std::invalid_argument("N512 byte offset exceeds int32");
  require(packed,syn_type_packed_mxfp4,{256,blocks*prepared_k});
  require(scales,syn_type_uint8,{512,blocks*prepared_k/32});
 }else throw std::invalid_argument("unknown MXFP4 exact layout");
 require(activation,syn_type_bf16,{k,m});require(output,syn_type_single,{n,m});
 if(bias)require(bias,syn_type_single,{n,1});
 if(mode!=Mode::ExactOnce&&mode!=Mode::RepairFp32)throw std::invalid_argument("unknown MXFP4 arithmetic mode");
 if(mode==Mode::RepairFp32)require(fast,syn_type_single,{n,m});
 else if(fast)throw std::invalid_argument("exact-once does not consume a fast result");
 synTensor inputs[5]={packed,scales,activation};unsigned count=3;
 if(mode==Mode::RepairFp32)inputs[count++]=fast;
 if(bias)inputs[count++]=bias;
 std::string guid="gk_mxfp4_integer_";
 guid+=mode==Mode::ExactOnce?"exact_":"repair_";
 guid+=layout==Layout::CompactV3?"v3":"v2";
 if(bias)guid+="_bias";
 int params[]={n,k};
 check(synNodeCreate(graph,inputs,&output,count,1,params,sizeof(params),guid.c_str(),prefix.c_str(),nullptr,nullptr),"MXFP4 integer node");
}
PredispatchResources append_predispatch_native_m1(
 synGraphHandle graph,synTensor packed,synTensor scales,synTensor activation,
 synTensor lut,synTensor mapping,synTensor output,synTensor bias,int n,int k,
 int scale_min,int scale_max,int splits,bool force_exact,const std::string&prefix){
 if(n<1||k<1||n%512||k%32||scale_min<0||scale_max>254||scale_min>scale_max||splits<0)
  throw std::invalid_argument("predispatch requires aligned M1 and finite verified scale bounds");
 int nb=n/512,groups=k/32;
 if(!splits){splits=std::min(groups,std::max(1,(24+nb-1)/nb));while(groups%splits)--splits;}
 if(splits>groups||groups%splits)throw std::invalid_argument("predispatch splits must divide K32 groups");
 int64_t stored=int64_t(nb)*k,tasks=int64_t(nb)*splits;
 if(stored*256>std::numeric_limits<int32_t>::max()||tasks*512>std::numeric_limits<int32_t>::max())
  throw std::invalid_argument("predispatch offsets exceed int32");
 require(packed,syn_type_packed_mxfp4,{256,stored});require(scales,syn_type_uint8,{512,stored/32});
 require(activation,syn_type_bf16,{k,1});require(lut,syn_type_bf16,{512,1});
 require(mapping,syn_type_int32,{tasks,1});require(output,syn_type_single,{n,1});
 if(bias)require(bias,syn_type_single,{n,1});
 PredispatchResources resources;resources.splits=splits;
 auto temporary=[&](std::string name,synDataType type,int first,int second){
  synTensorDescriptor d={};d.m_name=name.c_str();d.m_dataType=type;d.m_dims=2;
  d.m_sizes[0]=d.m_minSizes[0]=first;d.m_sizes[1]=d.m_minSizes[1]=second;
  synTensor t;check(synTensorCreate(&t,&d,nullptr,0),"predispatch temporary");resources.tensors.push_back(t);return t;
 };
 synTensor copied=temporary(prefix+"_activation",syn_type_bf16,k/splits,tasks);
 synTensor flags=temporary(prefix+"_flags",syn_type_uint16,1,tasks);
 synTensor partials=temporary(prefix+"_partials",syn_type_single,tasks*512,1);
 synTensor prep_inputs[]={activation,bias},prep_outputs[]={copied,flags};
 int prep_params[]={n,k,nb,splits,scale_min,scale_max,int(force_exact)};
 check(synNodeCreate(graph,prep_inputs,prep_outputs,bias?2:1,2,prep_params,sizeof(prep_params),
  bias?"gk_mxfp4_dispatch_prepare_bias":"gk_mxfp4_dispatch_prepare",(prefix+"_prepare").c_str(),nullptr,nullptr),"predispatch prepare");
 synTensor fast_inputs[]={packed,scales,copied,lut,mapping,flags};
 check(synNodeCreate(graph,fast_inputs,&partials,6,1,nullptr,0,"gk_mxfp4_dispatch_history",(prefix+"_conditional_history").c_str(),nullptr,nullptr),"predispatch history");
 synTensor finish_inputs[]={packed,scales,activation,partials,flags,bias};int finish_params[]={n,k,splits};
 check(synNodeCreate(graph,finish_inputs,&output,bias?6:5,1,finish_params,sizeof(finish_params),
  bias?"gk_mxfp4_dispatch_finish_bias":"gk_mxfp4_dispatch_finish",(prefix+"_finish").c_str(),nullptr,nullptr),"predispatch finish");
 return resources;
}

}
