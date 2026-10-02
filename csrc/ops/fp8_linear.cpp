#include "fp8_linear.hpp"
#include <synapse_common_types.hpp>
#include <limits>
#include <cmath>
namespace gaudi_kernels {
bool prepare_fp8_linear(const uint8_t* original,uint8_t* prepared,size_t count) {
 if(!original||!prepared)return false;
 for(size_t i=0;i<count;++i)if((original[i]&127)==127)return false;
 for(size_t i=0;i<count;++i){unsigned c=original[i],a=c&127;
  unsigned h=a>=16?a-8:(a>>1)+((a&1)&&((a>>1)&1));prepared[i]=(c&128)|h;}
 return true;
}
bool prepare_fp8_linear_channels(const uint8_t* original,uint8_t* prepared,
 const float* scales,float* prepared_scales,uint32_t n,uint32_t k) {
 if(!original||!prepared||!scales||!prepared_scales||!n||!k)return false;
 for(uint64_t i=0;i<uint64_t(n)*k;++i)if((original[i]&127)==127)return false;
 for(uint32_t row=0;row<n;++row)if(!std::isfinite(scales[row])||scales[row]<0)return false;
 for(uint32_t row=0;row<n;++row){bool adapt=false;
  for(uint32_t col=0;col<k;++col)adapt|=(original[uint64_t(row)*k+col]&127)>119;
  if(adapt&&scales[row]>std::numeric_limits<float>::max()/2)return false;
  prepared_scales[row]=scales[row]*(adapt?2.f:1.f);
  for(uint32_t col=0;col<k;++col){unsigned c=original[uint64_t(row)*k+col],a=c&127;
   unsigned h=a>=16?a-8:(a>>1)+((a&1)&&((a>>1)&1));prepared[uint64_t(row)*k+col]=adapt?((c&128)|h):c;}
 }return true;
}
void transpose_prepared_fp8_linear(const uint8_t* nk,uint8_t* kn,uint32_t n,uint32_t k) {
 for(uint32_t row=0;row<n;++row)for(uint32_t col=0;col<k;++col)kn[uint64_t(col)*n+row]=nk[uint64_t(row)*k+col];
}
synStatus add_fp8_linear(synGraphHandle graph,const FP8LinearSpec&s,
 synTensor x,synTensor w,synTensor ws,synTensor bias,synTensor y,FP8LinearBuild*b) {
 if(!b||!s.m||!s.n||!s.k||!x||!w||!ws||!bias||!y)return synInvalidArgument;
 if(s.decode_row_block!=8&&s.decode_row_block!=128)return synInvalidArgument;
 const bool a8=s.activation==FP8Activation::PerTokenE4M3;
 const bool lut=s.experimental_quantizer==FP8ExperimentalQuantizer::LutSingle||s.experimental_quantizer==FP8ExperimentalQuantizer::LutFour;
 if(lut&&(!a8||!s.reciprocal_table))return synInvalidArgument;
 auto align=[](uint64_t v){return (v+255)&~uint64_t(255);};
 uint64_t pb=align(uint64_t(s.m)*s.n*4),ab=align(uint64_t(s.m)*s.k),sb=align(uint64_t(s.m)*4);
 bool rmw=a8&&pb+ab+sb<=s.rmw_budget_bytes;
 uint64_t wb=align(uint64_t(s.n)*s.k*2);
 bool rmw_p=rmw||(!a8&&s.force_intermediate_sram&&pb+wb<=s.rmw_budget_bytes);
 if(s.force_intermediate_sram&&((a8&&!rmw)||(!a8&&!rmw_p)))return synInvalidArgument;
 synStatus status=synSuccess;
#define TRY(call) do{status=(call);if(status!=synSuccess)return status;}while(0)
 if(rmw_p){TRY(synSectionCreate(&b->rmw_section,0,graph));TRY(synSectionSetPersistent(b->rmw_section,false));TRY(synSectionSetRMW(b->rmw_section,true));b->rmw_bytes=rmw?pb+ab+sb:pb+wb;}
 auto tensor=[&](const char*label,synDataType dtype,uint32_t d0,uint32_t d1,uint64_t offset,bool use_rmw){
  synTensorDescriptor d={};std::string name=s.prefix+"_"+label;d.m_name=name.c_str();d.m_dataType=dtype;d.m_dims=2;
  d.m_sizes[0]=d.m_minSizes[0]=d0;d.m_sizes[1]=d.m_minSizes[1]=d1;
  synTensor t=nullptr;status=synTensorCreate(&t,&d,use_rmw?b->rmw_section:nullptr,offset);
  if(status!=synSuccess)return t;
  b->intermediates.push_back(t);
  if(dtype==syn_type_fp8_143){synFpQuantParam p{1.0,7};synFpQuantMetadata m{dtype,&p,1};status=synTensorSetQuantizationData(t,SYN_FP_QUANT_METADATA,&m,sizeof(m));}return t;
 };
 synTensor p=tensor("unscaled_accumulation",syn_type_single,s.n,s.m,0,rmw_p);if(status!=synSuccess)return status;
 synGEMMParams params{false,!s.weight_n_contiguous};
 auto node=[&](synTensor*in,unsigned count,synTensor*out,unsigned outputs,const char*guid,const char*name,void*parameters=nullptr,unsigned bytes=0){
  std::string fullname=s.prefix+"_"+name;return synNodeCreate(graph,in,out,count,outputs,parameters,bytes,guid,fullname.c_str(),nullptr,nullptr);
 };
 if(a8){
  synTensor a=tensor("native_activation",syn_type_fp8_143,s.k,s.m,pb,rmw);if(status!=synSuccess)return status;
  synTensor as=tensor("activation_scales",syn_type_single,1,s.m,pb+ab,rmw);if(status!=synSuccess)return status;
  const char* quant_guid=s.native_quant_conversion?"fp8_linear_activation_fast":"fp8_linear_activation";
  switch(s.experimental_quantizer){
   case FP8ExperimentalQuantizer::Existing:break;
   case FP8ExperimentalQuantizer::IntegerAmax:quant_guid="fp8_fp_quant_amax16";break;
   case FP8ExperimentalQuantizer::LutSingle:quant_guid="fp8_fp_quant_lut_single";break;
   case FP8ExperimentalQuantizer::LutFour:quant_guid="fp8_fp_quant_lut_single4";break;
  }
  synTensor qi[]={x,s.reciprocal_table},qo[]={a,as};TRY(node(qi,lut?2:1,qo,2,quant_guid,"quantize"));
  synTensor mi[]={a,w};TRY(node(mi,2,&p,1,"gemm","fp8_mme",&params,sizeof(params)));
  synTensor ei[]={p,as,ws,bias};TRY(node(ei,4,&y,1,s.reused_scale_epilogue?"fp8_linear_epilogue8_rows":s.interleaved_epilogue?"fp8_linear_epilogue8_fast":"fp8_linear_epilogue8","scale_bias"));
 }else{
  synTensor d=tensor("decoded_weight",syn_type_bf16,s.weight_n_contiguous?s.n:s.k,s.weight_n_contiguous?s.k:s.n,rmw_p?pb:0,rmw_p);if(status!=synSuccess)return status;
  TRY(node(&w,1,&d,1,s.decode_row_block==128?"fp8_linear_decode128":"fp8_linear_decode","decode"));
  synTensor mi[]={x,d};TRY(node(mi,2,&p,1,"gemm","bf16_mme",&params,sizeof(params)));
  synTensor ei[]={p,ws,bias};TRY(node(ei,3,&y,1,s.reused_scale_epilogue?"fp8_linear_epilogue16_rows":s.interleaved_epilogue?"fp8_linear_epilogue16_fast":"fp8_linear_epilogue16","scale_bias"));
 }
#undef TRY
 return synSuccess;
}
}
