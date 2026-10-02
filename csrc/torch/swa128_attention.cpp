// Public graph contribution only. No private OpBackend or independent device/recipe.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cmath>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta metadata(const at::Stack&s,bool window){
 auto q=s[0].toTensor(),k=s[1].toTensor(),v=s[2].toTensor(),p=s[3].toTensor(),st=s[4].toTensor(),pos=s[5].toTensor(),sink=s[6].toTensor();double scale=s[7].toDouble();
 for(auto t:{q,k,v,sink})TORCH_CHECK(t.scalar_type()==at::kBFloat16&&t.is_contiguous()&&!t.requires_grad(),"SWA128 BF16 contiguous inference operands required");
 for(auto t:{p,st,pos})TORCH_CHECK((t.scalar_type()==at::kInt||(window&&t.scalar_type()==at::kLong))&&t.is_contiguous(),"SWA128 metadata must be contiguous I32");
 for(auto t:{k,v,p,st,pos,sink})TORCH_CHECK(t.device()==q.device(),"SWA128 device mismatch");
 TORCH_CHECK(q.dim()==3&&q.size(0)==1,"SWA128 requires exactly one decode query");
 int64_t h=q.size(2)==192?q.size(1):q.size(2)/192;
 TORCH_CHECK(h>0&&h<=16&&((q.size(1)==h&&q.size(2)==192)||(q.size(1)==1&&q.size(2)==h*192)),"SWA128 query must be [1,H,192] or [1,1,H*192]");
 TORCH_CHECK(k.dim()==3&&v.dim()==3&&k.size(1)==1&&k.size(2)==192&&v.size(1)==1&&v.size(2)==128&&k.size(0)==v.size(0)&&k.size(0)>0&&k.size(0)%128==0&&k.size(0)<=INT32_MAX/192,"SWA128 flat KV geometry invalid");
 TORCH_CHECK(p.dim()==1&&(window?p.numel()>0&&p.numel()<=INT32_MAX:p.numel()==2)&&st.sizes()==p.sizes()&&(pos.dim()==1||(window&&pos.dim()==2))&&pos.numel()==1&&sink.dim()==1&&sink.numel()==h,"SWA128 page/sink geometry invalid");
 TORCH_CHECK(std::isfinite(scale)&&std::isfinite(float(scale))&&float(scale)>0,"SWA128 scale invalid");return {{at::kBFloat16,{1,h,128}}};
}
Meta meta(const at::Stack&s){return metadata(s,false);}
Meta window_meta(const at::Stack&s){return metadata(s,true);}
std::shared_ptr<void> params(const at::Stack&s,size_t&size){size=4;return std::make_shared<float>(s[7].toDouble());}
Tensor run(Tensor q,Tensor k,Tensor v,Tensor p,Tensor st,Tensor pos,Tensor sink,double scale,bool fp32,bool window=false,bool fast=false){at::Stack s={q,k,v,p,st,pos,sink,scale};metadata(s,window);TORCH_CHECK(q.device().type()==at::kHPU,"HPU inputs required");auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(fast?(window?"gaudi_swa128::forward_window_fast_fp32":"gaudi_swa128::forward_fast_fp32"):window?"gaudi_swa128::forward_window_fp32":fp32?"gaudi_swa128::forward_fp32":"gaudi_swa128::forward");return d.execute(s)[0];}
Tensor fake_window(Tensor q,Tensor k,Tensor v,Tensor p,Tensor st,Tensor pos,Tensor sink,double scale){auto m=window_meta({q,k,v,p,st,pos,sink,scale});return at::empty(m[0].shape,q.options());}
Tensor fake(Tensor q,Tensor k,Tensor v,Tensor p,Tensor st,Tensor pos,Tensor sink,double scale){auto m=meta({q,k,v,p,st,pos,sink,scale});return at::empty(m[0].shape,q.options());}
}
TORCH_LIBRARY(gaudi_swa128,m){
 m.def("forward_fast_fp32(Tensor query, Tensor key, Tensor value, Tensor pages, Tensor starts, Tensor position, Tensor sinks, float scale) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_swa128::forward_fast_fp32","gk_swa128_two_page_head_fast_fp32_v0",meta,params);
 m.def("forward_window_fast_fp32(Tensor query, Tensor key, Tensor value, Tensor window_blocks, Tensor window_groups, Tensor input_positions, Tensor sinks, float scale) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_swa128::forward_window_fast_fp32","gk_swa128_window_head_fast_fp32_v0",window_meta,params);
m.def("forward_window_fp32(Tensor query, Tensor key, Tensor value, Tensor window_blocks, Tensor window_groups, Tensor input_positions, Tensor sinks, float scale) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_swa128::forward_window_fp32","gk_swa128_window_head_fp32_v0",window_meta,params);m.def("forward_fp32(Tensor query, Tensor key, Tensor value, Tensor pages, Tensor starts, Tensor position, Tensor sinks, float scale) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_swa128::forward_fp32","gk_swa128_two_page_head_fp32_v0",meta,params);m.def("forward(Tensor query, Tensor key, Tensor value, Tensor pages, Tensor starts, Tensor position, Tensor sinks, float scale) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_swa128::forward","gk_swa128_two_page_head_bf16_v0",meta,params);}
TORCH_LIBRARY_IMPL(gaudi_swa128,HPU,m){
 m.impl("forward_fast_fp32",[](Tensor q,Tensor k,Tensor v,Tensor p,Tensor st,Tensor pos,Tensor sink,double scale){return run(q,k,v,p,st,pos,sink,scale,true,false,true);});
 m.impl("forward_window_fast_fp32",[](Tensor q,Tensor k,Tensor v,Tensor p,Tensor st,Tensor pos,Tensor sink,double scale){return run(q,k,v,p,st,pos,sink,scale,true,true,true);});
m.impl("forward_window_fp32",[](Tensor q,Tensor k,Tensor v,Tensor p,Tensor st,Tensor pos,Tensor sink,double scale){return run(q,k,v,p,st,pos,sink,scale,true,true);});m.impl("forward",[](Tensor q,Tensor k,Tensor v,Tensor p,Tensor st,Tensor pos,Tensor sink,double scale){return run(q,k,v,p,st,pos,sink,scale,false);});m.impl("forward_fp32",[](Tensor q,Tensor k,Tensor v,Tensor p,Tensor st,Tensor pos,Tensor sink,double scale){return run(q,k,v,p,st,pos,sink,scale,true);});}
TORCH_LIBRARY_IMPL(gaudi_swa128,Meta,m){m.impl("forward_fast_fp32",fake);m.impl("forward_window_fast_fp32",fake_window);m.impl("forward_window_fp32",fake_window);m.impl("forward",fake);m.impl("forward_fp32",fake);}
