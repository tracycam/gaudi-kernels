// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cmath>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta meta(const at::Stack&s){
 auto q=s[0].toTensor(),k=s[1].toTensor(),v=s[2].toTensor(),p=s[3].toTensor(),g=s[4].toTensor(),pos=s[5].toTensor(),sink=s[6].toTensor();double scale=s[7].toDouble();
 for(auto t:{q,k,v,sink})TORCH_CHECK(t.scalar_type()==at::kBFloat16&&t.is_contiguous()&&!t.requires_grad(),"BF16 contiguous inference required");
 for(auto t:{p,g,pos})TORCH_CHECK((t.scalar_type()==at::kInt||t.scalar_type()==at::kLong)&&t.is_contiguous(),"I32 metadata required (Lazy native I32 only)");
 for(auto t:{k,v,p,g,pos,sink})TORCH_CHECK(t.device()==q.device(),"device mismatch");
 TORCH_CHECK(q.dim()==3&&q.size(0)>0&&q.size(0)<=32&&q.size(1)==1&&q.size(2)==3072,"query [R,1,3072], R1..32 required");
 TORCH_CHECK(k.dim()==3&&v.dim()==3&&k.size(1)==1&&k.size(2)==192&&v.size(1)==1&&v.size(2)==128&&k.size(0)==v.size(0)&&k.size(0)>0&&k.size(0)%128==0&&k.size(0)<=INT32_MAX/192,"invalid flat KV");
 TORCH_CHECK(p.dim()==1&&p.numel()>0&&p.numel()<=4096&&g.sizes()==p.sizes()&&(pos.dim()==1||(pos.dim()==2&&pos.size(1)==1))&&pos.numel()==q.size(0)&&sink.dim()==1&&sink.numel()==16,"invalid pages/positions/sink");
 TORCH_CHECK(std::isfinite(scale)&&std::isfinite(float(scale))&&float(scale)>0,"invalid scale");
 return {{at::kBFloat16,{q.size(0),16,128}}};
}
std::shared_ptr<void> params(const at::Stack&s,size_t&n){n=4;return std::make_shared<float>(s[7].toDouble());}
Tensor run(Tensor q,Tensor k,Tensor v,Tensor p,Tensor g,Tensor pos,Tensor sink,double scale){at::Stack s={q,k,v,p,g,pos,sink,scale};meta(s);TORCH_CHECK(q.device().type()==at::kHPU,"HPU required");auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_swa128_batch::window_fp32");return descriptor.execute(s)[0];}
Tensor run_quad(Tensor q,Tensor k,Tensor v,Tensor p,Tensor g,Tensor pos,Tensor sink,double scale){at::Stack s={q,k,v,p,g,pos,sink,scale};meta(s);TORCH_CHECK(q.device().type()==at::kHPU,"HPU required");auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_swa128_batch::window_quad_fp32");return descriptor.execute(s)[0];}
Tensor fake(Tensor q,Tensor k,Tensor v,Tensor p,Tensor g,Tensor pos,Tensor sink,double scale){auto m=meta({q,k,v,p,g,pos,sink,scale});return at::empty(m[0].shape,q.options());}
}
TORCH_LIBRARY(gaudi_swa128_batch,m){m.def("window_fp32(Tensor query, Tensor key, Tensor value, Tensor pages, Tensor groups, Tensor positions, Tensor sinks, float scale) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_swa128_batch::window_fp32","gk_swa128_batch_window_fp32_v1",meta,params);m.def("window_quad_fp32(Tensor query, Tensor key, Tensor value, Tensor pages, Tensor groups, Tensor positions, Tensor sinks, float scale) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_swa128_batch::window_quad_fp32","gk_swa128_batch_window_quad_fp32_v2",meta,params);}
TORCH_LIBRARY_IMPL(gaudi_swa128_batch,HPU,m){m.impl("window_fp32",run);m.impl("window_quad_fp32",run_quad);}
TORCH_LIBRARY_IMPL(gaudi_swa128_batch,Meta,m){m.impl("window_fp32",fake);m.impl("window_quad_fp32",fake);}
