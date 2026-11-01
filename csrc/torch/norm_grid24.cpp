// Public graph-owned three-output node. No device acquisition or alias promise.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cmath>
using Tensor=at::Tensor;using Meta=habana::PartialOutputMetaDataVector;
namespace {
Meta meta(const at::Stack&s){
 auto x=s[0].toTensor(),r=s[1].toTensor(),g=s[2].toTensor();double e=s[3].toDouble();
 for(auto t:{x,r,g})TORCH_CHECK(t.scalar_type()==at::kBFloat16&&t.is_contiguous()&&!t.requires_grad(),"grid24 needs contiguous inference BF16 inputs");
 auto m=x.dim()==2?x.size(0):0,h=x.dim()==2?x.size(1):0,kb=h/128;
 TORCH_CHECK(x.dim()==2&&m>=1&&m<=16384&&h>=256&&h<=8192&&h%256==0&&r.sizes()==x.sizes()&&g.dim()==1&&g.size(0)==h,"grid24 requires x/residual [M,H], gamma [H] with 1<=M<=16384, 256<=H<=8192, H%256==0");
 TORCH_CHECK(x.device()==r.device()&&x.device()==g.device(),"grid24 devices differ");
 TORCH_CHECK(std::isfinite(e)&&std::isfinite(float(e))&&float(e)>0,"epsilon must become positive finite FP32");
 return {{at::kBFloat16,{m,h}},{at::kFloat8_e4m3fn,{kb,m,128}},{at::kFloat,{kb,m,1}}};
}
std::tuple<Tensor,Tensor,Tensor>run(Tensor x,Tensor r,Tensor g,double e){at::Stack s{x,r,g,e};meta(s);TORCH_CHECK(x.device().type()==at::kHPU,"HPU required");auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("gaudi_norm_grid24::block128");auto o=d.execute(s);return{o[0],o[1],o[2]};}
std::tuple<Tensor,Tensor,Tensor>fake(Tensor x,Tensor r,Tensor g,double e){auto m=meta({x,r,g,e});return{at::empty(m[0].shape,x.options().dtype(m[0].dtype)),at::empty(m[1].shape,x.options().dtype(m[1].dtype)),at::empty(m[2].shape,x.options().dtype(m[2].dtype))};}
}
TORCH_LIBRARY(gaudi_norm_grid24,m){m.def("block128(Tensor x, Tensor residual, Tensor gamma, float epsilon) -> (Tensor, Tensor, Tensor)");habana::custom_op::registerUserCustomOp("gaudi_norm_grid24::block128","gk_norm_block128_grid24_v1",meta,[](const at::Stack&s,size_t&n)->std::shared_ptr<void>{n=4;return std::make_shared<float>(s[3].toDouble());});}
TORCH_LIBRARY_IMPL(gaudi_norm_grid24,HPU,m){m.impl("block128",run);}
TORCH_LIBRARY_IMPL(gaudi_norm_grid24,Meta,m){m.impl("block128",fake);}
