// Public framework-current-graph registrations. No acquire/compile/replay API.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <synapse_common_types.hpp>
#include <climits>
using Tensor=at::Tensor;
using Meta=habana::PartialOutputMetaDataVector;
namespace {
void check(const Tensor&t,at::ScalarType type,int rank){
 TORCH_CHECK(t.scalar_type()==type&&t.dim()==rank&&t.is_contiguous()&&!t.requires_grad(),"block FP8 tensor dtype/rank/layout mismatch");
 for(auto n:t.sizes())TORCH_CHECK(n>0&&n<=INT32_MAX-128,"block FP8 unsupported empty/oversize shape");
}
void device(const Tensor&a,const Tensor&b){TORCH_CHECK(a.device()==b.device(),"block FP8 device mismatch");}
Meta quant_meta(const at::Stack&s){auto x=s[0].toTensor();check(x,at::kBFloat16,2);auto g=(x.size(1)+127)/128;
 return {{at::kFloat8_e4m3fn,{g,x.size(0),128}},{at::kFloat,{g,x.size(0),1}}};}
Meta batch_meta(const at::Stack&s){auto x=s[0].toTensor(),w=s[1].toTensor();check(x,at::kFloat8_e4m3fn,3);check(w,at::kFloat8_e4m3fn,3);device(x,w);
 TORCH_CHECK(x.size(0)==w.size(0)&&x.size(2)==128&&w.size(2)==128,"block FP8 batch geometry mismatch");
 return {{at::kFloat,{w.size(0),x.size(1),w.size(1)}}};}
Meta reduce_meta(const at::Stack&s){auto p=s[0].toTensor(),a=s[1].toTensor(),w=s[2].toTensor(),b=s[3].toTensor();
 check(p,at::kFloat,3);check(a,at::kFloat,3);check(w,at::kFloat,2);check(b,at::kFloat,1);device(p,a);device(p,w);device(p,b);
 TORCH_CHECK(a.size(0)==p.size(0)&&a.size(1)==p.size(1)&&a.size(2)==1&&w.size(0)==(p.size(2)+127)/128&&w.size(1)==p.size(0)&&b.size(0)==p.size(2),"block FP8 scale/bias geometry mismatch");
 return {{at::kBFloat16,{p.size(1),p.size(2)}}};}
Meta decode_meta(const at::Stack&s){auto w=s[0].toTensor(),sc=s[1].toTensor();auto k=s[2].toInt();check(w,at::kFloat8_e4m3fn,3);check(sc,at::kFloat,2);device(w,sc);
 TORCH_CHECK(k>0&&k<=INT32_MAX-128&&w.size(2)==128&&w.size(0)==(k+127)/128&&sc.size(0)==(w.size(1)+127)/128&&sc.size(1)==w.size(0),"block FP8 decode geometry mismatch");
 return {{at::kBFloat16,{w.size(1),k}}};}
Meta mm_meta(const at::Stack&s){auto x=s[0].toTensor(),w=s[1].toTensor();check(x,at::kBFloat16,2);check(w,at::kBFloat16,2);device(x,w);
 TORCH_CHECK(x.size(1)==w.size(1),"block FP8 BF16 MME K mismatch");return {{at::kFloat,{x.size(0),w.size(0)}}};}
Meta finish_meta(const at::Stack&s){auto p=s[0].toTensor(),b=s[1].toTensor();check(p,at::kFloat,2);check(b,at::kFloat,1);device(p,b);
 TORCH_CHECK(p.size(1)==b.size(0),"block FP8 finish bias geometry mismatch");return {{at::kBFloat16,p.sizes().vec()}};}
Meta split_meta(const at::Stack&s){auto x=s[0].toTensor();auto cut=s[1].toInt();check(x,at::kBFloat16,2);
 TORCH_CHECK(cut>0&&cut<x.size(0),"block FP8 row split out of range");
 return {{at::kBFloat16,{cut,x.size(1)}},{at::kBFloat16,{x.size(0)-cut,x.size(1)}}};}
std::shared_ptr<void> nop(const at::Stack&,size_t&n){n=0;return nullptr;}
std::shared_ptr<void> mmparams(const at::Stack&,size_t&n){n=sizeof(synGEMMParams);return std::make_shared<synGEMMParams>(false,true);}
std::vector<Tensor> execute(const char*name,const at::Stack&s){TORCH_CHECK(s[0].toTensor().device().type()==at::kHPU,"HPU required; no CPU fallback");
 auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return descriptor.execute(s);}
std::tuple<Tensor,Tensor> quant(Tensor x){quant_meta({x});auto v=execute("gaudi_block_fp8::quant",{x});return {v[0],v[1]};}
std::tuple<Tensor,Tensor> split(Tensor x,int64_t cut){split_meta({x,cut});auto v=execute("gaudi_block_fp8::split_rows",{x,cut});return {v[0],v[1]};}
Tensor batch(Tensor x,Tensor w){batch_meta({x,w});return execute("gaudi_block_fp8::batch_mm",{x,w})[0];}
Tensor reduce(Tensor p,Tensor a,Tensor w,Tensor b){reduce_meta({p,a,w,b});return execute("gaudi_block_fp8::reduce",{p,a,w,b})[0];}
Tensor decode(Tensor w,Tensor s,int64_t k){decode_meta({w,s,k});return execute("gaudi_block_fp8::decode",{w,s,k})[0];}
Tensor decode_fast(Tensor w,Tensor s,int64_t k){decode_meta({w,s,k});return execute("gaudi_block_fp8::decode_fast",{w,s,k})[0];}
Tensor mm(Tensor x,Tensor w){mm_meta({x,w});return execute("gaudi_block_fp8::mm",{x,w})[0];}
Tensor finish(Tensor p,Tensor b){finish_meta({p,b});return execute("gaudi_block_fp8::finish",{p,b})[0];}
Tensor fake(Tensor x,const Meta&v){return at::empty(v[0].shape,x.options().dtype(v[0].dtype));}
}
TORCH_LIBRARY(gaudi_block_fp8,m){using habana::custom_op::registerUserCustomOp;
 m.def("quant(Tensor x) -> (Tensor, Tensor)");m.def("batch_mm(Tensor x, Tensor w) -> Tensor");
 m.def("reduce(Tensor partial, Tensor a, Tensor w, Tensor bias) -> Tensor");
 m.def("decode(Tensor packed, Tensor scales, int k) -> Tensor");m.def("decode_fast(Tensor packed, Tensor scales, int k) -> Tensor");
 m.def("mm(Tensor x, Tensor w) -> Tensor");m.def("finish(Tensor partial, Tensor bias) -> Tensor");
 m.def("split_rows(Tensor x, int cut) -> (Tensor, Tensor)");
 registerUserCustomOp("gaudi_block_fp8::quant","gk_block128_quant_v1",quant_meta,nop);
 registerUserCustomOp("gaudi_block_fp8::batch_mm","batch_gemm",batch_meta,mmparams);
 registerUserCustomOp("gaudi_block_fp8::reduce","gk_block128_reduce_bias_v1",reduce_meta,nop);
 registerUserCustomOp("gaudi_block_fp8::decode","gk_block128_decode_v1",decode_meta,nop);
 registerUserCustomOp("gaudi_block_fp8::decode_fast","gk_block128_decode_fast_v1",decode_meta,nop);
 registerUserCustomOp("gaudi_block_fp8::mm","gemm",mm_meta,mmparams);
 registerUserCustomOp("gaudi_block_fp8::finish","gk_block128_finish_v1",finish_meta,nop);
 registerUserCustomOp("gaudi_block_fp8::split_rows","split",split_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=sizeof(synSplitParams);return std::make_shared<synSplitParams>(synSplitParams{1});});
}
TORCH_LIBRARY_IMPL(gaudi_block_fp8,HPU,m){m.impl("quant",quant);m.impl("batch_mm",batch);m.impl("reduce",reduce);m.impl("decode",decode);m.impl("decode_fast",decode_fast);m.impl("mm",mm);m.impl("finish",finish);m.impl("split_rows",split);}
TORCH_LIBRARY_IMPL(gaudi_block_fp8,Meta,m){
 m.impl("quant",[](Tensor x){auto v=quant_meta({x});return std::make_tuple(at::empty(v[0].shape,x.options().dtype(v[0].dtype)),at::empty(v[1].shape,x.options().dtype(v[1].dtype)));});
 m.impl("batch_mm",[](Tensor x,Tensor w){return fake(x,batch_meta({x,w}));});
 m.impl("reduce",[](Tensor p,Tensor a,Tensor w,Tensor b){return fake(p,reduce_meta({p,a,w,b}));});
 m.impl("decode",[](Tensor w,Tensor s,int64_t k){return fake(w,decode_meta({w,s,k}));});
 m.impl("decode_fast",[](Tensor w,Tensor s,int64_t k){return fake(w,decode_meta({w,s,k}));});
 m.impl("mm",[](Tensor x,Tensor w){return fake(x,mm_meta({x,w}));});
 m.impl("finish",[](Tensor p,Tensor b){return fake(p,finish_meta({p,b}));});
 m.impl("split_rows",[](Tensor x,int64_t cut){auto v=split_meta({x,cut});return std::make_tuple(at::empty(v[0].shape,x.options()),at::empty(v[1].shape,x.options()));});
}
