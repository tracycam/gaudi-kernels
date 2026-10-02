// Framework-current-graph nodes through the installed public CustomOp API.
// No private OpBackend ABI, device acquire, graph compile or synchronization.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <synapse_common_types.hpp>
using Tensor=at::Tensor;
using Meta=habana::PartialOutputMetaDataVector;
using habana::custom_op::registerUserCustomOp;
using habana::custom_op::UserCustomOpDescriptor;
namespace {
void input(const Tensor& t, at::ScalarType dtype, int rank) {
  TORCH_CHECK(t.scalar_type()==dtype && t.dim()==rank && t.is_contiguous() && !t.requires_grad(),
              "gaudi_kernels requires contiguous inference tensors with the declared dtype/rank");
  for(auto n:t.sizes()) TORCH_CHECK(n>0 && n<=INT32_MAX,"unsupported empty/oversize tensor");
}
Meta mm_meta(const at::Stack& s, bool fp8, bool bf16_out) {
  auto x=s[0].toTensor(),w=s[1].toTensor();
  auto dtype=fp8?at::kFloat8_e4m3fn:at::kBFloat16;
  input(x,dtype,2);input(w,dtype,2);
  TORCH_CHECK(x.device()==w.device() && x.size(1)==w.size(1),"linear device/K mismatch");
  return {{bf16_out?at::kBFloat16:at::kFloat,{x.size(0),w.size(0)}}};
}
std::shared_ptr<void> mm_params(const at::Stack&,size_t& bytes) {
  bytes=sizeof(synGEMMParams);return std::make_shared<synGEMMParams>(false,true);
}
std::shared_ptr<void> no_params(const at::Stack&,size_t& bytes){bytes=0;return nullptr;}
std::vector<Tensor> execute(const char* name,const at::Stack& s) {
  TORCH_CHECK(s[0].toTensor().device().type()==at::kHPU,"HPU tensors required; no CPU fallback");
  auto descriptor=UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
  return descriptor.execute(s);
}
Tensor mm_bf16(Tensor x,Tensor w){mm_meta({x,w},false,true);return execute("gaudi_kernels::_bf16_mm",{x,w})[0];}
Tensor mm_f32(Tensor x,Tensor w){mm_meta({x,w},false,false);return execute("gaudi_kernels::_bf16_mm_f32",{x,w})[0];}
Tensor mm_fp8(Tensor x,Tensor w){mm_meta({x,w},true,false);return execute("gaudi_kernels::_fp8_mm_f32",{x,w})[0];}
Meta bias_meta(const at::Stack& s) {
  auto p=s[0].toTensor(),b=s[1].toTensor();input(p,at::kFloat,2);input(b,at::kFloat,1);
  TORCH_CHECK(p.device()==b.device() && p.size(1)==b.size(0),"bias shape/device mismatch");
  return {{at::kBFloat16,p.sizes().vec()}};
}
Tensor bias(Tensor p,Tensor b){bias_meta({p,b});return execute("gaudi_kernels::_bf16_bias",{p,b})[0];}
Meta quant_meta(const at::Stack& s) {
  auto x=s[0].toTensor();input(x,at::kBFloat16,2);
  return {{at::kFloat8_e4m3fn,x.sizes().vec()},{at::kFloat,{x.size(0),1}}};
}
std::tuple<Tensor,Tensor> quant(Tensor x) {
  quant_meta({x});auto v=execute("gaudi_kernels::_fp8_quant",{x});return {v[0],v[1]};
}
std::tuple<Tensor,Tensor> quant_fast(Tensor x) {
  quant_meta({x});auto v=execute("gaudi_kernels::_fp8_quant_fast",{x});return {v[0],v[1]};
}
Meta quant_table_meta(const at::Stack& s) {
  auto x=s[0].toTensor(),table=s[1].toTensor();
  auto result=quant_meta({x});input(table,at::kFloat,2);
  TORCH_CHECK(table.size(0)==1&&table.size(1)==129&&table.device()==x.device(),"FP8 reciprocal table must be FP32 [1,129] on the activation device");
  TORCH_CHECK(x.size(1)<=INT32_MAX-511,"FP8 LUT quantizer K exceeds static index range");
  return result;
}
std::tuple<Tensor,Tensor> quant_lut1(Tensor x,Tensor t) {
  quant_table_meta({x,t});auto v=execute("gaudi_kernels::_fp8_quant_lut1",{x,t});return {v[0],v[1]};
}
std::tuple<Tensor,Tensor> quant_lut4(Tensor x,Tensor t) {
  quant_table_meta({x,t});auto v=execute("gaudi_kernels::_fp8_quant_lut4",{x,t});return {v[0],v[1]};
}
Meta epilogue_meta(const at::Stack& s) {
  auto p=s[0].toTensor(),a=s[1].toTensor(),w=s[2].toTensor(),b=s[3].toTensor();
  input(p,at::kFloat,2);input(a,at::kFloat,2);input(w,at::kFloat,1);input(b,at::kFloat,1);
  TORCH_CHECK(a.size(0)==p.size(0)&&a.size(1)==1&&w.size(0)==p.size(1)&&b.size(0)==p.size(1),"FP8 scale/bias shapes");
  TORCH_CHECK(p.device()==a.device()&&p.device()==w.device()&&p.device()==b.device(),"FP8 device mismatch");
  return {{at::kBFloat16,p.sizes().vec()}};
}
Tensor epilogue(Tensor p,Tensor a,Tensor w,Tensor b){epilogue_meta({p,a,w,b});return execute("gaudi_kernels::_fp8_epilogue",{p,a,w,b})[0];}
Tensor epilogue_rows(Tensor p,Tensor a,Tensor w,Tensor b){epilogue_meta({p,a,w,b});return execute("gaudi_kernels::_fp8_epilogue_rows",{p,a,w,b})[0];}
Meta decode_meta(const at::Stack& s){auto w=s[0].toTensor();input(w,at::kFloat8_e4m3fn,2);return {{at::kBFloat16,w.sizes().vec()}};}
Tensor decode(Tensor w){decode_meta({w});return execute("gaudi_kernels::_fp8_decode",{w})[0];}
Meta epilogue16_meta(const at::Stack& s){
  auto p=s[0].toTensor(),w=s[1].toTensor(),b=s[2].toTensor();
  bias_meta({p,b});input(w,at::kFloat,1);
  TORCH_CHECK(w.device()==p.device()&&w.size(0)==p.size(1),"FP8 channel scale shape/device mismatch");
  return {{at::kBFloat16,p.sizes().vec()}};
}
Tensor epilogue16(Tensor p,Tensor w,Tensor b){epilogue16_meta({p,w,b});return execute("gaudi_kernels::_fp8_epilogue16",{p,w,b})[0];}
Tensor epilogue16_rows(Tensor p,Tensor w,Tensor b){epilogue16_meta({p,w,b});return execute("gaudi_kernels::_fp8_epilogue16_rows",{p,w,b})[0];}
Tensor fake(const Tensor& x,const Meta& meta){return at::empty(meta[0].shape,x.options().dtype(meta[0].dtype));}
}
TORCH_LIBRARY(gaudi_kernels,m) {
  m.def("_bf16_mm(Tensor x, Tensor w) -> Tensor");
  m.def("_bf16_mm_f32(Tensor x, Tensor w) -> Tensor");
  m.def("_fp8_mm_f32(Tensor x, Tensor w) -> Tensor");
  m.def("_bf16_bias(Tensor partial, Tensor bias) -> Tensor");
  m.def("_fp8_quant(Tensor x) -> (Tensor, Tensor)");
  m.def("_fp8_epilogue(Tensor p, Tensor a, Tensor w, Tensor bias) -> Tensor");
  m.def("_fp8_decode(Tensor native_weight) -> Tensor");
  m.def("_fp8_epilogue16(Tensor p, Tensor w, Tensor bias) -> Tensor");
  m.def("_fp8_quant_fast(Tensor x) -> (Tensor, Tensor)");
  m.def("_fp8_quant_lut1(Tensor x, Tensor table) -> (Tensor, Tensor)");
  m.def("_fp8_quant_lut4(Tensor x, Tensor table) -> (Tensor, Tensor)");
  m.def("_fp8_epilogue_rows(Tensor p, Tensor a, Tensor w, Tensor bias) -> Tensor");
  m.def("_fp8_epilogue16_rows(Tensor p, Tensor w, Tensor bias) -> Tensor");
  registerUserCustomOp("gaudi_kernels::_bf16_mm","gemm",[](const at::Stack&s){return mm_meta(s,false,true);},mm_params);
  registerUserCustomOp("gaudi_kernels::_bf16_mm_f32","gemm",[](const at::Stack&s){return mm_meta(s,false,false);},mm_params);
  registerUserCustomOp("gaudi_kernels::_fp8_mm_f32","gemm",[](const at::Stack&s){return mm_meta(s,true,false);},mm_params);
  registerUserCustomOp("gaudi_kernels::_bf16_bias","gk_bf16_bias_bf16",bias_meta,no_params);
  registerUserCustomOp("gaudi_kernels::_fp8_quant","fp8_linear_activation",quant_meta,no_params);
  registerUserCustomOp("gaudi_kernels::_fp8_epilogue","fp8_linear_epilogue8",epilogue_meta,no_params);
  registerUserCustomOp("gaudi_kernels::_fp8_decode","fp8_linear_decode",decode_meta,no_params);
  registerUserCustomOp("gaudi_kernels::_fp8_epilogue16","fp8_linear_epilogue16",epilogue16_meta,no_params);
  registerUserCustomOp("gaudi_kernels::_fp8_quant_fast","fp8_linear_activation_fast",quant_meta,no_params);
  registerUserCustomOp("gaudi_kernels::_fp8_quant_lut1","fp8_fp_quant_lut_single",quant_table_meta,no_params);
  registerUserCustomOp("gaudi_kernels::_fp8_quant_lut4","fp8_fp_quant_lut_single4",quant_table_meta,no_params);
  registerUserCustomOp("gaudi_kernels::_fp8_epilogue_rows","fp8_linear_epilogue8_rows",epilogue_meta,no_params);
  registerUserCustomOp("gaudi_kernels::_fp8_epilogue16_rows","fp8_linear_epilogue16_rows",epilogue16_meta,no_params);
}
TORCH_LIBRARY_IMPL(gaudi_kernels,HPU,m) {
  m.impl("_bf16_mm",mm_bf16);m.impl("_bf16_mm_f32",mm_f32);m.impl("_fp8_mm_f32",mm_fp8);
  m.impl("_bf16_bias",bias);m.impl("_fp8_quant",quant);m.impl("_fp8_epilogue",epilogue);
  m.impl("_fp8_decode",decode);m.impl("_fp8_epilogue16",epilogue16);
  m.impl("_fp8_quant_fast",quant_fast);m.impl("_fp8_epilogue_rows",epilogue_rows);m.impl("_fp8_epilogue16_rows",epilogue16_rows);
  m.impl("_fp8_quant_lut1",quant_lut1);m.impl("_fp8_quant_lut4",quant_lut4);
}
TORCH_LIBRARY_IMPL(gaudi_kernels,Meta,m) {
  m.impl("_bf16_mm",[](Tensor x,Tensor w){return fake(x,mm_meta({x,w},false,true));});
  m.impl("_bf16_mm_f32",[](Tensor x,Tensor w){return fake(x,mm_meta({x,w},false,false));});
  m.impl("_fp8_mm_f32",[](Tensor x,Tensor w){return fake(x,mm_meta({x,w},true,false));});
  m.impl("_bf16_bias",[](Tensor p,Tensor b){return fake(p,bias_meta({p,b}));});
  m.impl("_fp8_quant",[](Tensor x){auto v=quant_meta({x});return std::make_tuple(at::empty(v[0].shape,x.options().dtype(v[0].dtype)),at::empty(v[1].shape,x.options().dtype(v[1].dtype)));});
  m.impl("_fp8_epilogue",[](Tensor p,Tensor a,Tensor w,Tensor b){return fake(p,epilogue_meta({p,a,w,b}));});
  m.impl("_fp8_decode",[](Tensor w){return fake(w,decode_meta({w}));});
  m.impl("_fp8_epilogue16",[](Tensor p,Tensor w,Tensor b){return fake(p,epilogue16_meta({p,w,b}));});
  m.impl("_fp8_quant_fast",[](Tensor x){auto v=quant_meta({x});return std::make_tuple(at::empty(v[0].shape,x.options().dtype(v[0].dtype)),at::empty(v[1].shape,x.options().dtype(v[1].dtype)));});
  m.impl("_fp8_quant_lut1",[](Tensor x,Tensor t){auto v=quant_table_meta({x,t});return std::make_tuple(at::empty(v[0].shape,x.options().dtype(v[0].dtype)),at::empty(v[1].shape,x.options().dtype(v[1].dtype)));});
  m.impl("_fp8_quant_lut4",[](Tensor x,Tensor t){auto v=quant_table_meta({x,t});return std::make_tuple(at::empty(v[0].shape,x.options().dtype(v[0].dtype)),at::empty(v[1].shape,x.options().dtype(v[1].dtype)));});
  m.impl("_fp8_epilogue_rows",[](Tensor p,Tensor a,Tensor w,Tensor b){return fake(p,epilogue_meta({p,a,w,b}));});
  m.impl("_fp8_epilogue16_rows",[](Tensor p,Tensor w,Tensor b){return fake(p,epilogue16_meta({p,w,b}));});
}
