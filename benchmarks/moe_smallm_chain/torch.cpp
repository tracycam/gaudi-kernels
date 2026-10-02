#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <tuple>
using T=at::Tensor;using V=std::vector<T>;using Meta=habana::PartialOutputMetaDataVector;
namespace {
#ifdef GK_SMALLM_DENSE
constexpr bool dense=true;
#else
constexpr bool dense=false;
#endif
void tensor(const T&t,at::ScalarType dtype,int rank){TORCH_CHECK(t.scalar_type()==dtype&&t.dim()==rank&&t.is_contiguous()&&!t.requires_grad(),"small-M tensor dtype/rank/contiguity");for(auto n:t.sizes())TORCH_CHECK(n>0,"nonempty small-M tensor required");}
void devices(const at::Stack&s){auto d=s[0].toTensor().device();TORCH_CHECK(d.type()==at::kMeta||d.type()==at::kHPU,"HPU or Meta only");for(auto&v:s)if(v.isTensor())TORCH_CHECK(v.toTensor().device()==d,"same device required");}
Meta flat_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor();tensor(x,at::kInt,2);TORCH_CHECK(x.size(1)==8&&x.size(0)<=8,"top8 T1..8");return{{at::kInt,{x.numel()}}};}
Meta metadata_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor();tensor(x,at::kInt,1);int64_t cap=s[1].toInt(),l=x.numel();TORCH_CHECK((cap==4||cap==8)&&l>=8&&l<=64&&l%8==0&&l/8<=cap,"metadata static capacity");return{{at::kInt,{l}},{at::kInt,{l}},{at::kInt,{l,cap}},{at::kInt,{l}}};}
Meta proj_meta(const at::Stack&s,int cap,bool down){
 devices(s);auto w=s[0].toTensor(),sc=s[1].toTensor(),lut=s[2].toTensor(),x=s[3].toTensor(),ids=s[4].toTensor(),counts=s[5].toTensor(),rows=s[6].toTensor(),inv=s[7].toTensor();
 tensor(w,at::kByte,2);tensor(sc,at::kByte,2);tensor(lut,at::kBFloat16,2);tensor(ids,at::kInt,1);tensor(counts,at::kInt,1);tensor(rows,at::kInt,2);tensor(inv,at::kInt,1);tensor(x,at::kBFloat16,down&&!dense?3:2);
 int64_t slots=ids.numel(),k=down?256:6144,n=down?6144:512;
 TORCH_CHECK(slots>=8&&slots<=64&&slots%8==0&&slots/8<=cap&&counts.numel()==slots&&rows.size(0)==slots&&rows.size(1)==cap&&inv.numel()==slots,"projection metadata geometry");
 TORCH_CHECK(w.size(0)==384*(n/512)*k&&w.size(1)==256&&sc.size(0)==w.size(0)/32&&sc.size(1)==512&&lut.size(0)==1&&lut.size(1)==512,"original HistoricalN512 TP8 owners only");
 if(down){TORCH_CHECK(x.size(0)==slots&&(dense?x.size(1)==256:(x.size(1)==cap&&x.size(2)==256)),"down compact activation");}
 else{TORCH_CHECK(x.size(0)==slots/8&&x.size(1)==6144,"GP original activation");}
 if(dense)return{{at::kFloat,{slots,down?1:3,n}}};
 return{{at::kFloat,{slots,down?1:3,cap,n}}};
}
Meta gate_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor(),counts=s[1].toTensor();tensor(x,at::kFloat,dense?3:4);tensor(counts,at::kInt,1);TORCH_CHECK(x.size(0)==counts.numel()&&x.size(1)==3&&(dense?x.size(2)==512:((x.size(2)==4||x.size(2)==8)&&x.size(3)==512)),"gate geometry");if(dense)return{{at::kBFloat16,{x.size(0),256}}};return{{at::kBFloat16,{x.size(0),x.size(2),256}}};}
Meta combine_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor(),r=s[1].toTensor(),inv=s[2].toTensor();tensor(x,at::kFloat,dense?3:4);tensor(r,at::kFloat,2);tensor(inv,at::kInt,1);TORCH_CHECK(r.size(1)==8&&x.size(0)==r.numel()&&x.size(1)==1&&(dense?x.size(2)==6144:((x.size(2)==4||x.size(2)==8)&&x.size(3)==6144))&&inv.numel()==r.numel(),"combine geometry");return{{at::kFloat,{r.size(0),6144}}};}
V execute(const char*name,const at::Stack&s,Meta m){if(s[0].toTensor().device().type()==at::kMeta){V out;for(auto&p:m)out.push_back(at::empty(p.shape,s[0].toTensor().options().dtype(p.dtype)));return out;}auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return op.execute(s);}
T flatten(T x){at::Stack s{x};return execute("gaudi_smallm::flatten",s,flat_meta(s))[0];}
auto metadata(T x,int64_t cap){at::Stack s{x,cap};auto v=execute("gaudi_smallm::metadata",s,metadata_meta(s));return std::make_tuple(v[0],v[1],v[2],v[3]);}
#define PROJ(NAME,CAP,DOWN) Meta NAME##_meta(const at::Stack&s){return proj_meta(s,CAP,DOWN);} T NAME(T w,T sc,T lut,T x,T ids,T counts,T rows,T inv){at::Stack s{w,sc,lut,x,ids,counts,rows,inv};return execute("gaudi_smallm::"#NAME,s,NAME##_meta(s))[0];}
PROJ(gp4,4,false) PROJ(down4,4,true) PROJ(gp8,8,false) PROJ(down8,8,true)
T gate(T x,T counts){at::Stack s{x,counts};return execute("gaudi_smallm::gate",s,gate_meta(s))[0];}
T combine(T x,T r,T inverse){at::Stack s{x,r,inverse};return execute("gaudi_smallm::combine",s,combine_meta(s))[0];}
}
TORCH_LIBRARY(gaudi_smallm,m){
 auto none=[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;};
 m.def("flatten(Tensor ids) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_smallm::flatten","reshape",flat_meta,none);
 m.def("metadata(Tensor ids, int cap) -> (Tensor, Tensor, Tensor, Tensor)");habana::custom_op::registerUserCustomOp("gaudi_smallm::metadata","gk_smallm_route_metadata",metadata_meta,[](const at::Stack&s,size_t&n)->std::shared_ptr<void>{n=4;return std::make_shared<int32_t>(s[1].toInt());});
#define REGISTER(N) m.def(#N "(Tensor w, Tensor scales, Tensor lut, Tensor x, Tensor ids, Tensor counts, Tensor rows, Tensor inverse) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_smallm::"#N,"gk_smallm_"#N,N##_meta,none);
 REGISTER(gp4) REGISTER(down4)
#ifdef GK_SMALLM_CAP8
 REGISTER(gp8) REGISTER(down8)
#endif
 m.def("gate(Tensor x, Tensor counts) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_smallm::gate","gk_smallm_gate",gate_meta,none);
 m.def("combine(Tensor x, Tensor routing, Tensor inverse) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_smallm::combine","gk_smallm_combine",combine_meta,none);
}
#ifdef GK_SMALLM_CAP8
#define GK_SMALLM_HIGH_IMPL m.impl("gp8",gp8);m.impl("down8",down8);
#else
#define GK_SMALLM_HIGH_IMPL
#endif
#define GK_SMALLM_IMPL m.impl("flatten",flatten);m.impl("metadata",metadata);m.impl("gp4",gp4);m.impl("down4",down4);GK_SMALLM_HIGH_IMPL m.impl("gate",gate);m.impl("combine",combine);
TORCH_LIBRARY_IMPL(gaudi_smallm,HPU,m){GK_SMALLM_IMPL}
TORCH_LIBRARY_IMPL(gaudi_smallm,Meta,m){GK_SMALLM_IMPL}
