// PRIVATE wide-row experiment: copied from csrc/torch/moe_route_tiles.cpp at 822e049.
// Only namespace/GUID and explicit row limit change; original interface is untouched.
// Explicit current-graph consumers for device route-tile metadata.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <climits>
using T=at::Tensor;using V=std::vector<T>;using Meta=habana::PartialOutputMetaDataVector;
namespace {
void check(const T&t,at::ScalarType type,int rank){TORCH_CHECK(t.scalar_type()==type&&t.dim()==rank&&t.is_contiguous()&&!t.requires_grad()&&t.numel()>0&&t.numel()<=INT32_MAX,"route-tile dtype/rank/contiguity contract");for(auto n:t.sizes())TORCH_CHECK(n>0,"empty route-tile dimension");}
void devices(const at::Stack&s){auto d=s[0].toTensor().device();TORCH_CHECK(d.type()==at::kHPU||d.type()==at::kMeta,"HPU or Meta only");for(const auto&v:s)if(v.isTensor())TORCH_CHECK(v.toTensor().device()==d,"same device required");}
void status(const T&s){check(s,at::kInt,1);TORCH_CHECK(s.numel()==1,"global status[1] required");}
Meta gather_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor(),map=s[1].toTensor();check(x,at::kBFloat16,2);check(map,at::kInt,1);status(s[2].toTensor());auto r=s[3].toInt(),c=s[4].toInt(),slot=s[5].toInt(),batch=s[6].toInt();TORCH_CHECK(x.size(0)>=1&&x.size(0)<=513&&x.size(1)==6144&&r>=1&&r<=8&&c>=1&&c<=128&&c<=x.size(0)&&slot>=0&&slot<=513*8&&batch>=1&&batch<=513*8&&map.numel()%c==0&&slot+batch<=map.numel()/c&&batch*c*6144<=INT32_MAX,"bounded route gather geometry");return{{at::kBFloat16,{batch,c,6144}}};}
Meta gate_meta(const at::Stack&s){devices(s);auto p=s[0].toTensor(),v=s[1].toTensor();check(p,at::kFloat,3);check(v,at::kInt,1);status(s[2].toTensor());auto slot=s[3].toInt();TORCH_CHECK(p.size(2)==512&&p.size(1)<=128&&p.size(0)<=513*8&&slot>=0&&slot<=513*8&&slot+p.size(0)<=v.numel(),"bounded literal gate geometry");return{{at::kBFloat16,{p.size(0),p.size(1),256}}};}
Meta combine_meta(const at::Stack&s){devices(s);auto p=s[0].toTensor(),r=s[1].toTensor(),inv=s[2].toTensor();check(p,at::kFloat,2);check(r,at::kFloat,2);check(inv,at::kInt,1);status(s[3].toTensor());TORCH_CHECK(p.size(0)<=513*8*128&&p.size(1)>=512&&p.size(1)<=6144&&p.size(1)%512==0&&r.size(0)<=513&&r.size(1)<=8&&inv.numel()==r.numel(),"ordered combine geometry");return{{at::kFloat,{r.size(0),p.size(1)}}};}
Meta reshape_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor();check(x,at::kInt,1);auto shape=s[1].toIntVector();int64_t n=1;TORCH_CHECK(shape.size()>=1&&shape.size()<=2,"int32 reshape rank1..2");for(auto d:shape){TORCH_CHECK(d>0&&d<=INT32_MAX&&n<=INT32_MAX/d,"int32 reshape overflow");n*=d;}TORCH_CHECK(n==x.numel(),"int32 reshape size mismatch");return{{at::kInt,shape}};}
using Fn=Meta(*)(const at::Stack&);
T run(const char*name,Fn fn,const at::Stack&s){auto m=fn(s);if(s[0].toTensor().device().type()==at::kMeta)return at::empty(m[0].shape,s[0].toTensor().options().dtype(m[0].dtype));auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return descriptor.execute(s)[0];}
std::shared_ptr<void> params(const at::Stack&s,size_t&n,int first,int count){n=count*4;auto p=std::shared_ptr<int[]>(new int[count]);for(int i=0;i<count;++i)p[i]=s[first+i].toInt();return std::shared_ptr<void>(p,p.get());}
T gather(T x,T m,T st,int64_t r,int64_t c,int64_t slot,int64_t batch){return run("gaudi_route_wide_tiles::gather",gather_meta,{x,m,st,r,c,slot,batch});}
T gate(T p,T v,T st,int64_t slot){return run("gaudi_route_wide_tiles::gate",gate_meta,{p,v,st,slot});}
T combine(T p,T r,T i,T st){return run("gaudi_route_wide_tiles::combine",combine_meta,{p,r,i,st});}
T reshape(T x,std::vector<int64_t>shape){return run("gaudi_route_wide_tiles::reshape_i32",reshape_meta,{x,shape});}
std::shared_ptr<void> nop(const at::Stack&,size_t&n){n=0;return nullptr;}
}
TORCH_LIBRARY(gaudi_route_wide_tiles,m){
 m.def("gather(Tensor input, Tensor row_map, Tensor status, int routes, int capacity, int slot_begin, int tile_batch) -> Tensor");
 m.def("gate(Tensor partial, Tensor valid_rows, Tensor status, int slot_begin) -> Tensor");
 m.def("combine(Tensor partial, Tensor routing, Tensor inverse, Tensor status) -> Tensor");
 m.def("reshape_i32(Tensor input, int[] shape) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_route_wide_tiles::gather","gk_route_wide_tile_gather_bf16_v1",gather_meta,[](const at::Stack&s,size_t&n){return params(s,n,3,3);});
 habana::custom_op::registerUserCustomOp("gaudi_route_wide_tiles::gate","gk_route_wide_tile_gate_bf16_v1",gate_meta,[](const at::Stack&s,size_t&n){return params(s,n,3,1);});
 habana::custom_op::registerUserCustomOp("gaudi_route_wide_tiles::combine","gk_route_wide_tile_combine_f32_v1",combine_meta,nop);
 habana::custom_op::registerUserCustomOp("gaudi_route_wide_tiles::reshape_i32","reshape",reshape_meta,nop);
}
#define GK_ROUTE_TILE_IMPL m.impl("gather",gather);m.impl("gate",gate);m.impl("combine",combine);m.impl("reshape_i32",reshape);
TORCH_LIBRARY_IMPL(gaudi_route_wide_tiles,HPU,m){GK_ROUTE_TILE_IMPL}
TORCH_LIBRARY_IMPL(gaudi_route_wide_tiles,Meta,m){GK_ROUTE_TILE_IMPL}
