#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <tuple>
using T=at::Tensor;using V=std::vector<T>;using Meta=habana::PartialOutputMetaDataVector;
namespace {
bool i32(const T&t){
 if(!t.is_contiguous())return false;
 if(t.scalar_type()==at::kInt)return true;
 // The installed Lazy TopK callback may retain logical Long after the public
 // caller's explicit to(int32).clone(). In native-INT64-disabled mode the SDK
 // maps that metadata to physical I32. This is a backend-only compatibility:
 // run() below still rejects every public Long tensor, including Meta inputs.
 const char*flag=std::getenv("PT_ENABLE_INT64_SUPPORT");
 const bool logical=t.scalar_type()==at::kLong&&t.device().type()==at::kHPU
                    &&flag&&!std::strcmp(flag,"0");
 if(logical&&std::getenv("GK_ROUTE_METADATA_DIAGNOSTIC"))
  std::fprintf(stderr,"ROUTE_METADATA_LOGICAL_LONG physical=I32 flag=0 dims=%ld numel=%ld\n",long(t.dim()),long(t.numel()));
 return logical;
}
void devices(const at::Stack&s){auto d=s[0].toTensor().device();TORCH_CHECK(d.type()==at::kMeta||d.type()==at::kHPU,"HPU/Meta only");for(auto&v:s)if(v.isTensor())TORCH_CHECK(v.toTensor().device()==d,"same device required");}
void ids(const T&t,int64_t e){TORCH_CHECK(i32(t)&&t.dim()==2&&t.size(0)>=1&&t.size(0)<=513&&t.size(1)>=1&&t.size(1)<=8&&e>=t.size(1)&&e<=384,"bounded int32 IDs[T,R] required; dtype=",t.scalar_type()," sizes=",t.sizes()," strides=",t.strides()," contiguous=",t.is_contiguous()," device=",t.device()," experts=",e," PT_ENABLE_INT64_SUPPORT=",(std::getenv("PT_ENABLE_INT64_SUPPORT")?std::getenv("PT_ENABLE_INT64_SUPPORT"):"<unset>"));}
void capacity(int64_t t,int64_t r,int64_t c,int64_t b){TORCH_CHECK(c>=1&&c<=32&&c<=t&&b>=1&&b<=t*r,"bounded rows/capacity required");}
Meta flat_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor();ids(x,384);return{{at::kInt,{x.numel()}}};}
Meta count_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor(),f=s[1].toTensor();auto e=s[2].toInt();ids(x,e);TORCH_CHECK(i32(f)&&f.dim()==1&&f.numel()==x.numel(),"flat IDs shape");return{{at::kInt,{e}},{at::kInt,{x.size(0)}},{at::kInt,{e,(x.numel()+63)/64+1}}};}
Meta prefix_meta(const at::Stack&s){devices(s);auto n=s[0].toTensor(),f=s[1].toTensor();auto r=s[2].toInt(),c=s[3].toInt(),b=s[4].toInt();TORCH_CHECK(i32(n)&&i32(f)&&n.dim()==1&&f.dim()==1&&r>=1&&r<=8&&n.numel()>=r&&n.numel()<=384&&f.numel()>=1&&f.numel()<=513,"bounded counts/flags");capacity(f.numel(),r,c,b);return{{at::kInt,{n.numel()+1}},{at::kInt,{b}},{at::kInt,{b}},{at::kInt,{b}},{at::kInt,{1}}};}
Meta inverse_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor(),flat=s[1].toTensor(),p=s[2].toTensor(),st=s[3].toTensor(),off=s[4].toTensor();TORCH_CHECK(i32(p)&&p.dim()==1,"prefix shape");ids(x,p.numel()-1);TORCH_CHECK(i32(flat)&&flat.dim()==1&&flat.numel()==x.numel()&&i32(st)&&st.dim()==1&&st.numel()==1&&i32(off)&&off.dim()==2&&off.size(0)==p.numel()-1&&off.size(1)==(x.numel()+63)/64+1,"offset/status/flat shapes");capacity(x.size(0),x.size(1),s[5].toInt(),s[6].toInt());return{{at::kInt,{x.numel()}}};}
Meta map_meta(const at::Stack&s){devices(s);auto inv=s[0].toTensor(),v=s[1].toTensor(),st=s[2].toTensor();auto c=s[3].toInt();TORCH_CHECK(i32(inv)&&inv.dim()==1&&inv.numel()>=1&&inv.numel()<=513*8&&i32(v)&&v.dim()==1&&v.numel()>=1&&v.numel()<=inv.numel()&&i32(st)&&st.dim()==1&&st.numel()==1&&c>=1&&c<=32,"bounded inverse/valid/status/rows");return{{at::kInt,{v.numel()*c}}};}
std::shared_ptr<void> params(const at::Stack&s,size_t&n,int first,int count){n=count*4;auto p=std::shared_ptr<int[]>(new int[count]);for(int i=0;i<count;++i)p[i]=s[first+i].toInt();return std::shared_ptr<void>(p,p.get());}
using Fn=Meta(*)(const at::Stack&);
V run(const char*name,Fn fn,const at::Stack&s){
 for(const auto&value:s)if(value.isTensor()){
  const auto&t=value.toTensor();
  TORCH_CHECK(t.scalar_type()==at::kInt&&t.is_contiguous(),name,
              " public inputs require contiguous int32; explicit conversion by caller required, got ",t.scalar_type());
 }
 auto m=fn(s);if(s[0].toTensor().device().type()==at::kMeta){V v;for(auto&o:m)v.push_back(at::empty(o.shape,s[0].toTensor().options().dtype(o.dtype)));return v;}auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return op.execute(s);}
T flatten(T x){return run("gaudi_route_metadata_v3::flatten",flat_meta,{x})[0];}
auto count(T x,T f,int64_t e){auto v=run("gaudi_route_metadata_v3::count",count_meta,{x,f,e});return std::make_tuple(v[0],v[1],v[2]);}
auto prefix(T n,T f,int64_t r,int64_t c,int64_t b){auto v=run("gaudi_route_metadata_v3::prefix",prefix_meta,{n,f,r,c,b});return std::make_tuple(v[0],v[1],v[2],v[3],v[4]);}
T inverse(T x,T f,T p,T st,T off,int64_t c,int64_t b){return run("gaudi_route_metadata_v3::inverse",inverse_meta,{x,f,p,st,off,c,b})[0];}
T row_map(T inv,T v,T st,int64_t c){return run("gaudi_route_metadata_v3::row_map",map_meta,{inv,v,st,c})[0];}
}
TORCH_LIBRARY(gaudi_route_metadata_v3,m){
 m.def("flatten(Tensor ids) -> Tensor");m.def("count(Tensor ids, Tensor flat_ids, int experts) -> (Tensor, Tensor, Tensor)");m.def("prefix(Tensor counts, Tensor row_status, int routes, int rows, int capacity) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");m.def("inverse(Tensor ids, Tensor flat_ids, Tensor prefix, Tensor status, Tensor offsets, int rows, int capacity) -> Tensor");m.def("row_map(Tensor inverse, Tensor valid_rows, Tensor status, int rows) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_route_metadata_v3::flatten","reshape",flat_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
 habana::custom_op::registerUserCustomOp("gaudi_route_metadata_v3::count","gk_route_count_i32_v3",count_meta,[](const at::Stack&s,size_t&n){return params(s,n,2,1);});
 habana::custom_op::registerUserCustomOp("gaudi_route_metadata_v3::prefix","gk_route_prefix_i32_v3",prefix_meta,[](const at::Stack&s,size_t&n){return params(s,n,2,3);});
 habana::custom_op::registerUserCustomOp("gaudi_route_metadata_v3::inverse","gk_route_inverse_i32_v3",inverse_meta,[](const at::Stack&s,size_t&n){return params(s,n,5,2);});
 habana::custom_op::registerUserCustomOp("gaudi_route_metadata_v3::row_map","gk_route_row_map_i32_v3",map_meta,[](const at::Stack&s,size_t&n){return params(s,n,3,1);});
}
#define GK_V3_IMPL m.impl("flatten",flatten);m.impl("count",count);m.impl("prefix",prefix);m.impl("inverse",inverse);m.impl("row_map",row_map);
TORCH_LIBRARY_IMPL(gaudi_route_metadata_v3,HPU,m){GK_V3_IMPL}
TORCH_LIBRARY_IMPL(gaudi_route_metadata_v3,Meta,m){GK_V3_IMPL}
