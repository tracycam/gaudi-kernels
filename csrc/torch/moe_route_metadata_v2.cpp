#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <cstring>
#include <tuple>
using T=at::Tensor;using V=std::vector<T>;
namespace {
bool i32(const T&t){return t.scalar_type()==at::kInt&&t.is_contiguous();}
void ids(const T&t,int64_t e){TORCH_CHECK(i32(t)&&t.dim()==2&&t.size(0)>=1&&t.size(0)<=513&&t.size(1)>=1&&t.size(1)<=8&&e>=t.size(1)&&e<=384,"bounded route IDs[T,R], T<=513 R<=8 E<=384 required");}
void capacity(int64_t t,int64_t r,int64_t c,int64_t b){TORCH_CHECK(c>=1&&c<=32&&c<=t&&b>=1&&b<=t*r,"bounded C/B required");}
habana::PartialOutputMetaDataVector meta0(const at::Stack&s){auto x=s[0].toTensor();int64_t e=s[2].toInt();ids(x,e);auto flat=s[1].toTensor();TORCH_CHECK(i32(flat)&&flat.dim()==1&&flat.numel()==x.numel(),"graph-owned flat IDs alias required");return{{at::kInt,{e}},{at::kInt,{x.size(0)}},{at::kInt,{e,(x.numel()+63)/64}}};}
habana::PartialOutputMetaDataVector meta1(const at::Stack&s){auto n=s[0].toTensor(),f=s[1].toTensor();int64_t r=s[2].toInt(),c=s[3].toInt(),b=s[4].toInt();TORCH_CHECK(i32(n)&&i32(f)&&n.dim()==1&&f.dim()==1&&n.numel()>=r&&n.numel()<=384&&f.numel()>=1&&f.numel()<=513&&r>=1&&r<=8,"bounded count/status required");capacity(f.numel(),r,c,b);return{{at::kInt,{n.numel()+1}},{at::kInt,{b}},{at::kInt,{b}},{at::kInt,{b}},{at::kInt,{1}}};}
habana::PartialOutputMetaDataVector meta2(const at::Stack&s){auto x=s[0].toTensor(),p=s[1].toTensor(),f=s[2].toTensor();TORCH_CHECK(i32(p)&&i32(f)&&p.dim()==1&&f.dim()==1&&f.numel()==1,"prefix/status vectors required");ids(x,p.numel()-1);auto chunks=s[3].toTensor();TORCH_CHECK(i32(chunks)&&chunks.dim()==2&&chunks.size(0)==p.numel()-1&&chunks.size(1)==(x.numel()+63)/64,"chunk counts[E,ceil(TR/64)] required");int64_t c=s[4].toInt(),b=s[5].toInt();capacity(x.size(0),x.size(1),c,b);return{{at::kInt,{b*c}},{at::kInt,{x.numel()}}};}
std::shared_ptr<void> params(const at::Stack&s,size_t&n,int first,int count){n=count*sizeof(int);auto p=std::shared_ptr<int[]>(new int[count]);for(int i=0;i<count;++i)p[i]=s[first+i].toInt();return std::shared_ptr<void>(p,p.get());}
V execute(const char*name,at::Stack s){auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return op.execute(s);}
habana::PartialOutputMetaDataVector meta_flat(const at::Stack&s){auto x=s[0].toTensor();ids(x,384);return{{at::kInt,{x.numel()}}};}
auto triple(V v){return std::make_tuple(v[0],v[1],v[2]);}
auto pair(V v){return std::make_tuple(v[0],v[1]);}
auto five(V v){return std::make_tuple(v[0],v[1],v[2],v[3],v[4]);}
V fake(const at::Stack&s,const habana::PartialOutputMetaDataVector&m){V v;for(auto&o:m)v.push_back(at::empty(o.shape,s[0].toTensor().options().dtype(at::kInt)));return v;}
}
TORCH_LIBRARY(gaudi_route_metadata_v2,m){
 m.def("flatten(Tensor ids) -> Tensor");habana::custom_op::registerUserCustomOp("gaudi_route_metadata_v2::flatten","reshape",meta_flat,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
 m.def("count(Tensor ids, Tensor flat_ids, int experts) -> (Tensor, Tensor, Tensor)");m.def("prefix(Tensor counts, Tensor row_status, int routes, int rows, int capacity) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");m.def("scatter(Tensor ids, Tensor prefix, Tensor status, Tensor chunk_counts, int rows, int capacity) -> (Tensor, Tensor)");
 habana::custom_op::registerUserCustomOp("gaudi_route_metadata_v2::count","gk_route_count_i32_v2",meta0,[](const at::Stack&s,size_t&n){return params(s,n,2,1);});
 habana::custom_op::registerUserCustomOp("gaudi_route_metadata_v2::prefix","gk_route_prefix_i32_v2",meta1,[](const at::Stack&s,size_t&n){return params(s,n,2,3);});
 habana::custom_op::registerUserCustomOp("gaudi_route_metadata_v2::scatter","gk_route_scatter_i32_v2",meta2,[](const at::Stack&s,size_t&n){return params(s,n,4,2);});
}
TORCH_LIBRARY_IMPL(gaudi_route_metadata_v2,HPU,m){
 m.impl("flatten",[](T x){at::Stack s{x};meta_flat(s);return execute("gaudi_route_metadata_v2::flatten",s)[0];});
 m.impl("count",[](T x,T flat,int64_t e){at::Stack s{x,flat,e};meta0(s);return triple(execute("gaudi_route_metadata_v2::count",s));});
 m.impl("prefix",[](T n,T f,int64_t r,int64_t c,int64_t b){at::Stack s{n,f,r,c,b};meta1(s);return five(execute("gaudi_route_metadata_v2::prefix",s));});
 m.impl("scatter",[](T x,T p,T f,T chunks,int64_t c,int64_t b){at::Stack s{x,p,f,chunks,c,b};meta2(s);return pair(execute("gaudi_route_metadata_v2::scatter",s));});
}
TORCH_LIBRARY_IMPL(gaudi_route_metadata_v2,Meta,m){
 m.impl("flatten",[](T x){at::Stack s{x};return fake(s,meta_flat(s))[0];});
 m.impl("count",[](T x,T flat,int64_t e){at::Stack s{x,flat,e};return triple(fake(s,meta0(s)));});
 m.impl("prefix",[](T n,T f,int64_t r,int64_t c,int64_t b){at::Stack s{n,f,r,c,b};return five(fake(s,meta1(s)));});
 m.impl("scatter",[](T x,T p,T f,T chunks,int64_t c,int64_t b){at::Stack s{x,p,f,chunks,c,b};return pair(fake(s,meta2(s)));});
}
