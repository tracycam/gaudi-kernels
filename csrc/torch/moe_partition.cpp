#include <ATen/ATen.h>
#include <torch/library.h>
#include <hpu_custom_op_pt2.h>
#include <climits>
using T=at::Tensor;using V=std::vector<T>;using Meta=habana::PartialOutputMetaDataVector;
namespace {
void tensor(const T&t,at::ScalarType d,int rank){TORCH_CHECK(t.scalar_type()==d&&t.dim()==rank&&t.is_contiguous()&&!t.requires_grad()&&t.numel()>0&&t.numel()<=INT32_MAX,"expert partition tensor contract");}
void devices(const at::Stack&s){auto d=s[0].toTensor().device();TORCH_CHECK(d.type()==at::kHPU||d.type()==at::kMeta,"HPU/Meta required");for(auto&v:s)if(v.isTensor())TORCH_CHECK(v.toTensor().device()==d,"same device required");}
void status(const T&t){tensor(t,at::kInt,1);TORCH_CHECK(t.numel()==1,"status[1]");}
Meta prefix_meta(const at::Stack&s){devices(s);auto n=s[0].toTensor(),f=s[1].toTensor();tensor(n,at::kInt,1);tensor(f,at::kInt,1);
 auto r=s[2].toInt(),c=s[3].toInt(),b=s[4].toInt(),lo=s[5].toInt(),hi=s[6].toInt(),overflow=s[7].toInt();
 TORCH_CHECK(r>=1&&r<=8&&n.numel()>=r&&n.numel()<=384&&f.numel()<=513&&c>=1&&c<=f.numel()&&b>=1&&b<=n.numel()&&lo>=1&&lo<=hi&&hi<=c&&overflow>=0&&overflow<=1,"bounded expert bucket");
 return{{at::kInt,{n.numel()+1}},{at::kInt,{b}},{at::kInt,{b}},{at::kInt,{b}},{at::kInt,{1}}};}
Meta inverse_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor(),f=s[1].toTensor(),p=s[2].toTensor(),off=s[4].toTensor();tensor(x,at::kInt,2);tensor(f,at::kInt,1);tensor(p,at::kInt,1);tensor(off,at::kInt,2);status(s[3].toTensor());auto c=s[5].toInt(),b=s[6].toInt();
 TORCH_CHECK(x.size(0)<=513&&x.size(1)<=8&&p.numel()>=x.size(1)+1&&p.numel()<=385&&f.numel()==x.numel()&&off.size(0)==p.numel()-1&&off.size(1)==(x.numel()+63)/64+1&&c>=1&&c<=x.size(0)&&b>=1&&b<p.numel(),"bucket inverse geometry");return{{at::kInt,{x.numel()}}};}
Meta map_meta(const at::Stack&s){devices(s);auto inv=s[0].toTensor(),v=s[1].toTensor();tensor(inv,at::kInt,1);tensor(v,at::kInt,1);status(s[2].toTensor());auto c=s[3].toInt();TORCH_CHECK(inv.numel()<=513*8&&v.numel()<=384&&c>=1&&c<=513,"bucket map geometry");return{{at::kInt,{v.numel()*c}}};}
Meta gather_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor(),map=s[1].toTensor();tensor(x,at::kBFloat16,2);tensor(map,at::kInt,1);status(s[2].toTensor());auto r=s[3].toInt(),c=s[4].toInt(),slot=s[5].toInt(),b=s[6].toInt();TORCH_CHECK(x.size(0)<=513&&x.size(1)==6144&&r>=1&&r<=8&&c>=1&&c<=x.size(0)&&slot>=0&&b>=1&&map.numel()%c==0&&slot+b<=map.numel()/c,"bucket gather geometry");return{{at::kBFloat16,{b,c,6144}}};}
Meta gate_meta(const at::Stack&s){devices(s);auto p=s[0].toTensor(),v=s[1].toTensor();tensor(p,at::kFloat,3);tensor(v,at::kInt,1);status(s[2].toTensor());auto slot=s[3].toInt();TORCH_CHECK(p.size(2)==512&&p.size(1)<=513&&slot>=0&&slot+p.size(0)<=v.numel(),"bucket gate geometry");return{{at::kBFloat16,{p.size(0),p.size(1),256}}};}
Meta combine_meta(const at::Stack&s){devices(s);auto p=s[0].toTensor(),r=s[1].toTensor(),inv=s[2].toTensor();tensor(p,at::kFloat,2);tensor(r,at::kFloat,2);tensor(inv,at::kInt,1);status(s[3].toTensor());TORCH_CHECK(p.size(1)>=512&&p.size(1)<=6144&&p.size(1)%512==0&&r.size(0)<=513&&r.size(1)<=8&&inv.numel()==r.numel(),"bucket combine geometry");return{{at::kFloat,{r.size(0),p.size(1)}}};}
Meta decode_meta(const at::Stack&s){devices(s);auto w=s[0].toTensor(),sc=s[1].toTensor(),lut=s[2].toTensor(),map=s[3].toTensor();
 tensor(w,at::kByte,2);tensor(sc,at::kByte,2);tensor(lut,at::kBFloat16,2);tensor(map,at::kInt,2);
 auto k=s[4].toInt(),n=s[5].toInt(),nt=s[6].toInt(),slot=s[7].toInt(),batch=s[8].toInt(),nb=s[9].toInt(),mode=s[10].toInt();
 TORCH_CHECK(((k==6144&&n==512)||(k==256&&n==6144))&&nt>=512&&nt%512==0&&nb>=0&&nb%512==0&&nb+nt<=n&&slot>=0&&batch>=1&&slot+batch<=map.size(1)&&map.size(0)==1&&batch<=16&&batch*k*nt*2<=16*1024*1024&&mode>=0&&mode<=2,"expert decoder tile contract");
 TORCH_CHECK(w.size(1)==256&&sc.size(1)==512&&w.size(0)%((n/512)*k)==0&&w.size(0)/((n/512)*k)<=384&&sc.size(0)*32==w.size(0)&&lut.size(0)==1&&lut.size(1)==512,"HistoricalN512 owner contract");return {{at::kBFloat16,{batch,k,nt}}};}
Meta queue_meta(const at::Stack&s){devices(s);auto ids=s[0].toTensor();tensor(ids,at::kInt,2);TORCH_CHECK(ids.size(0)<=513&&ids.size(1)<=8,"bounded route queue");return{{at::kInt,{ids.numel()*3}},{at::kInt,{ids.numel()*12}}};}
Meta queue_gemv_meta(const at::Stack&s){devices(s);auto w=s[0].toTensor(),sc=s[1].toTensor(),x=s[2].toTensor(),lut=s[3].toTensor(),map=s[4].toTensor(),q=s[5].toTensor();tensor(w,at::kByte,2);tensor(sc,at::kByte,2);tensor(x,at::kBFloat16,2);tensor(lut,at::kBFloat16,2);tensor(map,at::kInt,2);tensor(q,at::kInt,1);TORCH_CHECK((x.size(1)==2048||x.size(1)==256)&&w.size(1)==256&&sc.size(1)==512&&w.size(0)==sc.size(0)*32&&lut.size(0)==1&&lut.size(1)==512&&map.size(0)==1&&map.size(1)==x.size(0)&&q.numel()==x.size(0),"queued masked GEMV geometry");return{{at::kFloat,{1,x.size(0)*512}}};}
Meta counted_gather_meta(const at::Stack&s){devices(s);auto x=s[0].toTensor(),map=s[1].toTensor(),v=s[2].toTensor();tensor(x,at::kBFloat16,2);tensor(map,at::kInt,1);tensor(v,at::kInt,1);status(s[3].toTensor());auto r=s[4].toInt(),c=s[5].toInt(),slot=s[6].toInt(),batch=s[7].toInt(),mode=s[8].toInt();TORCH_CHECK(x.size(0)<=513&&x.size(1)==6144&&r>=1&&r<=8&&c>=1&&c<=x.size(0)&&slot>=0&&batch>=1&&slot+batch<=v.numel()&&map.numel()==v.numel()*c&&mode>=0&&mode<=2,"counted gather geometry");return{{at::kBFloat16,{batch,c,6144}}};}
using Fn=Meta(*)(const at::Stack&);
V run(const char*name,Fn fn,const at::Stack&s){auto m=fn(s);if(s[0].toTensor().device().type()==at::kMeta){V v;for(auto&o:m)v.push_back(at::empty(o.shape,s[0].toTensor().options().dtype(o.dtype)));return v;}auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);return op.execute(s);}
std::shared_ptr<void> params(const at::Stack&s,size_t&n,int first,int count){n=count*4;auto p=std::shared_ptr<int[]>(new int[count]);for(int i=0;i<count;++i)p[i]=s[first+i].toInt();return std::shared_ptr<void>(p,p.get());}
auto prefix(T n,T f,int64_t r,int64_t c,int64_t b,int64_t lo,int64_t hi,int64_t overflow){auto v=run("gaudi_expert_partition::prefix",prefix_meta,{n,f,r,c,b,lo,hi,overflow});return std::make_tuple(v[0],v[1],v[2],v[3],v[4]);}
T inverse(T x,T f,T p,T st,T off,int64_t c,int64_t b){return run("gaudi_expert_partition::inverse",inverse_meta,{x,f,p,st,off,c,b})[0];}
T row_map(T inv,T v,T st,int64_t c){return run("gaudi_expert_partition::row_map",map_meta,{inv,v,st,c})[0];}
T gather(T x,T map,T st,int64_t r,int64_t c,int64_t slot,int64_t b){return run("gaudi_expert_partition::gather",gather_meta,{x,map,st,r,c,slot,b})[0];}
T gate(T p,T v,T st,int64_t slot){return run("gaudi_expert_partition::gate",gate_meta,{p,v,st,slot})[0];}
T combine(T p,T r,T inv,T st){return run("gaudi_expert_partition::combine",combine_meta,{p,r,inv,st})[0];}
T decode(T w,T sc,T lut,T map,int64_t k,int64_t n,int64_t nt,int64_t slot,int64_t batch,int64_t nb,int64_t mode){return run("gaudi_expert_partition::decode",decode_meta,{w,sc,lut,map,k,n,nt,slot,batch,nb,mode})[0];}
auto queue(T ids){auto v=run("gaudi_expert_partition::queue",queue_meta,{ids});return std::make_tuple(v[0],v[1]);}
T queue_gemv(T w,T sc,T x,T lut,T map,T q){return run("gaudi_expert_partition::queue_gemv",queue_gemv_meta,{w,sc,x,lut,map,q})[0];}
T sparse_map(T inv,T v,T st,int64_t c){return run("gaudi_expert_partition::sparse_map",map_meta,{inv,v,st,c})[0];}
T counted_gather(T x,T map,T v,T st,int64_t r,int64_t c,int64_t slot,int64_t batch,int64_t mode){return run("gaudi_expert_partition::counted_gather",counted_gather_meta,{x,map,v,st,r,c,slot,batch,mode})[0];}
}
TORCH_LIBRARY(gaudi_expert_partition,m){
 m.def("sparse_map(Tensor inverse, Tensor valid_rows, Tensor status, int rows) -> Tensor");
 m.def("counted_gather(Tensor x, Tensor row_map, Tensor valid_rows, Tensor status, int routes, int rows, int slot, int batch, int mode) -> Tensor");
 m.def("queue(Tensor ids) -> (Tensor,Tensor)");
 m.def("queue_gemv(Tensor packed, Tensor scales, Tensor x, Tensor lut, Tensor mapping, Tensor order) -> Tensor");
 m.def("decode(Tensor packed, Tensor scales, Tensor lut, Tensor expert_map, int k, int n, int n_tile, int slot_begin, int batch, int n_begin, int empty_mode) -> Tensor");
 m.def("prefix(Tensor counts, Tensor flags, int routes, int rows, int slots, int lower, int upper, int overflow_to_tpc) -> (Tensor,Tensor,Tensor,Tensor,Tensor)");
 m.def("inverse(Tensor ids, Tensor flat_ids, Tensor prefix, Tensor status, Tensor offsets, int rows, int slots) -> Tensor");
 m.def("row_map(Tensor inverse, Tensor valid_rows, Tensor status, int rows) -> Tensor");
 m.def("gather(Tensor x, Tensor row_map, Tensor status, int routes, int rows, int slot, int batch) -> Tensor");
 m.def("gate(Tensor partial, Tensor valid_rows, Tensor status, int slot) -> Tensor");
 m.def("combine(Tensor partial, Tensor routing, Tensor inverse, Tensor status) -> Tensor");
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::sparse_map","gk_expert_sparse_map",map_meta,[](const at::Stack&s,size_t&n){return params(s,n,3,1);});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::counted_gather","gk_expert_counted_gather",counted_gather_meta,[](const at::Stack&s,size_t&n){n=16;auto p=std::shared_ptr<int[]>(new int[4]);p[0]=s[4].toInt();p[1]=s[5].toInt();p[2]=s[6].toInt();p[3]=s[8].toInt();return std::shared_ptr<void>(p,p.get());});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::queue","gk_expert_queue",queue_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::queue_gemv","gk_expert_queue_gemv",queue_gemv_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::decode","gk_expert_decode_k8",decode_meta,[](const at::Stack&s,size_t&n){n=16;auto p=std::shared_ptr<int[]>(new int[4]);p[0]=s[5].toInt()/512;p[1]=s[7].toInt();p[2]=s[9].toInt()/512;p[3]=s[10].toInt();return std::shared_ptr<void>(p,p.get());});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::prefix","gk_expert_prefix",prefix_meta,[](const at::Stack&s,size_t&n){return params(s,n,2,6);});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::inverse","gk_expert_inverse",inverse_meta,[](const at::Stack&s,size_t&n){return params(s,n,5,2);});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::row_map","gk_expert_map",map_meta,[](const at::Stack&s,size_t&n){return params(s,n,3,1);});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::gather","gk_expert_gather",gather_meta,[](const at::Stack&s,size_t&n){return params(s,n,3,3);});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::gate","gk_expert_gate",gate_meta,[](const at::Stack&s,size_t&n){return params(s,n,3,1);});
 habana::custom_op::registerUserCustomOp("gaudi_expert_partition::combine","gk_expert_combine",combine_meta,[](const at::Stack&,size_t&n)->std::shared_ptr<void>{n=0;return nullptr;});
}
#define GK_EXPERT_PARTITION_IMPL m.impl("sparse_map",sparse_map);m.impl("counted_gather",counted_gather);m.impl("queue",queue);m.impl("queue_gemv",queue_gemv);m.impl("decode",decode);m.impl("prefix",prefix);m.impl("inverse",inverse);m.impl("row_map",row_map);m.impl("gather",gather);m.impl("gate",gate);m.impl("combine",combine);
TORCH_LIBRARY_IMPL(gaudi_expert_partition,HPU,m){GK_EXPERT_PARTITION_IMPL}
TORCH_LIBRARY_IMPL(gaudi_expert_partition,Meta,m){GK_EXPERT_PARTITION_IMPL}
