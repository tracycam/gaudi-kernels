#include "mxfp4_overhead.hpp"
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
namespace gaudi_kernels::mxfp4_overhead {
static void ck(synStatus s,const char*w){if(s!=synSuccess)throw std::runtime_error(std::string(w)+": "+std::to_string(s));}
static void require(synTensor t,synDataType dtype,std::initializer_list<int> sizes){
 if(!t)throw std::invalid_argument("missing prepared N512 tensor");
 synTensorGeometry shape={};synDataType type;ck(synTensorGetGeometry(t,&shape,synGeometryMaxSizes),"shape");ck(synTensorGetDeviceDataType(t,&type),"dtype");
 if(type!=dtype||shape.dims!=sizes.size())throw std::invalid_argument("N512-native-v2 tensor dtype/rank mismatch");
 unsigned i=0;for(int size:sizes)if(shape.sizes[i++]!=unsigned(size))throw std::invalid_argument("N512-native-v2 tensor shape mismatch");
}
Geometry geometry(int n,int k,int m,int splits){
 if(n<1||k<1||m<1||splits<0)throw std::invalid_argument("invalid GEMV dimensions");
 int nb=(n+511)/512,maxsplit=(k+31)/32;
 if(!splits){splits=std::min(maxsplit,std::max(1,(24+nb*m-1)/(nb*m)));while(maxsplit%splits)--splits;}
 if(splits>maxsplit||maxsplit%splits)throw std::invalid_argument("K32 partitions must divide load-time padding");
 int kp=maxsplit/splits*32;
 return {n,k,m,splits,kp,nb,nb*splits*m,kp*splits};
}
Resources append(synGraphHandle graph,synTensor packed,synTensor scales,synTensor x,synTensor lut,synTensor mapping,synTensor output,synTensor bias,const Geometry&g,const std::string&kind,const std::string&prefix){
 if(kind!="history"&&kind!="guarded"&&kind!="exact512"&&kind!="prologue")throw std::invalid_argument("unknown GEMV implementation");
 if(g.layout_version!=2)throw std::invalid_argument("N512-native-v2 required; N256-v1 is not interchangeable");
 auto expected=geometry(g.n,g.k,g.m,g.splits);
 if(g.nblocks!=expected.nblocks||g.kpart!=expected.kpart||g.kprepared!=expected.kprepared||g.tasks!=expected.tasks)throw std::invalid_argument("inconsistent prepared geometry");
 require(packed,syn_type_packed_mxfp4,{256,g.nblocks*g.kprepared});
 require(scales,syn_type_uint8,{512,g.nblocks*g.kprepared/32});
 require(x,syn_type_bf16,{g.k,g.m});require(lut,syn_type_bf16,{512,1});require(mapping,syn_type_int32,{g.tasks,1});
 require(output,syn_type_single,{g.n,g.m});if(bias)require(bias,syn_type_single,{g.n,1});
 Resources r;auto tensor=[&](std::string name,synDataType dt,int a,int b){synTensorDescriptor d={};d.m_name=name.c_str();d.m_dataType=dt;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=a;d.m_sizes[1]=d.m_minSizes[1]=b;synTensor t;ck(synTensorCreate(&t,&d,nullptr,0),"temporary tensor");r.tensors.push_back(t);return t;};
 auto ax=tensor(prefix+"_activation",syn_type_bf16,g.kpart,g.tasks);
 auto flags=tensor(prefix+"_unsafe",syn_type_uint16,1,g.tasks);
 auto partial=tensor(prefix+"_partials",syn_type_single,g.tasks*512,1);
 int params[]={g.nblocks*g.splits,g.splits};synTensor po[]={ax,flags};
 ck(synNodeCreate(graph,&x,po,1,2,params,8,"gk_mxfp4_prepare",(prefix+"_prepare").c_str(),nullptr,nullptr),"prepare");
 synTensor gi[]={packed,scales,ax,lut,mapping,flags};std::string guid="gk_mxfp4_"+kind;
 ck(synNodeCreate(graph,gi,&partial,kind=="history"||kind=="prologue"?5:6,1,nullptr,0,guid.c_str(),(prefix+"_compute").c_str(),nullptr,nullptr),"GEMV");
 params[0]=g.nblocks;synTensor fi[]={partial,bias};
 ck(synNodeCreate(graph,fi,&output,bias?2:1,1,params,8,bias?"gk_mxfp4_finish_bias":"gk_mxfp4_finish",(prefix+"_finish").c_str(),nullptr,nullptr),"finish");
 return r;
}
}
