#pragma once
#include "../../csrc/common/experiment_device.hpp"
#include <synapse_api.h>
#include <synapse_common_types.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
inline void check(synStatus s,const char*where){if(s!=synSuccess)throw std::runtime_error(std::string(where)+" status="+std::to_string(s));}
inline double h8(uint8_t u){int e=(u>>2)&31,m=u&3;if(e==31)throw std::runtime_error("nonfinite E5M2");double v=e?std::ldexp(double(4+m),e-17):std::ldexp(double(m),-16);return std::copysign(v,u&128?-1.:1.);}
inline void save(const std::string&name,const void*data,size_t bytes){auto p=std::string(std::getenv("PROBE_OUT"))+"/"+name;FILE*f=std::fopen(p.c_str(),"wb");if(!f||std::fwrite(data,1,bytes,f)!=bytes)throw std::runtime_error("save "+p);std::fclose(f);}
struct Buffer{std::string name;synTensor tensor;uint64_t bytes,address;void*host;bool output;};
struct Harness{
 bool closed=false;
 ~Harness(){if(!closed){synDeviceRelease(dev);synDestroy();}}
 synDeviceId dev;synGraphHandle graph;synRecipeHandle recipe;synStreamHandle stream;uint64_t workspace=0,workspace_bytes=0;std::vector<Buffer> buffers;std::vector<synLaunchTensorInfo> launches;
 Harness(){check(synInitialize(),"initialize");check(synDeviceAcquireByModuleId(&dev,gaudi_experiment_module()),"acquire assigned module");check(synGraphCreate(&graph,synDeviceGaudi2),"graph");buffers.reserve(16);}
 synTensor tensor(std::string name,synDataType dtype,std::vector<int>dims,bool persistent,uint64_t bytes=0,bool output=false){
  synTensorDescriptor d{};d.m_name=name.c_str();d.m_dataType=dtype;d.m_dims=dims.size();for(unsigned i=0;i<dims.size();++i)d.m_sizes[i]=d.m_minSizes[i]=dims[i];
  synSectionHandle section=nullptr;if(persistent){check(synSectionCreate(&section,0,graph),"section");check(synSectionSetPersistent(section,true),"persistent");}
  synTensor t;check(synTensorCreate(&t,&d,section,0),"tensor");
  if(dtype==syn_type_fp8_152||dtype==syn_type_fp8_143){synFpQuantParam p{1.,dtype==syn_type_fp8_152?15u:7u};synFpQuantMetadata meta{dtype,&p,1};check(synTensorSetQuantizationData(t,SYN_FP_QUANT_METADATA,&meta,sizeof(meta)),"FP8 metadata");}
  if(persistent){Buffer b{name,t,bytes,0,nullptr,output};check(synHostMalloc(dev,bytes,0,&b.host),"host allocation");check(synDeviceMalloc(dev,bytes,0,0,&b.address),"device allocation");std::memset(b.host,0,bytes);buffers.push_back(b);}return t;
 }
 void node(const char*guid,const char*name,std::vector<synTensor>in,std::vector<synTensor>out,void*params=nullptr,unsigned bytes=0){check(synNodeCreate(graph,in.data(),out.data(),in.size(),out.size(),params,bytes,guid,name,nullptr,nullptr),name);}
 void compile(){check(synGraphCompile(&recipe,graph,"fp8_mme_contract",nullptr),"compile");}
 bool compile_only(){if(std::getenv("GK_FP8_CONTRACT_COMPILE_ONLY")){std::puts("{\"stage\":\"compile_only\"}");return true;}return false;}
 void prepare(){check(synStreamCreateGeneric(&stream,dev,0),"stream");for(auto&b:buffers){synLaunchTensorInfo l{};l.tensorName=b.name.c_str();l.tensorType=DATA_TENSOR;l.pTensorAddress=b.address;launches.push_back(l);if(!b.output)check(synMemCopyAsync(stream,uint64_t(b.host),b.bytes,b.address,HOST_TO_DRAM),"upload");save(b.name+".bin",b.host,b.bytes);}check(synWorkspaceGetSize(&workspace_bytes,recipe),"workspace size");if(workspace_bytes)check(synDeviceMalloc(dev,workspace_bytes,0,0,&workspace),"workspace allocation");}
 void run(){check(synLaunch(stream,launches.data(),launches.size(),workspace,recipe,SYN_FLAGS_TENSOR_NAME),"launch");}
 void poison(){for(auto&b:buffers)if(b.output)check(synMemsetD32Async(b.address,0x7fc12345,b.bytes/4,stream),"poison");}
 void download(const std::string&suffix){for(auto&b:buffers)if(b.output)check(synMemCopyAsync(stream,b.address,b.bytes,uint64_t(b.host),DRAM_TO_HOST),"download");check(synStreamSynchronize(stream),"download sync");for(auto&b:buffers)if(b.output)save(b.name+suffix+".bin",b.host,b.bytes);}
 void close(){check(synDeviceRelease(dev),"release");check(synDestroy(),"destroy");closed=true;}
};
