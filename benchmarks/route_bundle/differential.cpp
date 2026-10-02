// CPU only: original public C glue versus the single-database dispatcher.
#include <tpc_kernel_lib_interface.h>
#include <array>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace tpc_lib_api;
using Instantiate=GlueCodeReturn(*)(HabanaKernelParams*,HabanaKernelInstantiation*);
void check(bool ok,const std::string&why){if(!ok)throw std::runtime_error(why);}
struct Api {
 void*h; pfnGetKernelGuids guids; Instantiate instantiate; pfnGetShapeInference shape;
 pfnGetSupportedDataLayout layout; pfnGetLibVersion version;
 explicit Api(const char*path){h=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());
  guids=(pfnGetKernelGuids)dlsym(h,"GetKernelGuids");instantiate=(Instantiate)dlsym(h,"InstantiateTpcKernel");
  shape=(pfnGetShapeInference)dlsym(h,"GetShapeInference");layout=(pfnGetSupportedDataLayout)dlsym(h,"GetSupportedDataLayout");
  version=(pfnGetLibVersion)dlsym(h,"GetLibVersion");check(guids&&instantiate&&shape&&layout&&version,"missing API");
  check(!dlsym(h,"GetSupportedDataLayouts"),"unexpected new plural layout API");check(version()==1,"version!=1");}
};
Tensor tensor(TensorDataType type,std::initializer_list<unsigned>shape){Tensor t{};t.geometry.dataType=type;t.geometry.dims=shape.size();t.quantizationParam.scale=1;
 for(unsigned i=0;i<MAX_TENSOR_DIM;++i)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=1;
 unsigned i=0;for(auto n:shape)t.geometry.minSizes[i]=t.geometry.maxSizes[i]=n,++i;return t;}
struct Case{unsigned owner;std::string name;std::vector<Tensor>in,out;std::vector<int>params;};
std::vector<Case>cases(){
 auto I=[](std::initializer_list<unsigned>s){return tensor(DATA_I32,s);};
 auto F=[](std::initializer_list<unsigned>s){return tensor(DATA_F32,s);};
 auto B=[](std::initializer_list<unsigned>s){return tensor(DATA_BF16,s);};
 auto U=[](std::initializer_list<unsigned>s){return tensor(DATA_U8,s);};
 return {
 {0,"gk_mxfp4_graph_count",{I({2,3})},{I({4,1}),I({3,1})},{}},
 {0,"gk_mxfp4_graph_plan",{I({4,1}),I({3,1})},{I({4,1}),I({4,1})},{2,4}},
 {0,"gk_mxfp4_graph_gather",{B({64,3}),I({2,3}),I({4,1}),I({4,1})},{B({64,3,2})},{2,4}},
 {0,"gk_mxfp4_graph_decode_historical",{U({256,128}),U({512,4}),B({512,1}),I({2,1})},{B({512,64,2})},{1,0,0}},
 {0,"gk_mxfp4_graph_gate_rows",{F({512,3,2}),I({2,1}),I({2,1})},{B({256,3,2})},{0}},
 {0,"gk_mxfp4_graph_combine",{F({512,6}),F({2,3}),I({2,3}),I({6}),I({4})},{F({512,3})},{2}},
 {0,"gk_mxfp4_graph_literal_gate",{F({256,3}),F({256,3})},{B({256,3})},{256}},
 {1,"gk_route_count_i32_v3",{I({2,3}),I({6})},{I({4}),I({3}),I({2,4})},{4}},
 {1,"gk_route_prefix_i32_v3",{I({4}),I({3})},{I({5}),I({4}),I({4}),I({4}),I({1})},{2,2,4}},
 {1,"gk_route_inverse_i32_v3",{I({2,3}),I({6}),I({5}),I({1}),I({2,4})},{I({6})},{2,4}},
 {1,"gk_route_row_map_i32_v3",{I({6}),I({4}),I({1})},{I({8})},{2}},
 {2,"gk_route_tile_gather_bf16_v1",{B({6144,3}),I({8}),I({1})},{B({6144,2,2})},{2,2,1}},
 {2,"gk_route_tile_gate_bf16_v1",{F({512,2,2}),I({4}),I({1})},{B({256,2,2})},{1}},
 {2,"gk_route_tile_combine_f32_v1",{F({512,8}),F({2,3}),I({6}),I({1})},{F({512,3})},{}}
 };}
struct Snapshot {
 GlueCodeReturn status;HabanaKernelParams in{};HabanaKernelInstantiation out{};
 std::array<TensorAccessPattern,16>ia{},oa{};std::vector<Tensor>inputs,outputs;std::vector<unsigned char>elf;
};
Snapshot invoke(Api&api,Case c,unsigned capacity,int mutation=0){
 Snapshot s;s.inputs=c.in;s.outputs=c.out;s.elf.resize(capacity+64,0x5a);
 auto&p=s.in;auto&k=s.out;p.deviceId=DEVICE_ID_GAUDI2;std::strcpy(p.guid.name,c.name.c_str());
 p.inputTensors=s.inputs.data();p.outputTensors=s.outputs.data();p.inputTensorNr=c.in.size();p.outputTensorNr=c.out.size();
 p.nodeParams.nodeParams=c.params.data();p.nodeParams.nodeParamsSize=c.params.size()*4;
 if(mutation==1)++p.inputTensorNr;if(mutation==2)++p.outputTensorNr;if(mutation==3)p.nodeParams.nodeParamsSize+=4;
 if(mutation==4)s.inputs[0].geometry.dataType=s.inputs[0].geometry.dataType==DATA_I32?DATA_F32:DATA_I32;
 if(mutation==5)s.outputs[0].geometry.maxSizes[0]=0;
 if(mutation==6)s.inputs[0].geometry.minSizes[0]=0;
 if(mutation==7)p.deviceId=DEVICE_ID_GAUDI;
 k.inputTensorAccessPattern=s.ia.data();k.outputTensorAccessPattern=s.oa.data();k.kernel.kernelElf=s.elf.data();k.kernel.elfSize=capacity;
 s.status=api.instantiate(&p,&k);
 check(k.kernel.kernelElf==s.elf.data(),"unexpected replaced ELF owner");
 p.inputTensors=p.outputTensors=nullptr;p.nodeParams.nodeParams=nullptr;
 k.inputTensorAccessPattern=k.outputTensorAccessPattern=nullptr;k.kernel.kernelElf=nullptr;
 return s;
}
void equal(const Snapshot&a,const Snapshot&b,const std::string&name){
 check(a.status==b.status,name+" return code");check(!std::memcmp(&a.in,&b.in,sizeof(a.in)),name+" input params mutation");
 check(!std::memcmp(&a.out,&b.out,sizeof(a.out)),name+" instantiation fields");
 check(!std::memcmp(a.ia.data(),b.ia.data(),sizeof(a.ia)),name+" input access patterns");
 check(!std::memcmp(a.oa.data(),b.oa.data(),sizeof(a.oa)),name+" output access patterns");
 check(!std::memcmp(a.inputs.data(),b.inputs.data(),a.inputs.size()*sizeof(Tensor)),name+" input tensor mutation");
 check(!std::memcmp(a.outputs.data(),b.outputs.data(),a.outputs.size()*sizeof(Tensor)),name+" output tensor mutation");
 check(a.elf==b.elf,name+" full ELF or buffer canary");
}
void layout_equal(Api&a,Api&b,const Case&c,unsigned cap){
 std::array<TensorDataLayout,16>ai,ao,bi,bo;std::memset(ai.data(),0x5a,sizeof(ai));std::memset(ao.data(),0x5a,sizeof(ao));bi=ai;bo=ao;
 NodeDataLayouts al{},bl{};al.inputs=ai.data();al.outputs=ao.data();bl.inputs=bi.data();bl.outputs=bo.data();
 HabanaKernelParams p{};std::strcpy(p.guid.name,c.name.c_str());p.inputTensorNr=c.in.size();p.outputTensorNr=c.out.size();
 uint32_t ac=cap,bc=cap;auto ar=a.layout(&p,cap?&al:nullptr,&ac),br=b.layout(&p,cap?&bl:nullptr,&bc);
 check(ar==br&&ac==bc,"layout status/count");check(!std::memcmp(ai.data(),bi.data(),sizeof(ai))&&!std::memcmp(ao.data(),bo.data(),sizeof(ao)),"layout fields");
 al.inputs=al.outputs=nullptr;bl.inputs=bl.outputs=nullptr;check(!std::memcmp(&al,&bl,sizeof(al)),"layout descriptor");
}
int main(int argc,char**argv){try{
 check(argc==7,"core metadata tiles bundle duplicate_bundle output_dir required");
 Api core(argv[1]),meta(argv[2]),tiles(argv[3]),bundle(argv[4]),duplicate(argv[5]);Api*old[]={&core,&meta,&tiles};
 std::filesystem::path output=argv[6];std::filesystem::create_directories(output);
 std::array<GuidInfo,16>expected;std::memset(expected.data(),0x5a,sizeof(expected));unsigned offset=0;
 for(auto*api:old){uint32_t n=0;check(api->guids(DEVICE_ID_GAUDI2,&n,nullptr)==GLUE_SUCCESS,"original guid size");unsigned count=n;check(api->guids(DEVICE_ID_GAUDI2,&n,expected.data()+offset)==GLUE_SUCCESS&&n==count,"original guids");offset+=n;}
 check(offset==14,"fourteen GUIDs");
 for(unsigned cap=0;cap<=16;++cap){std::array<GuidInfo,16>actual;std::memset(actual.data(),0x5a,sizeof(actual));uint32_t n=cap;
  check(bundle.guids(DEVICE_ID_GAUDI2,&n,actual.data())==GLUE_SUCCESS&&n==14,"bundle guid count");
  for(unsigned i=0;i<16;++i){GuidInfo sentinel;std::memset(&sentinel,0x5a,sizeof(sentinel));check(!std::memcmp(&actual[i],i<std::min(cap,14u)?&expected[i]:&sentinel,sizeof(GuidInfo)),"GUID capacity/caller-byte preservation");}}
 uint32_t n=99;check(bundle.guids(DEVICE_ID_GAUDI,&n,nullptr)==GLUE_SUCCESS&&n==0,"unsupported device enum");
 n=0;check(duplicate.guids(DEVICE_ID_GAUDI2,&n,nullptr)==GLUE_FAILED&&n==0,"duplicate registry not rejected");
 unsigned comparisons=0;auto fixtures=cases();std::cout<<"{\"status\":\"PASS_CPU_GLUE_DIFFERENTIAL\",\"guids\":[";
 for(unsigned index=0;index<fixtures.size();++index){auto c=fixtures[index];Api&original=*old[c.owner];
  check(c.name==expected[index].name,"fixture GUID order");
  auto required=invoke(original,c,0),query=invoke(bundle,c,0);equal(required,query,c.name);++comparisons;
  check(required.status==GLUE_INSUFFICIENT_ELF_BUFFER&&required.out.kernel.elfSize>0,"valid fixture rejected");
  unsigned size=required.out.kernel.elfSize;
  for(unsigned cap:{size-1,size,size+64}){auto a=invoke(original,c,cap),b=invoke(bundle,c,cap);equal(a,b,c.name);++comparisons;check(a.status==(cap<size?GLUE_INSUFFICIENT_ELF_BUFFER:GLUE_SUCCESS),"buffer negotiation");}
  for(int mutation=1;mutation<=7;++mutation){equal(invoke(original,c,size,mutation),invoke(bundle,c,size,mutation),c.name);++comparisons;}
  auto good=invoke(bundle,c,size);std::ofstream elf(output/(c.name+".o"),std::ios::binary);elf.write((char*)good.elf.data(),size);
  std::ofstream descriptor(output/(c.name+".descriptor"),std::ios::binary);descriptor.write((char*)&good.out,sizeof(good.out));descriptor.write((char*)good.ia.data(),sizeof(good.ia));descriptor.write((char*)good.oa.data(),sizeof(good.oa));
  layout_equal(original,bundle,c,0);layout_equal(original,bundle,c,1);
  for(int mode=0;mode<3;++mode){ShapeInferenceParams p{};ShapeInferenceOutput ao{},bo{};GuidInfo named{};
   std::strcpy(p.guid.name,mode==1?"unknown":c.name.c_str());std::strcpy(named.name,mode==2?"unknown":c.name.c_str());if(mode)p.pGuid=&named;
   auto ar=original.shape(DEVICE_ID_GAUDI2,&p,&ao),br=bundle.shape(DEVICE_ID_GAUDI2,&p,&bo);check(ar==br&&!std::memcmp(&ao,&bo,sizeof(ao)),"shape/pGuid dispatch");}
  check(invoke(duplicate,c,size).status==GLUE_FAILED,"duplicate instantiate accepted");
  if(index)std::cout<<',';std::cout<<"{\"name\":\""<<c.name<<"\",\"elf_bytes\":"<<size<<",\"instantiate_cases\":11}";
 }
 Case unknown=fixtures[0];unknown.name="gk_unknown_not_registered";
 for(auto*api:old){equal(invoke(*api,unknown,0),invoke(bundle,unknown,0),"unknown GUID");layout_equal(*api,bundle,unknown,1);}
 std::cout<<"],\"instantiate_comparisons\":"<<comparisons<<",\"GUID_capacity_cases\":17,\"duplicate_rejected\":true,\"device_acquired\":false}"<<std::endl;
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
