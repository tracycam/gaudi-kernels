// Local instruction simulator only. No simulator implementation is vendored.
#include <tpc_test_core_api.h>
#include <tpc_kernel_lib_interface.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
static Tensor tensor(TensorDataType type,unsigned width,unsigned rows=1){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;for(unsigned d=0;d<MAX_TENSOR_DIM;++d)t.geometry.minSizes[d]=t.geometry.maxSizes[d]=d==0?width:d==1?rows:1;t.quantizationParam.scale=1;return t;}
static tpc_tests::TensorDesc2 descriptor(void*data,unsigned width,unsigned log2,unsigned rows=1){tpc_tests::TensorDesc2 d{};d.baseAddrUnion.baseAddr=reinterpret_cast<uint64_t>(data);d.configuration=log2|(3u<<8)|(1u<<16);unsigned stride=1;for(unsigned i=0;i<5;++i){unsigned size=i==0?width:i==1?rows:1;d.dimDescriptors[i].size=size;d.dimDescriptors[i].stride=stride;stride*=size;}return d;}
static uint16_t bf(float x){uint32_t bits;std::memcpy(&bits,&x,4);return bits>>16;}
int main(int argc,char**argv){
 if(argc!=2||!std::getenv("TPC_RUNNER")||std::strcmp(std::getenv("TPC_RUNNER"),"0"))return 2;
 bool fused=std::string(argv[1])!="direct",narrow=std::string(argv[1])=="fused256"||std::string(argv[1])=="fused256c";unsigned n=1536,k=32,tokens=2,routes=3,experts=2,nb=n/512;
 std::vector<uint8_t>w(experts*n*k/2),s(experts*n*k/32,127),elf(1<<20);
 std::vector<uint16_t>a(tokens*routes*k),lut(512);std::vector<int32_t>ids={0,1,-1,1,0,1};std::vector<float>routing={.5f,.25f,7.f,.25f,.5f,.25f};
 for(unsigned e=0;e<experts;++e)for(unsigned i=0;i<n*k/2;++i)w[e*n*k/2+i]=e?0x44:0x22;
 for(unsigned row=0;row<tokens*routes;++row)for(unsigned z=0;z<k;++z)a[row*k+z]=bf(float(row+1));
 const float q[]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};for(int i=0;i<256;++i){lut[2*i]=bf(q[i&15]);lut[2*i+1]=bf(q[i>>4]);}
 unsigned rows=fused?tokens:tokens*routes;std::vector<float>y(n*rows,-999);
 std::vector<Tensor>inputs={tensor(DATA_PACKED_MXFP4,narrow?128:256,experts*nb*k*(narrow?2:1)),tensor(DATA_U8,narrow?256:512,experts*nb*k/32*(narrow?2:1)),tensor(DATA_BF16,k,tokens*routes),tensor(DATA_BF16,512),tensor(DATA_I32,routes,tokens),tensor(DATA_F32,routes,tokens)};
 std::vector<tpc_tests::TensorDesc2>desc={descriptor(w.data(),narrow?128:256,0,experts*nb*k*(narrow?2:1)),descriptor(s.data(),narrow?256:512,0,experts*nb*k/32*(narrow?2:1)),descriptor(a.data(),k,1,tokens*routes),descriptor(lut.data(),512,1),descriptor(ids.data(),routes,2,tokens),descriptor(routing.data(),routes,2,tokens),descriptor(y.data(),n,2,rows)};
 Tensor outputs[]={tensor(DATA_F32,n,rows)};TensorAccessPattern in_ap[6]{},out_ap[1]{};AuxTensor auxiliary[16]{};HabanaKernelParams in{};HabanaKernelInstantiation out{};
 in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=inputs.size();in.outputTensorNr=1;in.inputTensors=inputs.data();in.outputTensors=outputs;
 std::strcpy(in.guid.name,std::string(argv[1])=="fused256c"?"gk_mxfp4_moe_fused256c":narrow?"gk_mxfp4_moe_fused256":fused?"gk_mxfp4_moe_fused512":"gk_mxfp4_moe_direct512");if(narrow)nb*=2;uint32_t params[]={nb,uint32_t(((uint64_t(1)<<32)+nb-1)/nb),experts,uint32_t(((uint64_t(1)<<32)+routes-1)/routes)};in.nodeParams.nodeParams=params;in.nodeParams.nodeParamsSize=16;
 out.inputTensorAccessPattern=in_ap;out.outputTensorAccessPattern=out_ap;out.auxiliaryTensors=auxiliary;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();auto status=InstantiateTpcKernel(&in,&out);if(status!=GLUE_SUCCESS){std::cerr<<"glue "<<status<<"\n";return 3;}
 VPEStats stats;unsigned cycles=tpc_tests::RunSimulation(in,out,desc,stats);unsigned bad=0;
 for(unsigned row=0;row<rows;++row){float expected=0;if(fused){for(unsigned slot=0;slot<routes;++slot){unsigned at=row*routes+slot;if(ids[at]>=0)expected+=float(k*(at+1)*(ids[at]+1))*routing[at];}}else if(ids[row]>=0)expected=k*(row+1)*(ids[row]+1);
  for(unsigned col=0;col<n;++col)if(y[row*n+col]!=expected){if(bad<8)std::cerr<<row<<","<<col<<" "<<y[row*n+col]<<" expected "<<expected<<"\n";++bad;}}
 std::cout<<"{\"mode\":\""<<argv[1]<<"\",\"checked\":"<<y.size()<<",\"bad\":"<<bad<<",\"cycles\":"<<cycles<<",\"weight_loads\":"<<stats.vectorLoadPerTensor[0]<<",\"scale_loads\":"<<stats.vectorLoadPerTensor[1]<<"}\n";return bad?4:0;
}
