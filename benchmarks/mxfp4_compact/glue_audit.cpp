// CPU-only invocation of the actual built TPC glue. No Synapse runtime/device.
#include <tpc_kernel_lib_interface.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <initializer_list>
using namespace tpc_lib_api;
extern "C" GlueCodeReturn GetKernelGuids(DeviceId,uint32_t*,GuidInfo*);
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
static void geometry(Tensor&t,TensorDataType type,std::initializer_list<unsigned>sizes){
 t={};t.geometry.dataType=type;t.geometry.dims=sizes.size();unsigned d=0;
 for(auto size:sizes){t.geometry.minSizes[d]=t.geometry.maxSizes[d]=size;++d;}
}
int main(){
 unsigned count=0;assert(GetKernelGuids(DEVICE_ID_GAUDI2,&count,nullptr)==GLUE_SUCCESS&&count==3);
 GuidInfo guids[3]={};unsigned cap=1;assert(GetKernelGuids(DEVICE_ID_GAUDI2,&cap,guids)==GLUE_INCOMPATIBLE_OUTPUT_SIZE);
 cap=3;assert(GetKernelGuids(DEVICE_ID_GAUDI2,&cap,guids)==GLUE_SUCCESS);
 unsigned pass=0;
 for(unsigned kind=0;kind<3;++kind)for(unsigned k:kind?std::vector<unsigned>{1,2,3,15,16,17,31,32,64,256,288}:std::vector<unsigned>{32,64,256}){
  for(unsigned width:kind?std::vector<unsigned>{1,127,511,513}:std::vector<unsigned>{256,512}){
   Tensor input[3]={},output[1]={};TensorAccessPattern ia[3]={},oa[1]={};
   int params[2]={1,width==256?1:0};
   if(kind){geometry(input[0],DATA_U8,{(k+1)/2,width});geometry(input[1],DATA_U8,{(k+31)/32,width});geometry(output[0],DATA_BF16,{k,width});}
   else{geometry(input[0],DATA_U8,{256,k,2});geometry(input[1],DATA_U8,{512,k/32,2});geometry(output[0],DATA_BF16,{width,k});}
   geometry(input[2],DATA_BF16,{512,1});HabanaKernelParams p={};p.deviceId=DEVICE_ID_GAUDI2;p.guid=guids[kind];p.inputTensors=input;p.outputTensors=output;p.inputTensorNr=3;p.outputTensorNr=1;
   if(!kind){p.nodeParams.nodeParams=params;p.nodeParams.nodeParamsSize=8;}
   HabanaKernelInstantiation out={};out.inputTensorAccessPattern=ia;out.outputTensorAccessPattern=oa;
   assert(InstantiateTpcKernel(&p,&out)==GLUE_INSUFFICIENT_ELF_BUFFER);
   std::vector<char>elf(out.kernel.elfSize);out.kernel.kernelElf=elf.data();
   assert(InstantiateTpcKernel(&p,&out)==GLUE_SUCCESS&&out.auxiliaryTensorNr==0);
   assert(elf.size()>4&&elf[0]==127&&elf[1]=='E'&&elf[2]=='L'&&elf[3]=='F');
   unsigned task=kind==2?256:32;assert(out.indexSpaceGeometry[0]==(k+task-1)/task&&out.indexSpaceGeometry[1]==(kind?width:1));
   if(kind){assert(ia[0].mapping[0].end_b==int((k<task?(k+1)/2:task/2)-1));assert(ia[1].mapping[0].a==task/32);}
   else{assert(ia[0].mapping[0].start_b==params[1]*128);assert(ia[0].mapping[0].end_b==params[1]*128+int(width/2)-1);}
   input[0].geometry.dataType=DATA_BF16;assert(InstantiateTpcKernel(&p,&out)==GLUE_INCOMPATIBLE_DATA_TYPE);input[0].geometry.dataType=DATA_U8;
   output[0].geometry.maxSizes[0]=uint64_t(1)<<32;assert(InstantiateTpcKernel(&p,&out)==GLUE_INCOMPATIBLE_INPUT_SIZE);
   ++pass;
  }
 }
 std::cout<<"{\"state\":\"cpu_glue_pass_device_unvalidated\",\"valid_cases\":"<<pass<<",\"bad_dtype_and_huge_geometry_rejected\":true}\n";
}
