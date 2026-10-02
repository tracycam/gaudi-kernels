// CPU-only public ABI instantiation and bounds contracts.
#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <iostream>
#include <vector>
using namespace tpc_lib_api;
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*);
Tensor tensor(TensorDataType type,unsigned width,unsigned height){Tensor t{};t.geometry.dims=2;t.geometry.dataType=type;for(unsigned d=0;d<5;++d)t.geometry.minSizes[d]=t.geometry.maxSizes[d]=d==0?width:d==1?height:1;return t;}
int main(){unsigned checked=0;
 for(unsigned experts:{8u,24u,384u})for(unsigned variant=0;variant<3;++variant){bool old=variant==2;unsigned wf=old?256:128,rows=experts*(old?3072:6144);
  std::vector<Tensor>inputs={tensor(DATA_U8,wf,rows),tensor(DATA_U8,wf*2,rows/32),tensor(DATA_BF16,256,8),tensor(DATA_BF16,512,1),tensor(DATA_I32,8,1)};if(!old){inputs.push_back(tensor(DATA_F32,8,1));inputs.push_back(tensor(DATA_U8,256,2));}
  Tensor output=tensor(DATA_F32,old?49152:6144,1);HabanaKernelParams in{};HabanaKernelInstantiation out{};std::vector<unsigned char>elf(1<<20);std::vector<TensorAccessPattern>access(inputs.size());TensorAccessPattern outaccess{};in.deviceId=DEVICE_ID_GAUDI2;in.inputTensorNr=inputs.size();in.outputTensorNr=1;in.inputTensors=inputs.data();in.outputTensors=&output;
  std::strcpy(in.guid.name,old?"gk_down_combine_old_body_v0":variant?"gk_down_combine_m1_p6_v0":"gk_down_combine_m1_p4_v0");out.inputTensorAccessPattern=access.data();out.outputTensorAccessPattern=&outaccess;out.kernel.kernelElf=elf.data();out.kernel.elfSize=elf.size();
  if(InstantiateTpcKernel(&in,&out)!=GLUE_SUCCESS||out.indexSpaceGeometry[0]!=(old?96:24)||outaccess.mapping[0].a!=(old?512:256)||out.kernel.paramsNr!=(old?1:0))return 2;
  if(old&&out.kernel.scalarParams[0]!=536870912)return 3;
  ++checked;inputs[2].geometry.maxSizes[0]=257;if(InstantiateTpcKernel(&in,&out)==GLUE_SUCCESS)return 4;inputs[2].geometry.maxSizes[0]=256;
  inputs[4].geometry.dataType=DATA_F32;if(InstantiateTpcKernel(&in,&out)==GLUE_SUCCESS)return 5;inputs[4].geometry.dataType=DATA_I32;
  output.geometry.maxSizes[0]--;if(InstantiateTpcKernel(&in,&out)==GLUE_SUCCESS)return 6;
 }
 std::cout<<"{\"device_used\":false,\"valid_instantiations\":"<<checked<<",\"negative_contract_checks\":"<<checked*3<<",\"all_pass\":true}\n";
}
