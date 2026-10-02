#include "fp8_first_principles.hpp"
namespace gaudi_kernels::experimental::fp8_first_principles {
synStatus add_quantizer(synGraphHandle graph,Quantizer kind,synTensor x,
                       synTensor table,synTensor q,synTensor scales){
 const char* guid;
 switch(kind){
  case Quantizer::Amax16:guid="fp8_fp_quant_amax16";break;
  case Quantizer::LutCorrected:guid="fp8_fp_quant_lut_corrected";break;
  case Quantizer::LutSingle:guid="fp8_fp_quant_lut_single";break;
  case Quantizer::LutSingle4:guid="fp8_fp_quant_lut_single4";break;
  default:return synInvalidArgument;
 }
 synTensor inputs[]={x,table},outputs[]={q,scales};
 return synNodeCreate(graph,inputs,outputs,kind==Quantizer::Amax16?1:2,2,
                      nullptr,0,guid,"offline_quant_candidate",nullptr,nullptr);
}
}
