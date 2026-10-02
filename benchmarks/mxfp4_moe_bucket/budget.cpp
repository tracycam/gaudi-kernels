#include "../../csrc/ops/mxfp4_moe_bucket.hpp"
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){if(argc!=5)return 2;using namespace gaudi_kernels::mxfp4_moe_bucket;Plan p;p.unique_experts_per_token=true;p.scratch_buffers=std::atoi(argv[4]);auto b=budget(512,6144,std::atoi(argv[1]),std::atoi(argv[3]),std::atoi(argv[2]),p);std::printf("{\"row_slots\":%d,\"expert_slots\":%d,\"expert_batch\":%d,\"per_buffer_sram_bytes\":%llu,\"sram_bytes\":%llu,\"A_bytes\":%llu,\"Y_bytes\":%llu,\"mme_nodes\":%u}\n",b.row_slots,b.expert_slots,b.expert_batch,(unsigned long long)b.per_buffer_sram_bytes,(unsigned long long)b.decoded_sram_bytes,(unsigned long long)b.grouped_activation_bytes,(unsigned long long)b.fp32_output_bytes,b.mme_nodes);}
