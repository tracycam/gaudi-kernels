#pragma once
#include <synapse_api.h>
#include <vector>
namespace gaudi_kernels {
// Synapse dimension order is fastest-first. X=[K,M], output=[N,M].
// MME W=[K,N]; TPC W=[N,K] is an offline BF16 transpose, without quantization.
// Bias is FP32 [N,1], added to FP32 accumulated values before final rounding.
struct Bf16LinearOptions {
    unsigned m, n, k;
    bool output_bf16 = true;
    bool tpc = false;  // experimental; M <= 2; split K into 256-wide chunks
    unsigned split_m = 0;  // explicit measured candidate; 0 leaves compiler geometry
    bool tpc_native_mac = false; // [ceil(N/128)*128,K], lane map j=(i>>1)|((i&1)<<6)
    bool sram_intermediate = false; // caller opts into <=16MiB Gaudi2 RMW section
    const char* name_prefix = "bf16_linear"; // caller must make unique within graph
    bool weight_transposed = false; // MME W=[N,K] fastest-first, prepared offline
    bool row_dot = false; // whole-row BF16 TPC candidate; original NK weights, M<=8
    bool row_dot_unroll4 = false; // four independent K chains; FP32 final reduction
    bool row_tail = false; // split main MME plus <=8 whole-row TPC tail tokens
};
struct Bf16LinearGraph {
    std::vector<synTensor> intermediates;
    std::vector<synSectionHandle> sections;
};
// Adds nodes to the caller's graph. No device acquisition, allocations, or launch.
// Caller retains supplied tensors and destroys returned intermediates with graph.
synStatus add_bf16_linear(synGraphHandle graph, synTensor x, synTensor weight,
    synTensor bias, synTensor output, const Bf16LinearOptions& options,
    Bf16LinearGraph* resources);
}
