#pragma once
#include <synapse_api.h>
#include <string>
#include <vector>
namespace gaudi_kernels::mxfp4_overhead {
struct Geometry {int n,k,m,splits,kpart,nblocks,tasks,kprepared;int layout_version=2;};
// Experimental opt-in API; existing N256-v1 dispatch stays unchanged.
// Floating-point limitation: FP32 subnormal products/results are not fully
// supported (hardware flushes some contributions). See qualification report.
// Static plan; split<=ceil(K/32), zero pad only the final K32 partition.
Geometry geometry(int n,int k,int m,int splits=0);
struct Resources {std::vector<synTensor> tensors;};
// Prepared N512-native-v2 includes a reversible 128-lane even/odd
// permutation so native FP32 accumulator stores are in logical N order.
// N512 prepared bytes and E8M0 remain original-width HBM inputs. The mapping
// contains task % (nblocks*splits). The caller supplies device/graph/lifetimes.
// kind=history is a bounded comparator: activation nonzeros must lie in
// [2^-118,2^119]. kind=guarded selects exact arithmetic for unsafe tasks.
Resources append(synGraphHandle graph,synTensor packed,synTensor scales,synTensor x,
 synTensor lut,synTensor mapping,synTensor output,synTensor bias,const Geometry& g,
 const std::string& kind="guarded",const std::string& prefix="mxfp4_overhead");
}
