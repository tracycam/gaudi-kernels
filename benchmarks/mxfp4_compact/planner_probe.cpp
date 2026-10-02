// Offline pure-planner probe. Linked with --gc-sections, no Synapse runtime.
#include "../../csrc/ops/mxfp4_compact.hpp"
#include <cstdlib>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=4&&argc!=5)return 2;
 using namespace gaudi_kernels::mxfp4_compact;
 try{
  Plan p;p.native_n_tile=std::atoi(argv[3]);
  if(argc==5)p.row_n_tile=std::atoi(argv[4]);
  auto tiles=plan_tiles(std::atoi(argv[1]),std::atoi(argv[2]),p);
  for(auto&t:tiles)std::cout<<(t.kind==Kind::NativeBody?0:1)<<' '<<t.n_begin<<' '<<t.n_count<<' '<<t.k_begin<<' '<<t.k_count<<' '<<t.block_offset<<' '<<t.pair_offset<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
