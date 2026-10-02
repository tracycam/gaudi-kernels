#pragma once
// Shared helpers for native graph device probes.
#include "../../csrc/native_graph/graph.h"
#include "../../csrc/common/experiment_device.hpp"
#include <synapse_api.h>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
constexpr int N=6144;constexpr uint64_t Bytes=N*4;
void ck(synStatus s,const char*w){if(s!=synSuccess){std::cerr<<w<<": "<<s<<'\n';std::exit(2);}}
void ok(gkg_graph*g,gkg_status s,const char*w){if(s!=GKG_OK){std::cerr<<w<<": "<<s<<" "<<gkg_error(g)<<'\n';std::exit(3);}}
uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Producer {
 synGraphHandle graph{};synRecipeHandle recipe{};std::vector<synTensor>tensors;std::vector<synSectionHandle>sections;std::vector<std::string>names;
 Producer(int nodes,const std::string&path){
  ck(synGraphCreate(&graph,synDeviceGaudi2),"graph");tensors.resize(nodes+1);sections.resize(nodes+1);names.reserve(nodes+1);
  for(int i=0;i<=nodes;++i){names.push_back("v"+std::to_string(i));if(i==0||i==nodes){ck(synSectionCreate(&sections[i],0,graph),"section");ck(synSectionSetPersistent(sections[i],true),"persistent");}
   synTensorDescriptor d{};d.m_name=names.back().c_str();d.m_dataType=syn_type_single;d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=N;d.m_sizes[1]=d.m_minSizes[1]=1;ck(synTensorCreate(&tensors[i],&d,sections[i],0),"tensor");
  }
  float increment=.03125f;for(int i=0;i<nodes;++i)ck(synNodeCreate(graph,&tensors[i],&tensors[i+1],1,1,&increment,4,"gk_native_graph_affine_f32_v1",("add"+std::to_string(i)).c_str(),nullptr,nullptr),"node");ck(synGraphCompile(&recipe,graph,path.c_str(),nullptr),"compile");
 }
 void close(){ck(synRecipeDestroy(recipe),"original recipe destroy");recipe=nullptr;for(auto t:tensors)ck(synTensorDestroy(t),"tensor destroy");for(auto s:sections)if(s)ck(synSectionDestroy(s),"section destroy");ck(synGraphDestroy(graph),"graph destroy");}
};
gkg_graph*create(synDeviceId dev,const std::string&root,const std::string&name,int flights=4){std::string directory=root+'/'+name;std::filesystem::create_directory(directory);gkg_config c{sizeof(c),GKG_ABI_VERSION,dev,uint32_t(flights),directory.c_str()};gkg_graph*g=nullptr;ok(g,gkg_create(&c,&g),"create");return g;}
void bind(gkg_graph*g,const char*recipe,const char*input,const char*output,int last=1){std::string out="v"+std::to_string(last);gkg_binding b[2]={{sizeof(gkg_binding),1,"v0",input,0},{sizeof(gkg_binding),1,out.c_str(),output,0}};ok(g,gkg_launch(g,recipe,b,2),"launch plan");}
