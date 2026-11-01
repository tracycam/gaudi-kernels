// Keep the existing library's GUIDs and ELFs unchanged; add the previously
// validated TP8 gate without consuming another Synapse library slot.
#include <dlfcn.h>
#include <cstdlib>
#define GetKernelGuids gate_GetKernelGuids
#define InstantiateTpcKernel gate_InstantiateTpcKernel
#define GetShapeInference gate_GetShapeInference
#define GetSupportedDataLayout gate_GetSupportedDataLayout
#define GetLibVersion gate_GetLibVersion
#include "../../benchmarks/fp32_tp8_collective/glue.cpp"
#undef GetKernelGuids
#undef InstantiateTpcKernel
#undef GetShapeInference
#undef GetSupportedDataLayout
#undef GetLibVersion
#ifndef PARENT_LIBRARY
#error PARENT_LIBRARY must identify the frozen original library
#endif
template<class F> static F parent(const char* name) {
  static void* handle=dlopen(PARENT_LIBRARY,RTLD_NOW|RTLD_LOCAL);
  if(!handle) std::abort();
  auto fn=reinterpret_cast<F>(dlsym(handle,name));
  if(!fn) std::abort();
  return fn;
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId d,uint32_t* n,GuidInfo* g){
  auto f=parent<decltype(&gate_GetKernelGuids)>("GetKernelGuids");
  uint32_t capacity=*n,a=0,b=0;auto rc=f(d,&a,nullptr);if(rc!=GLUE_SUCCESS)return rc;
  rc=gate_GetKernelGuids(d,&b,nullptr);if(rc!=GLUE_SUCCESS)return rc;*n=a+b;
  if(g&&capacity>=*n){uint32_t count=a;rc=f(d,&count,g);if(rc!=GLUE_SUCCESS)return rc;
    count=b;return gate_GetKernelGuids(d,&count,g+a);}
  return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*p,HabanaKernelInstantiation*q){
  return known(p->guid.name)?gate_InstantiateTpcKernel(p,q):
    parent<decltype(&gate_InstantiateTpcKernel)>("InstantiateTpcKernel")(p,q);
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId d,const ShapeInferenceParams*p,ShapeInferenceOutput*q){
  return known(p->pGuid?p->pGuid->name:p->guid.name)?gate_GetShapeInference(d,p,q):
    parent<decltype(&gate_GetShapeInference)>("GetShapeInference")(d,p,q);
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*p,NodeDataLayouts*l,uint32_t*n){
  return known(p->guid.name)?gate_GetSupportedDataLayout(p,l,n):
    parent<decltype(&gate_GetSupportedDataLayout)>("GetSupportedDataLayout")(p,l,n);
}
extern "C" uint64_t GetLibVersion(){return parent<decltype(&gate_GetLibVersion)>("GetLibVersion")();}
