// Experimental single-database forwarding only: qualified TPC ELFs are unchanged.
#include <tpc_kernel_lib_interface.h>
#include <algorithm>
#include <array>
#include <cstring>
using namespace tpc_lib_api;

#define DECLARE(role) \
extern "C" GlueCodeReturn gk_bundle_##role##_GetKernelGuids(DeviceId,uint32_t*,GuidInfo*); \
extern "C" GlueCodeReturn gk_bundle_##role##_InstantiateTpcKernel(HabanaKernelParams*,HabanaKernelInstantiation*); \
extern "C" GlueCodeReturn gk_bundle_##role##_GetShapeInference(DeviceId,const ShapeInferenceParams*,ShapeInferenceOutput*); \
extern "C" GlueCodeReturn gk_bundle_##role##_GetSupportedDataLayout(const HabanaKernelParams*,NodeDataLayouts*,uint32_t*); \
extern "C" uint64_t gk_bundle_##role##_GetLibVersion();
DECLARE(core) DECLARE(metadata) DECLARE(tiles)

namespace {
struct Provider {
    decltype(&gk_bundle_core_GetKernelGuids) guids;
    decltype(&gk_bundle_core_InstantiateTpcKernel) instantiate;
    decltype(&gk_bundle_core_GetShapeInference) shape;
    decltype(&gk_bundle_core_GetSupportedDataLayout) layout;
    decltype(&gk_bundle_core_GetLibVersion) version;
    unsigned count;
};
#define PROVIDER(role,n) {gk_bundle_##role##_GetKernelGuids, \
 gk_bundle_##role##_InstantiateTpcKernel,gk_bundle_##role##_GetShapeInference, \
 gk_bundle_##role##_GetSupportedDataLayout,gk_bundle_##role##_GetLibVersion,n}
const Provider providers[]={PROVIDER(core,7),PROVIDER(metadata,4),PROVIDER(tiles,3)};

struct Registry {
    std::array<GuidInfo,14> guids{};
    std::array<unsigned,14> owners{};
    bool valid=false;
    Registry() {
        unsigned offset=0;
        for(unsigned owner=0;owner<3;++owner) {
            const auto&p=providers[owner];uint32_t count=0;
            if(p.version()!=1 || p.guids(DEVICE_ID_GAUDI2,&count,nullptr)!=GLUE_SUCCESS
               || count!=p.count)return;
            // The original core enumerator assumes enough capacity. Always give
            // it its full count; the outer aggregate follows SDK capacity bounds.
            std::array<GuidInfo,7> local{};
            if(p.guids(DEVICE_ID_GAUDI2,&count,local.data())!=GLUE_SUCCESS
               || count!=p.count)return;
            for(unsigned i=0;i<count;++i) {
                if(!std::memchr(local[i].name,0,sizeof(local[i].name)) || !local[i].name[0])return;
#if defined(GK_ROUTE_BUNDLE_DUPLICATE_TEST) && GK_ROUTE_BUNDLE_DUPLICATE_TEST
                if(owner==1 && i==0)std::strcpy(local[i].name,guids[0].name);
#endif
                for(unsigned previous=0;previous<offset;++previous)
                    if(!std::strcmp(local[i].name,guids[previous].name))return;
                guids[offset]=local[i];owners[offset++]=owner;
            }
        }
        valid=offset==guids.size();
    }
    int owner(const char*name)const {
        if(!valid)return -2;
        for(unsigned i=0;i<guids.size();++i)
            if(!std::strcmp(name,guids[i].name))return owners[i];
        return -1;
    }
};
const Registry&registry(){static const Registry value;return value;}
GlueCodeReturn missing(int owner){return owner==-2?GLUE_FAILED:GLUE_NODE_NOT_FOUND;}
}

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device,uint32_t*count,GuidInfo*guids) {
    if(!count)return GLUE_FAILED;
    const auto&r=registry();unsigned capacity=*count;
    if(!r.valid){*count=0;return GLUE_FAILED;}
    *count=device==DEVICE_ID_GAUDI2?r.guids.size():0;
    if(guids && *count) {
        unsigned offset=0;
        for(const auto&p:providers) {
            unsigned available=offset<capacity?std::min(capacity-offset,p.count):0;
            std::array<GuidInfo,7> local{};
            // Preserve caller bytes not written by the original provider too.
            if(available)std::copy_n(guids+offset,available,local.data());
            uint32_t size=p.count;
            auto result=p.guids(device,&size,local.data());
            if(result!=GLUE_SUCCESS)return result;
            if(size!=p.count)return GLUE_FAILED;
            if(available)std::copy_n(local.data(),available,guids+offset);
            offset+=p.count;
        }
    }
    return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams*in,HabanaKernelInstantiation*out) {
    if(!in||!out)return GLUE_FAILED;
    int owner=registry().owner(in->guid.name);
    return owner<0?missing(owner):providers[owner].instantiate(in,out);
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId device,const ShapeInferenceParams*in,ShapeInferenceOutput*out) {
    if(!in)return GLUE_FAILED;
    int owner=registry().owner(in->pGuid?in->pGuid->name:in->guid.name);
    return owner<0?missing(owner):providers[owner].shape(device,in,out);
}
// Deliberately preserve the three original libraries' singular export. The
// installed header's string macro is plural; adding that alias would activate a
// different loader path and is outside this byte/behavior-preserving bundle.
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams*in,NodeDataLayouts*layouts,uint32_t*count) {
    if(!in)return GLUE_FAILED;
    int owner=registry().owner(in->guid.name);
    return owner<0?missing(owner):providers[owner].layout(in,layouts,count);
}
extern "C" uint64_t GetLibVersion(){return 1;}
