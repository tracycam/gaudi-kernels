// One SDK database; delegate to byte-identical, manifest-pinned providers.
// Sidecars are fixed filenames beside this library, never searched via LD_PATH.
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <filesystem>
#include <vector>
#include <cstring>
#include <algorithm>
using namespace tpc_lib_api;
namespace {
void anchor() {}
struct Provider {
    using Guids = GlueCodeReturn (*)(DeviceId,uint32_t*,GuidInfo*);
    using Instantiate = GlueCodeReturn (*)(HabanaKernelParams*,HabanaKernelInstantiation*);
    using Shape = GlueCodeReturn (*)(DeviceId,const ShapeInferenceParams*,ShapeInferenceOutput*);
    using Layout = GlueCodeReturn (*)(const HabanaKernelParams*,NodeDataLayouts*,uint32_t*);
    void* handle=nullptr;
    Guids guids=nullptr;
    Instantiate instantiate=nullptr;
    Shape shape=nullptr;
    Layout layout=nullptr;
    unsigned count=0;
    bool open(const std::filesystem::path& path, unsigned expected) {
        handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(!handle)return false;
        guids=reinterpret_cast<Guids>(dlsym(handle,"GetKernelGuids"));
        instantiate=reinterpret_cast<Instantiate>(dlsym(handle,"InstantiateTpcKernel"));
        shape=reinterpret_cast<Shape>(dlsym(handle,"GetShapeInference"));
        layout=reinterpret_cast<Layout>(dlsym(handle,"GetSupportedDataLayout"));
        auto version=reinterpret_cast<uint64_t(*)()>(dlsym(handle,"GetLibVersion"));
        return guids&&instantiate&&shape&&layout&&version&&version()==1&&
            guids(DEVICE_ID_GAUDI2,&count,nullptr)==GLUE_SUCCESS&&count==expected;
    }
};
struct Registry {
    Provider providers[2];
    std::vector<GuidInfo> guids;
    std::vector<unsigned> owners;
    bool valid=false;
    Registry() {
        Dl_info info{};
        if(!dladdr(reinterpret_cast<void*>(&anchor),&info)||!info.dli_fname)return;
        auto dir=std::filesystem::absolute(info.dli_fname).parent_path();
        const char* files[]={"provider_batch.so","provider_expert.so"};
        unsigned counts[]={7,12};
        for(unsigned p=0;p<2;++p){
            if(!providers[p].open(dir/files[p],counts[p]))return;
            uint32_t count=counts[p];std::vector<GuidInfo> local(count);
            if(providers[p].guids(DEVICE_ID_GAUDI2,&count,local.data())!=GLUE_SUCCESS||count!=counts[p])return;
            for(const auto& g:local){
                if(!std::memchr(g.name,0,sizeof(g.name))||!g.name[0])return;
                for(const auto& existing:guids)if(!std::strcmp(g.name,existing.name))return;
                guids.push_back(g);owners.push_back(p);
            }
        }
        valid=true;
    }
    int owner(const char* name)const {
        if(!valid)return -1;
        for(size_t i=0;i<guids.size();++i)if(!std::strcmp(name,guids[i].name))return owners[i];
        return -1;
    }
};
const Registry& registry(){static const Registry r;return r;}
}
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device,uint32_t* count,GuidInfo* out){
    if(!count)return GLUE_FAILED;
    const auto&r=registry();unsigned capacity=*count;
    if(!r.valid){*count=0;return GLUE_FAILED;}
    *count=device==DEVICE_ID_GAUDI2?r.guids.size():0;
    if(out)std::copy_n(r.guids.data(),std::min(capacity,*count),out);
    return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* in,HabanaKernelInstantiation* out){
    if(!in||!out)return GLUE_FAILED;const auto&r=registry();int p=r.owner(in->guid.name);
    return p<0?GLUE_NODE_NOT_FOUND:r.providers[p].instantiate(in,out);
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId d,const ShapeInferenceParams* in,ShapeInferenceOutput* out){
    if(!in)return GLUE_FAILED;const auto&r=registry();int p=r.owner(in->pGuid?in->pGuid->name:in->guid.name);
    return p<0?GLUE_NODE_NOT_FOUND:r.providers[p].shape(d,in,out);
}
extern "C" GlueCodeReturn GetSupportedDataLayout(const HabanaKernelParams* in,NodeDataLayouts* out,uint32_t* count){
    if(!in)return GLUE_FAILED;const auto&r=registry();int p=r.owner(in->guid.name);
    return p<0?GLUE_NODE_NOT_FOUND:r.providers[p].layout(in,out,count);
}
extern "C" uint64_t GetLibVersion(){return 1;}
