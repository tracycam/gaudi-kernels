#include "bf16_linear.hpp"
#include <synapse_common_types.hpp>
#include <string>
namespace gaudi_kernels {
synStatus add_bf16_linear(synGraphHandle graph, synTensor x, synTensor w,
    synTensor bias, synTensor y, const Bf16LinearOptions& o, Bf16LinearGraph* r) {
    if (!r || !o.m || !o.n || !o.k || o.split_m >= o.m || (o.tpc && (o.m > 2 || o.split_m))) return synInvalidArgument;
    if(o.row_dot && (o.tpc || o.split_m || o.weight_transposed || o.m>8)) return synInvalidArgument;
    if(o.row_tail && (!o.split_m || o.m-o.split_m>8 || o.weight_transposed || o.tpc || o.row_dot)) return synInvalidArgument;
    synStatus status = synSuccess;
    const std::string prefix=std::string(o.name_prefix?o.name_prefix:"bf16_linear")+"_";
    const uint64_t partial_bytes=uint64_t(o.n)*o.m*(o.tpc?(o.k+255)/256:1)*4;
    if(o.sram_intermediate && partial_bytes>16777216) return synInvalidArgument;
    auto tensor=[&](const std::string& name, synDataType type, unsigned d0,unsigned d1,unsigned d2=0) {
        synTensorDescriptor d={};std::string full=prefix+name;d.m_name=full.c_str();d.m_dataType=type;d.m_dims=d2?3:2;
        d.m_sizes[0]=d.m_minSizes[0]=d0;d.m_sizes[1]=d.m_minSizes[1]=d1;
        if(d2)d.m_sizes[2]=d.m_minSizes[2]=d2;
        synSectionHandle section=nullptr;
        if(o.sram_intermediate && (name=="partials" || name=="accumulated")){
            if(status==synSuccess)status=synSectionCreate(&section,0,graph);
            if(section)r->sections.push_back(section);
            if(status==synSuccess)status=synSectionSetPersistent(section,false);
            if(status==synSuccess)status=synSectionSetRMW(section,true);
        }
        synTensor t=nullptr;if(status==synSuccess)status=synTensorCreate(&t,&d,section,0);
        if(t)r->intermediates.push_back(t);return t;
    };
    auto node=[&](synTensor* in,unsigned ni,synTensor* out,unsigned no,const void* p,unsigned bytes,const char* guid,const char* name) {
        if(status==synSuccess)status=synNodeCreate(graph,in,out,ni,no,p,bytes,guid,(prefix+name).c_str(),nullptr,nullptr);
    };
    if(o.row_dot) {
        synTensor inputs[]={x,w,bias};
        std::string guid=std::string(o.row_dot_unroll4?"gk_bf16_rowdot4":"gk_bf16_rowdot")+std::string(bias?"_bias":"")+(o.output_bf16?"_bf16":"_f32");
        node(inputs,bias?3:2,&y,1,nullptr,0,guid.c_str(),"rowdot");
        return status;
    }
    if(o.tpc) {
        synTensor partial=tensor("partials",syn_type_single,o.n,o.m,(o.k+255)/256);
        synTensor inputs[]={x,w};node(inputs,2,&partial,1,nullptr,0,o.tpc_native_mac?(o.m==1?"gk_bf16_gemv_acc321":"gk_bf16_gemv_acc322"):(o.m==1?"gk_bf16_gemv1":"gk_bf16_gemv2"),"gemv");
        synTensor ep[]={partial,bias};std::string guid="gk_bf16_reduce"+std::string(bias?"_bias":"")+(o.output_bf16?"_bf16":"_f32");
        node(ep,bias?2:1,&y,1,nullptr,0,guid.c_str(),"reduce");
        return status;
    }
    synTensor accumulated=bias?tensor("accumulated",syn_type_single,o.n,o.m):y;
    synDataType type=bias||!o.output_bf16?syn_type_single:syn_type_bf16;
    synGEMMParams params{false,!o.weight_transposed};
    if(o.split_m) {
        synTensor xs[]={tensor("x_main",syn_type_bf16,o.k,o.split_m),tensor("x_tail",syn_type_bf16,o.k,o.m-o.split_m)};
        synTensor ys[]={tensor("y_main",type,o.n,o.split_m),tensor("y_tail",type,o.n,o.m-o.split_m)};
        synSplitParams split{1};node(&x,1,xs,2,&split,sizeof(split),"split","split_rows");
        for(unsigned i=0;i<2;++i){synTensor inputs[]={xs[i],w};
            if(i && o.row_tail){std::string guid=std::string(o.row_dot_unroll4?"gk_bf16_rowdot4":"gk_bf16_rowdot")+(type==syn_type_bf16?"_bf16":"_f32");node(inputs,2,&ys[i],1,nullptr,0,guid.c_str(),"tpc_tail");}
            else node(inputs,2,&ys[i],1,&params,sizeof(params),"gemm",i?"mme_tail":"mme_main");
        }
        synConcatenateParams concat{1};node(ys,2,&accumulated,1,&concat,sizeof(concat),"concat","join_rows");
    } else {synTensor inputs[]={x,w};node(inputs,2,&accumulated,1,&params,sizeof(params),"gemm","mme");}
    if(bias) {
        synTensor inputs[]={accumulated,bias};std::string guid="gk_bf16_bias"+std::string(o.output_bf16?"_bf16":"_f32");
        node(inputs,2,&y,1,nullptr,0,guid.c_str(),"bias_before_rounding");
    }
    return status;
}
}
