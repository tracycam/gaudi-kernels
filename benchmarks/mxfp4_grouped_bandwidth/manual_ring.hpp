#pragma once
// Diagnostic host graph only. The decoder ELF and numerical contract are unchanged.
// Ordinary transients are NOT a placement promise: audit the compiled graph.
// Observed: bundling can discard requested control edges. Only compiled storage
// and traces qualify a schedule. No caller-owned buffers are aliased here.
template<class MakeTensor>
static void append_manual_ring(synGraphHandle graph,MakeTensor internal,
                              synTensor w,synTensor scales,synTensor lut,synTensor ids,
                              synTensor x,synTensor y,int E,int N,int K,int M,
                              int slots,int chunk,bool prefill){
 if(slots<2||slots>4||chunk<1||(E+chunk-1)/chunk<slots)
  throw std::invalid_argument("manual ring requires 2..4 slots and at least that many chunks");
 if(uint64_t(slots)*chunk*N*K*2>48ull*1024*1024)
  throw std::invalid_argument("manual ring live weights exceed physical SRAM");
 const int count=(E+chunk-1)/chunk;
 const bool data_prefill=std::getenv("GK_GROUPED_RING_DATA_PREFILL")!=nullptr;
 if(data_prefill&&slots!=3)throw std::invalid_argument("data prefill diagnostic requires exactly three slots");
 std::vector<synTensor> av,iv,yv,dv;
 for(int i=0;i<count;++i){
  auto tag="ring_"+std::to_string(i);
  int ce=std::min(chunk,E-i*chunk);
  av.push_back(internal((tag+"_activation").c_str(),syn_type_bf16,{K,M,ce}));
  iv.push_back(internal((tag+"_ids").c_str(),syn_type_int32,{ce,1}));
  yv.push_back(internal((tag+"_output").c_str(),syn_type_single,{N,M,ce}));
  dv.push_back(internal(("decoded_weights_ring_"+std::to_string(i)).c_str(),syn_type_bf16,{N,K,ce}));
 }
 synSplitParams ax{2},ix{0};synConcatenateParams cat{2};
 synTensor used_x=x;
 if(data_prefill){
  used_x=internal("ring_ready_activation",syn_type_bf16,{K,M,E});
  synTensor gate_inputs[]={x,dv[0],dv[1],dv[2]};
  ck(synNodeCreate(graph,gate_inputs,&used_x,4,1,nullptr,0,"gk_grouped_ring_gate","ring_data_prefill",nullptr,nullptr),"activation data prefill");
 }
 ck(synNodeCreate(graph,&used_x,av.data(),1,count,&ax,sizeof(ax),"split","ring_activation_views",nullptr,nullptr),"split activation views");
 ck(synNodeCreate(graph,&ids,iv.data(),1,count,&ix,sizeof(ix),"split","ring_id_views",nullptr,nullptr),"split id views");
 std::vector<synNodeId> decode(count),mme(count);
 for(int i=0;i<count;++i){
  std::string tag="ring_"+std::to_string(i);
  synTensor di[]={w,scales,lut,iv[i]};
  ck(synNodeCreateWithId(graph,di,&dv[i],4,1,nullptr,0,"gk_grouped_mxfp4_decode",(tag+"_decode").c_str(),&decode[i],nullptr,nullptr),"manual decode");
  synTensor mi[]={av[i],dv[i]};synGEMMParams gp{false,false};
  ck(synNodeCreateWithId(graph,mi,&yv[i],2,1,&gp,sizeof(gp),"batch_gemm",(tag+"_mme").c_str(),&mme[i],nullptr,nullptr),"manual MME");
 }
 auto edge=[&](synNodeId a,synNodeId b){ck(synNodeDependencySet(graph,&a,&b,1,1),"ring lifetime edge");};
 for(int i=1;i<count;++i){edge(decode[i-1],decode[i]);edge(mme[i-1],mme[i]);}
 for(int i=slots;i<count;++i)edge(mme[i-slots],decode[i]);
 // Requests prefill ordering; this is NOT proof that the compiler retains it.
 // The compile/trace audit is mandatory (the current stack discards this edge).
 if(prefill)edge(decode[slots-1],mme[0]);
 ck(synNodeCreate(graph,yv.data(),&y,count,1,&cat,sizeof(cat),"concat","ring_output_views",nullptr,nullptr),"concat output views");
 std::printf("{\"stage\":\"manual_ring\",\"slots\":%d,\"chunk_experts\":%d,\"chunks\":%d,\"prefill\":%s,\"data_prefill\":%s,\"live_weight_budget_bytes\":%llu,\"placement\":\"compiler transient, must audit\"}\n",slots,chunk,count,prefill?"true":"false",data_prefill?"true":"false",(unsigned long long)(uint64_t(slots)*chunk*N*K*2));
}
