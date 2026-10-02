#include "probe_helpers.hpp"
int main(int argc,char**argv){
 if(argc!=2&&argc!=3)return 1;
 bool supplied=argc==3&&std::string(argv[2])=="supplied";if(argc==3&&!supplied)return 1;
 std::string root=argv[1];std::ofstream records(root+"/owned.jsonl");
 ck(synInitialize(),"initialize");synDeviceId dev;ck(synDeviceAcquireByModuleId(&dev,gaudi_experiment_module()),"acquire");
 uint64_t a,b,c,d,weight;for(auto*p:{&a,&b,&c,&d,&weight})ck(synDeviceMalloc(dev,Bytes,0,0,p),"original allocation");
 auto*g=create(dev,root,"good");ok(g,gkg_buffer(g,"input",a,Bytes,GKG_INPUT),"input");ok(g,gkg_buffer(g,"scratch",b,Bytes,GKG_TEMP),"scratch");ok(g,gkg_buffer(g,"result",c,Bytes,GKG_OUTPUT),"output");ok(g,gkg_buffer(g,"weight",weight,Bytes,GKG_EXTERNAL_READ),"external");
 ok(g,gkg_buffer(g,"scratch2",d,Bytes,GKG_TEMP),"scratch2");Producer one(1,root+"/producer-one");ok(g,gkg_recipe(g,"one",one.recipe),"clone");one.close();
 bind(g,"one","input","scratch");for(int i=1;i<7;++i)bind(g,"one",i%2?"scratch":"scratch2",i%2?"scratch2":"scratch");bind(g,"one","scratch","result");
 uint64_t arena=0,arena_bytes=0;if(supplied){ok(g,gkg_pool_requirements(g,&arena_bytes),"arena requirements");ck(synDeviceMalloc(dev,arena_bytes,0,0,&arena),"producer arena");ok(g,gkg_bind_pool(g,arena,arena_bytes),"dedicated arena");}
 ok(g,gkg_instantiate(g),"instantiate");uint64_t own=0;ok(g,gkg_address(g,"scratch",&own),"address");if(own==b)return 4;
 for(auto p:{a,b,c,d})ck(synDeviceFree(dev,p,0),"free original intermediates");
 // Large allocation and a larger compiled/owned graph between replays. This
 // is an allocator/workspace stress gate, not a MiMo prefill coverage claim.
 auto churn=[&](int phase){std::vector<uint64_t>allocs;for(int i=0;i<16;++i){uint64_t p;ck(synDeviceMalloc(dev,1<<20,0,0,&p),"churn allocate");allocs.push_back(p);}for(auto p:allocs)ck(synDeviceFree(dev,p,0),"churn free");
  auto*other=create(dev,root,"interloper"+std::to_string(phase),1);ok(other,gkg_buffer(other,"input",0,Bytes,GKG_INPUT),"interloper input");ok(other,gkg_buffer(other,"output",0,Bytes,GKG_OUTPUT),"interloper output");Producer big(64,root+"/producer-big"+std::to_string(phase));ok(other,gkg_recipe(other,"big",big.recipe),"interloper clone");big.close();bind(other,"big","input","output",64);ok(other,gkg_instantiate(other),"interloper instantiate");
  std::vector<float>x(N,.125f),y(N);gkg_input in{sizeof(in),1,"input",x.data(),Bytes};uint64_t ticket;ok(other,gkg_replay(other,&in,1,&ticket),"interloper replay");ok(other,gkg_wait(other,ticket),"interloper wait");ok(other,gkg_read(other,ticket,"output",y.data(),Bytes),"interloper read");for(float v:y)if(v!=2.125f)return false;ok(other,gkg_release(other,ticket),"interloper release");ok(other,gkg_report(other,(root+"/interloper"+std::to_string(phase)+".json").c_str()),"interloper report");ok(other,gkg_destroy(other),"interloper destroy");return true;
 };
 for(int phase=0;phase<3;++phase){if(!churn(phase))return 5;uint64_t tickets[4];std::vector<std::vector<float>>expected(4,std::vector<float>(N));
  for(int mutation=0;mutation<4;++mutation){std::vector<float>x(N);for(int i=0;i<N;++i){x[i]=float((i*17+phase*19+mutation*7)%131-65)/32;expected[mutation][i]=x[i]+.25f;}gkg_input in{sizeof(in),1,"input",x.data(),Bytes};ok(g,gkg_replay(g,&in,1,&tickets[mutation]),"queue changed input");std::fill(x.begin(),x.end(),12345.f);}
  std::vector<float>x(N),y(N);gkg_input extra{sizeof(extra),1,"input",x.data(),Bytes};uint64_t ignored;
  if(gkg_replay(g,&extra,1,&ignored)!=GKG_BUSY||gkg_destroy(g)!=GKG_BUSY)return 6;
  for(int mutation=0;mutation<4;++mutation){ok(g,gkg_wait(g,tickets[mutation]),"wait");ok(g,gkg_read(g,tickets[mutation],"result",y.data(),Bytes),"read");int bad=0;for(int i=0;i<N;++i)bad+=std::memcmp(&y[i],&expected[mutation][i],4)!=0;records<<"{\"stage\":\"owned_gate\",\"phase\":"<<phase<<",\"mutation\":"<<mutation<<",\"checked\":"<<N<<",\"bad\":"<<bad<<"}\n";if(bad)return 7;ok(g,gkg_release(g,tickets[mutation]),"release");if(gkg_query(g,tickets[mutation])!=GKG_STALE)return 8;}
 }
 // Same graph, order sync / queued / queued / sync. Inputs independent so
 // only ownership/host overlap is measured, not a decode recurrence benefit.
 for(int phase=0;phase<4;++phase){bool queued=phase==1||phase==2;for(int trial=0;trial<5;++trial){std::vector<float>x(N,.25f),y(N);gkg_input in{sizeof(in),1,"input",x.data(),Bytes};auto start=ns();uint64_t submit_ns=0;constexpr int Repeats=100;
   for(int group=0;group<Repeats;group+=queued?4:1){int batch=queued?4:1;uint64_t t[4];auto begun=ns();for(int i=0;i<batch;++i)ok(g,gkg_replay(g,&in,1,&t[i]),"timing replay");submit_ns+=ns()-begun;for(int i=0;i<batch;++i){ok(g,gkg_wait(g,t[i]),"timing wait");ok(g,gkg_read(g,t[i],"result",y.data(),Bytes),"timing read");for(float v:y)if(v!=.5f)return 9;ok(g,gkg_release(g,t[i]),"timing release");}}
   records<<"{\"stage\":\"owned_timing\",\"phase\":"<<phase<<",\"queued\":"<<(queued?"true":"false")<<",\"trial\":"<<trial<<",\"wall_us\":"<<(ns()-start)/100000.<<",\"submit_us\":"<<submit_ns/100000.<<"}\n";
  }
 }
 ok(g,gkg_check_external(g,"weight",weight,Bytes),"external stable");if(gkg_check_external(g,"weight",weight+256,Bytes)!=GKG_STALE)return 10;
 ok(g,gkg_report(g,(root+"/good-report.json").c_str()),"report");ok(g,gkg_destroy(g),"destroy good");
 if(supplied){
  for(int kind=0;kind<3;++kind){auto*bad=create(dev,root,"arena-reject"+std::to_string(kind),1);
   ok(bad,gkg_buffer(bad,"input",kind==2?arena:0,Bytes,GKG_INPUT),"arena reject input");ok(bad,gkg_buffer(bad,"output",0,Bytes,GKG_OUTPUT),"arena reject output");ok(bad,gkg_copy(bad,"input",0,"output",0,Bytes),"arena reject copy");
   auto status=gkg_bind_pool(bad,kind==1?arena+1:arena,kind==0?8:arena_bytes);
   if(kind!=1){ok(bad,status,"arena reject bind");status=gkg_instantiate(bad);}if(status!=GKG_RANGE)return 15;
   records<<"{\"stage\":\"pool_rejection\",\"kind\":"<<kind<<",\"status\":"<<status<<"}\n";ok(bad,gkg_destroy(bad),"arena reject destroy");
  }
  ck(synDeviceFree(dev,arena,0),"producer owns arena release");
 }
 for(int kind=0;kind<4;++kind){auto*bad=create(dev,root,"reject"+std::to_string(kind),1);ok(bad,gkg_buffer(bad,"input",0,Bytes,GKG_INPUT),"reject input");ok(bad,gkg_buffer(bad,"scratch",0,Bytes,GKG_TEMP),"reject scratch");ok(bad,gkg_buffer(bad,"output",0,Bytes,GKG_OUTPUT),"reject output");gkg_status status;
  if(kind==0){ok(bad,gkg_copy(bad,"scratch",0,"output",0,Bytes),"uninitialized copy");status=gkg_instantiate(bad);if(status!=GKG_UNINITIALIZED)return 11;}
  else if(kind==1){ok(bad,gkg_copy(bad,"input",0,"output",0,Bytes-4),"partial output");status=gkg_instantiate(bad);if(status!=GKG_UNINITIALIZED)return 12;}
  else if(kind==2){status=gkg_copy(bad,"input",1,"output",0,Bytes);if(status!=GKG_RANGE)return 13;}
  else{ok(bad,gkg_buffer(bad,"weight",weight,Bytes,GKG_EXTERNAL_READ),"reject weight");ok(bad,gkg_copy(bad,"input",0,"weight",0,Bytes),"weight overwrite");status=gkg_instantiate(bad);if(status!=GKG_READ_ONLY)return 14;}
  records<<"{\"stage\":\"rejection\",\"kind\":"<<kind<<",\"status\":"<<status<<"}\n";ok(bad,gkg_report(bad,(root+"/reject"+std::to_string(kind)+".json").c_str()),"reject report");ok(bad,gkg_destroy(bad),"reject destroy");
 }
 ck(synDeviceFree(dev,weight,0),"external free");ck(synDeviceRelease(dev),"release device");ck(synDestroy(),"destroy Synapse");return 0;
}
