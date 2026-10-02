"""Exercise the real C ABI binding code with an instrumented SDK substitute."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class NativeMultiBindingTests(unittest.TestCase):
    def test_complete_inputs_validate_before_mutation_and_partial_failure_is_fatal(self):
        header = Path(__file__).resolve().parents[1] / 'csrc/legacy_executor'
        code = r'''
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <cassert>
enum {HOST_TO_DRAM=0,DRAM_TO_HOST=1};
struct Command {int kind,type;std::string semantic_role;uint64_t count;
  void* src;void* dst;std::vector<unsigned char> host_bytes;};
static std::vector<Command> commands;
static void* serial_stream=(void*)1;
static std::atomic<bool> active{false};
static int rejected=0;
static struct {bool ready=false;} native_step_state;
struct Guard {};
static int calls=0,error=0,syncs=0;
static int enqueue(unsigned,uint64_t* a,uint64_t* b,uint64_t* c){
  ++calls;*a=1;*b=2;*c=3;return error;
}
static int synStreamSynchronize(void*){++syncs;return 0;}
#include "native_multi.inc"
int main(){
  std::vector<uint32_t> memory(10,7),new_values(10,99);
  size_t j=0;
  for(auto& role:native_input_roles){
    commands.push_back({6,HOST_TO_DRAM,role,4,&memory[j++],nullptr,{0,0,0,0}});
  }
  uint32_t sample=42;
  commands.push_back({6,DRAM_TO_HOST,"sampled_tokens",4,nullptr,&sample,{}});
  uint64_t bytes=0,times[4]={};uint32_t output=0;
  assert(native_multi_prepare(&bytes)==0 && bytes==4);
  std::vector<NativeInputBinding> bindings;
  j=0;
  for(auto& role:native_input_roles)bindings.push_back({role.c_str(),&new_values[j++],4});
  auto saved=bindings.back();bindings.back().bytes=8;
  assert(native_multi_step(bindings.data(),10,&output,4,times)==-4);
  assert(calls==0 && memory==std::vector<uint32_t>(10,7));
  bindings.back()=saved;bindings.back().role=bindings.front().role;
  assert(native_multi_step(bindings.data(),10,&output,4,times)==-3);
  assert(calls==0 && memory==std::vector<uint32_t>(10,7));
  bindings.back()=saved;
  assert(native_multi_step(bindings.data(),10,&output,4,times)==0);
  assert(calls==1 && syncs==1 && output==42 && memory==new_values);
  error=8;
  assert(native_multi_step(bindings.data(),10,&output,4,times)==10008);
  assert(!native_multi_state.ready && calls==2 && syncs==2);
  assert(native_multi_step(bindings.data(),10,&output,4,times)==-1 && calls==2);
  native_multi_state=NativeMultiState{};
  commands.push_back(commands[0]);
  assert(native_multi_prepare(&bytes)==-3 && !native_multi_state.ready);
}'''
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'test'
            subprocess.run(['g++', '-std=c++17', '-I'+str(header), '-x', 'c++', '-', '-o', str(target)],
                           input=code, text=True, check=True)
            subprocess.run([str(target)], check=True)
