from pathlib import Path
import subprocess
import tempfile
import unittest


class ReplayScheduleTests(unittest.TestCase):
    def test_event_filter_preserves_call_order_and_late_input_mutation(self):
        header = Path(__file__).resolve().parents[1]/'csrc/legacy_executor'
        code = r'''
#include "replay_schedule.hpp"
#include <cassert>
struct Command {int kind,value;};
int main(){
  std::vector<Command> commands={{4,0},{6,1},{9,0},{1,2},{5,0},{3,3},{7,4},{8,0}};
  auto schedule=replay_indices(commands);
  assert((schedule==std::vector<size_t>{1,3,5,6}));
  commands[1].value=99;
  assert(commands[schedule[0]].value==99);
  std::vector<int> old_values,new_values;
  for(auto& c:commands)if(c.kind==1||c.kind==2||c.kind==3||c.kind==6||c.kind==7)old_values.push_back(c.value);
  for(auto i:schedule)new_values.push_back(commands[i].value);
  assert(old_values==new_values);
}'''
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'test'
            subprocess.run(['g++','-std=c++17','-I'+str(header),'-x','c++','-','-o',str(target)], input=code, text=True, check=True)
            subprocess.run([str(target)], check=True)
