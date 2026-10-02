#pragma once
#include <cstddef>
#include <vector>

template<class Commands>
std::vector<size_t> replay_indices(const Commands& commands) {
  std::vector<size_t> result;
  result.reserve(commands.size());
  for(size_t i=0;i<commands.size();++i){
    const int kind=commands[i].kind;
    if(kind==1||kind==2||kind==3||kind==6||kind==7)result.push_back(i);
  }
  return result;
}
