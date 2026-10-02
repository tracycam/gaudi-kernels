// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
namespace gkg {
inline bool fits(uint64_t offset,uint64_t bytes,uint64_t capacity){return bytes&&offset<=capacity&&bytes<=capacity-offset;}
inline bool valid_address(uint64_t address,uint64_t bytes){return address&&bytes&&bytes<=std::numeric_limits<uint64_t>::max()-address;}
struct Written {
 std::vector<std::pair<uint64_t,uint64_t>> spans;
 bool contains(uint64_t offset,uint64_t bytes)const {
  if(!bytes||bytes>std::numeric_limits<uint64_t>::max()-offset)return false;
  for(auto s:spans)if(s.first<=offset&&s.second>=offset+bytes)return true;
  return false;
 }
 void add(uint64_t offset,uint64_t bytes){
  spans.emplace_back(offset,offset+bytes);std::sort(spans.begin(),spans.end());
  std::vector<std::pair<uint64_t,uint64_t>> merged;
  for(auto s:spans){if(merged.empty()||merged.back().second<s.first)merged.push_back(s);else merged.back().second=std::max(merged.back().second,s.second);}
  spans=std::move(merged);
 }
};
}
