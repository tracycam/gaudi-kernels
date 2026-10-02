#include "../../csrc/native_graph/ranges.hpp"
#include <cassert>
#include <limits>
int main(){
 using namespace gkg;
 Written w;w.add(0,16);w.add(32,16);assert(!w.contains(8,32));
 w.add(16,16);assert(w.contains(8,32));assert(!w.contains(0,49));
 w.add(8,8);assert(w.spans.size()==1);assert(!w.contains(48,1));
 assert(fits(16,16,32));assert(!fits(17,16,32));assert(!fits(0,0,32));
 assert(!valid_address(0,8));assert(!valid_address(std::numeric_limits<uint64_t>::max()-4,8));
}
