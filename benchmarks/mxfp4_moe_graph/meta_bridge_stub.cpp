// OFFLINE META ONLY. This must never be linked into the HPU bridge build.
#include <hpu_custom_op_pt2.h>
#include <stdexcept>
#include <unordered_map>
namespace habana::custom_op {
static auto& registry(){static std::unordered_map<std::string,UserCustomOpDescriptor> r;return r;}
void registerUserCustomOp(const std::string&s,const std::string&g,OutputMetaFn m,FillParamsFn p){registry().emplace(s,UserCustomOpDescriptor(s,g,m,p));}
const UserCustomOpDescriptor&UserCustomOpDescriptor::getUserCustomOpDescriptor(const std::string&s){return registry().at(s);}
std::vector<at::Tensor>UserCustomOpDescriptor::execute(const std::vector<c10::IValue>&){throw std::runtime_error("OFFLINE META stub cannot execute HPU");}
}
