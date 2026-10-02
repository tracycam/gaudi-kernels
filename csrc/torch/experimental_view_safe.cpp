// Experimental, exact-build-pinned private function ABI. The pointee signature
// is taken from view_utils.h/.cpp and the installed symbol, not guessed from
// opaque object storage. The original public descriptor still executes nodes.
#include "experimental_view_safe.hpp"
#include "view_safe_runtime_pin.h"
#include <ATen/ATen.h>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <iomanip>
#include <link.h>
#include <sstream>

namespace gaudi_kernels::experimental {
namespace {
using HandleList = std::vector<at::Tensor> (*)(at::TensorList);
constexpr const char* kSymbol =
    "_ZN11habana_lazy17HbLazyTensorViews21HandleViewsTensorListEN3c108ArrayRefIN2at6TensorEEE";
struct LoadedId { void* base; std::string build_id; };
size_t align4(size_t n) { return (n + 3) & ~size_t(3); }
int loaded_build_id(dl_phdr_info* info, size_t, void* opaque) {
  auto& found = *static_cast<LoadedId*>(opaque);
  if (reinterpret_cast<void*>(info->dlpi_addr) != found.base) return 0;
  for (size_t i = 0; i < info->dlpi_phnum; ++i) {
    const auto& p = info->dlpi_phdr[i];
    if (p.p_type != PT_NOTE) continue;
    auto begin = reinterpret_cast<const unsigned char*>(info->dlpi_addr + p.p_vaddr);
    size_t offset = 0;
    while (offset + sizeof(ElfW(Nhdr)) <= p.p_memsz) {
      ElfW(Nhdr) note{};
      std::memcpy(&note, begin + offset, sizeof(note));
      offset += sizeof(note);
      const size_t names = align4(note.n_namesz), desc = align4(note.n_descsz);
      if (names > p.p_memsz - offset || desc > p.p_memsz - offset - names) break;
      if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
          std::memcmp(begin + offset, "GNU\0", 4) == 0) {
        std::ostringstream text;
        text << std::hex << std::setfill('0');
        for (size_t j = 0; j < note.n_descsz; ++j)
          text << std::setw(2) << unsigned(begin[offset + names + j]);
        found.build_id = text.str();
        return 1;
      }
      offset += names + desc;
    }
  }
  return 1;
}
void lazy_only() {
  const char* mode = std::getenv("PT_HPU_LAZY_MODE");
  TORCH_CHECK(mode && std::strcmp(mode, "1") == 0,
              "private view helper requires explicit PT_HPU_LAZY_MODE=1");
}
HandleList resolve() {
  lazy_only();
  static HandleList handler = [] {
    // Never load a missing plugin or resolve the eager plugin by global symbol
    // precedence. Keep the acquired reference alive with this experiment.
    void* library = dlopen(kViewSafeRuntimePath, RTLD_NOW | RTLD_NOLOAD);
    TORCH_CHECK(library, "pinned Lazy bridge must already be loaded: ", kViewSafeRuntimePath);
    dlerror();
    void* symbol = dlsym(library, kSymbol);
    const char* error = dlerror();
    TORCH_CHECK(!error && symbol, "pinned view-handler symbol unavailable");
    Dl_info info{};
    TORCH_CHECK(dladdr(symbol, &info) && info.dli_fname && info.dli_fbase,
                "cannot identify view-handler owner");
    TORCH_CHECK(std::filesystem::canonical(info.dli_fname) ==
                    std::filesystem::canonical(kViewSafeRuntimePath),
                "view-handler resolved to a different library");
    LoadedId id{info.dli_fbase, {}};
    dl_iterate_phdr(loaded_build_id, &id);
    TORCH_CHECK(id.build_id == kViewSafeRuntimeBuildId,
                "private bridge build-ID mismatch: ", id.build_id);
    // POSIX dlsym function-pointer conversion; no object-pointer dereference.
    return reinterpret_cast<HandleList>(symbol);
  }();
  return handler;
}
std::pair<c10::IValue, bool> normalize(const c10::IValue& value, HandleList handler) {
  if (value.isTensor()) {
    const auto& tensor = value.toTensor();
    if (!tensor.defined() || tensor.device().type() != at::kHPU) return {value, false};
    std::vector<at::Tensor> tensors{tensor};
    auto updated = handler(tensors);
    TORCH_CHECK(updated.size() == 1, "private view handler changed list length");
    return {updated[0], true};
  }
  if (value.isList()) {
    auto list = value.toList().copy();
    bool changed = false;
    for (size_t i = 0; i < list.size(); ++i) {
      auto item = normalize(list.get(i), handler);
      if (item.second) { list.set(i, std::move(item.first)); changed = true; }
    }
    if (changed) return {list, true};
  }
  return {value, false};
}
} // namespace
std::string view_safe_runtime_pin() {
  resolve(); // resolution/build-ID only; no Tensor, acquire, or device call
  return std::string("private_abi_lazy_only sha256=") + kViewSafeRuntimeSha256 +
      " build_id=" + kViewSafeRuntimeBuildId + " path=" + kViewSafeRuntimePath;
}
std::vector<c10::IValue> normalize_view_inputs(const std::vector<c10::IValue>& inputs) {
  auto handler = resolve();
  auto updated = inputs;
  for (auto& input : updated) input = normalize(input, handler).first;
  return updated;
}
std::vector<at::Tensor> execute_view_safe_symbols(
    habana::custom_op::UserCustomOpDescriptor descriptor,
    const std::vector<c10::IValue>& inputs) {
  auto updated = normalize_view_inputs(inputs);
  return descriptor.execute(updated);
}
} // namespace gaudi_kernels::experimental
