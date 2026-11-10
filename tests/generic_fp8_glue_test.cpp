// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>

#include <dlfcn.h>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace tpc_lib_api;

static void shape(Tensor& tensor, TensorDataType type, std::initializer_list<uint64_t> sizes) {
  tensor = {};
  tensor.geometry.dataType = type;
  tensor.geometry.dims = sizes.size();
  unsigned i = 0;
  for (auto size : sizes) {
    tensor.geometry.maxSizes[i] = tensor.geometry.minSizes[i] = size;
    ++i;
  }
}

static void check_axes(const Tensor& tensor, const TensorAccessPattern& access, unsigned rank) {
  if (access.allRequired) throw std::runtime_error("unexpected allRequired access");
  std::set<unsigned> axes;
  for (unsigned d = 0; d < tensor.geometry.dims; ++d) {
    auto axis = access.mapping[d].indexSpaceDim;
    if (axis >= rank || !axes.insert(axis).second)
      throw std::runtime_error("tensor dimensions share or exceed index-space axes");
  }
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("provider path required");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    using Instantiate = GlueCodeReturn (*)(HabanaKernelParams*, HabanaKernelInstantiation*);
    auto instantiate = reinterpret_cast<Instantiate>(dlsym(library, "InstantiateTpcKernel"));
    if (!instantiate || !dlsym(library, "GetSupportedDataLayouts"))
      throw std::runtime_error("provider entry point missing");
    for (uint64_t rows : {1u, 3u, 16u, 64u}) {
      Tensor inputs[5]{}, outputs[2]{};
      TensorAccessPattern in_access[5]{}, out_access[2]{};
      HabanaKernelParams params{};
      HabanaKernelInstantiation instance{};
      params.inputTensors = inputs;
      params.outputTensors = outputs;
      instance.inputTensorAccessPattern = in_access;
      instance.outputTensorAccessPattern = out_access;
      std::strcpy(params.guid.name, "gk_fp8_block_quant_bf16_v1");
      params.inputTensorNr = 1;
      params.outputTensorNr = 2;
      shape(inputs[0], DATA_BF16, {256, rows});
      shape(outputs[0], DATA_F8_143, {128, rows, 2});
      shape(outputs[1], DATA_F32, {1, rows, 2});
      if (instantiate(&params, &instance) != GLUE_INSUFFICIENT_ELF_BUFFER ||
          !instance.kernel.elfSize || instance.indexSpaceRank != 3 ||
          instance.indexSpaceGeometry[0] != 2 || instance.indexSpaceGeometry[1] != rows)
        throw std::runtime_error("FP8 quant geometry rejected");
      check_axes(inputs[0], in_access[0], instance.indexSpaceRank);
      for (unsigned i = 0; i < 2; ++i) check_axes(outputs[i], out_access[i], instance.indexSpaceRank);
      std::vector<unsigned char> quant_elf(instance.kernel.elfSize);
      instance.kernel.kernelElf = quant_elf.data();
      if (instantiate(&params, &instance) != GLUE_SUCCESS)
        throw std::runtime_error("FP8 quant ELF rejected");

      params = {};
      instance = {};
      std::memset(in_access, 0, sizeof(in_access));
      std::memset(out_access, 0, sizeof(out_access));
      std::strcpy(params.guid.name, "gk_fp8_block_reduce_bf16_v1");
      params.inputTensorNr = 4;
      params.outputTensorNr = 1;
      params.inputTensors = inputs;
      params.outputTensors = outputs;
      instance.inputTensorAccessPattern = in_access;
      instance.outputTensorAccessPattern = out_access;
      shape(inputs[0], DATA_F32, {512, rows, 2});
      shape(inputs[1], DATA_F32, {1, rows, 2});
      shape(inputs[2], DATA_F32, {2, 4});
      shape(inputs[3], DATA_F32, {512});
      shape(outputs[0], DATA_BF16, {512, rows});
      auto tile = rows <= 4 ? 1u : rows <= 16 ? 2u : 4u;
      if (instantiate(&params, &instance) != GLUE_INSUFFICIENT_ELF_BUFFER ||
          !instance.kernel.elfSize || instance.indexSpaceRank != 4 ||
          instance.indexSpaceGeometry[0] != 4 ||
          instance.indexSpaceGeometry[1] != (rows + tile - 1) / tile)
        throw std::runtime_error("FP8 reduction geometry rejected");
      for (unsigned i = 0; i < 4; ++i) check_axes(inputs[i], in_access[i], instance.indexSpaceRank);
      check_axes(outputs[0], out_access[0], instance.indexSpaceRank);
      std::vector<unsigned char> reduce_elf(instance.kernel.elfSize);
      instance.kernel.kernelElf = reduce_elf.data();
      if (instantiate(&params, &instance) != GLUE_SUCCESS)
        throw std::runtime_error("FP8 reduction ELF rejected");
      shape(inputs[2], DATA_F32, {2, 3});
      if (instantiate(&params, &instance) != GLUE_INCOMPATIBLE_INPUT_SIZE)
        throw std::runtime_error("invalid scale grid accepted");

      params = {};
      instance = {};
      std::memset(in_access, 0, sizeof(in_access));
      std::memset(out_access, 0, sizeof(out_access));
      std::strcpy(params.guid.name, "gk_fp8_token_quant_bf16_v1");
      params.inputTensorNr = 1;
      params.outputTensorNr = 2;
      params.inputTensors = inputs;
      params.outputTensors = outputs;
      instance.inputTensorAccessPattern = in_access;
      instance.outputTensorAccessPattern = out_access;
      shape(inputs[0], DATA_BF16, {256, rows});
      shape(outputs[0], DATA_F8_143, {256, rows});
      shape(outputs[1], DATA_F32, {1, rows});
      if (instantiate(&params, &instance) != GLUE_INSUFFICIENT_ELF_BUFFER ||
          !instance.kernel.elfSize || instance.indexSpaceRank != 2 ||
          instance.indexSpaceGeometry[0] != rows)
        throw std::runtime_error("token FP8 quant geometry rejected");
      check_axes(inputs[0], in_access[0], instance.indexSpaceRank);
      for (unsigned i = 0; i < 2; ++i) check_axes(outputs[i], out_access[i], instance.indexSpaceRank);

      params = {};
      instance = {};
      std::memset(in_access, 0, sizeof(in_access));
      std::memset(out_access, 0, sizeof(out_access));
      std::strcpy(params.guid.name, "gk_fp8_channel_finish_bf16_v1");
      params.inputTensorNr = 4;
      params.outputTensorNr = 1;
      params.inputTensors = inputs;
      params.outputTensors = outputs;
      instance.inputTensorAccessPattern = in_access;
      instance.outputTensorAccessPattern = out_access;
      shape(inputs[0], DATA_F32, {512, rows});
      shape(inputs[1], DATA_F32, {1, rows});
      shape(inputs[2], DATA_F32, {512});
      shape(inputs[3], DATA_F32, {512});
      shape(outputs[0], DATA_BF16, {512, rows});
      if (instantiate(&params, &instance) != GLUE_INSUFFICIENT_ELF_BUFFER ||
          !instance.kernel.elfSize || instance.indexSpaceRank != 3 ||
          instance.indexSpaceGeometry[0] != 4 || instance.indexSpaceGeometry[1] != rows)
        throw std::runtime_error("FP8 channel finish geometry rejected");
      for (unsigned i = 0; i < 4; ++i) check_axes(inputs[i], in_access[i], instance.indexSpaceRank);
      check_axes(outputs[0], out_access[0], instance.indexSpaceRank);

      auto check_multi = [&](const char* guid, std::initializer_list<Tensor> ins,
                             std::initializer_list<Tensor> outs,
                             unsigned expected_rank, uint64_t expected_dim0) {
        params = {};
        instance = {};
        std::memset(in_access, 0, sizeof(in_access));
        std::memset(out_access, 0, sizeof(out_access));
        std::strcpy(params.guid.name, guid);
        unsigned idx = 0, oidx = 0;
        for (const auto& t : ins) inputs[idx++] = t;
        for (const auto& t : outs) outputs[oidx++] = t;
        params.inputTensorNr = idx;
        params.outputTensorNr = oidx;
        params.inputTensors = inputs;
        params.outputTensors = outputs;
        instance.inputTensorAccessPattern = in_access;
        instance.outputTensorAccessPattern = out_access;
        if (instantiate(&params, &instance) != GLUE_INSUFFICIENT_ELF_BUFFER ||
            !instance.kernel.elfSize || instance.indexSpaceRank != expected_rank ||
            instance.indexSpaceGeometry[0] != expected_dim0)
          throw std::runtime_error(std::string("FP8 hybrid geometry rejected: ") + guid);
        for (unsigned i = 0; i < idx; ++i) check_axes(inputs[i], in_access[i], instance.indexSpaceRank);
        for (unsigned i = 0; i < oidx; ++i) check_axes(outputs[i], out_access[i], instance.indexSpaceRank);
        std::vector<unsigned char> elf(instance.kernel.elfSize);
        instance.kernel.kernelElf = elf.data();
        if (instantiate(&params, &instance) != GLUE_SUCCESS)
          throw std::runtime_error(std::string("FP8 hybrid ELF rejected: ") + guid);
      };

      Tensor t_in{}, t_rscale{}, t_a1k{}, t_a2k{}, t_w{}, t_ws{}, t_bscale{}, t_w1k{}, t_w2k{};
      Tensor t_c0{}, t_c1{}, t_bias{}, t_out{};
      shape(t_in, DATA_BF16, {256, rows});
      shape(t_rscale, DATA_F32, {1, rows});
      shape(t_a1k, DATA_BF16, {256, rows});
      shape(t_a2k, DATA_F8_143, {512, rows});
      shape(t_w, DATA_F8_143, {128, 512, 3});
      shape(t_ws, DATA_F32, {3, 4});
      shape(t_bscale, DATA_F32, {1, 4});
      shape(t_w1k, DATA_BF16, {384, 512});
      shape(t_w2k, DATA_F8_143, {768, 512});
      shape(t_c0, DATA_F32, {512, rows});
      shape(t_c1, DATA_F32, {512, rows});
      shape(t_bias, DATA_F32, {512});
      shape(t_out, DATA_BF16, {512, rows});

      check_multi("gk_fp8_act_row_scale_f32_v1", {t_in}, {t_rscale}, 2, (rows + 15) / 16);
      check_multi("gk_fp8_quant_act_hybrid_v1", {t_in, t_rscale}, {t_a1k, t_a2k}, 3, 1);
      check_multi("gk_fp8_dequant_weight_bf16_1k_v1", {t_w, t_ws}, {t_w1k}, 3, 3);
      check_multi("gk_fp8_weight_blk_scale_f32_v1", {t_ws}, {t_bscale}, 2, 1);
      check_multi("gk_fp8_dequant_weight_fp8_2k_v1", {t_w, t_ws, t_bscale}, {t_w2k}, 3, 3);
      check_multi("gk_fp8_hybrid_epilogue_bf16_v1", {t_c0, t_c1, t_rscale, t_bscale}, {t_out}, 3, 1);
      check_multi("gk_fp8_hybrid_epilogue_bias_bf16_v1", {t_c0, t_c1, t_rscale, t_bscale, t_bias}, {t_out}, 3, 1);
    }
    dlclose(library);
    std::cout << "FP8 Glue geometry, unique axes, ELF and invalid inputs passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
