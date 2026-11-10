// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "generic_attention_glue.hpp"

extern "C" unsigned char _binary_fp8_block_quant_o_start, _binary_fp8_block_quant_o_end;
extern "C" unsigned char _binary_fp8_block_reduce_o_start, _binary_fp8_block_reduce_o_end;

namespace generic_fp8 {
using namespace tpc_lib_api;

constexpr const char* kQuant = "gk_fp8_block_quant_bf16_v1";
constexpr const char* kReduce = "gk_fp8_block_reduce_bf16_v1";

inline GlueCodeReturn instantiate_quant(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& x = input->inputTensors[0].geometry;
  const auto width = x.maxSizes[0], rows = x.maxSizes[1];
  if (!width || width > INT_MAX || width % 128 || !rows || rows > 4096 ||
      !generic_attention::shape(input->inputTensors[0], DATA_BF16, {width, rows}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_F8_143, {128, rows, width / 128}) ||
      !generic_attention::shape(input->outputTensors[1], DATA_F32, {1, rows, width / 128}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = width / 128;
  output->indexSpaceGeometry[1] = rows;
  output->indexSpaceGeometry[2] = 1;
  generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 0, 127, 128);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 1, 0, 1);
  for (unsigned i = 0; i < 2; ++i) {
    auto& access = output->outputTensorAccessPattern[i];
    generic_attention::mapped(access, 0, 2, i == 0 ? 127 : 0, 0);
    generic_attention::mapped(access, 1, 1, 0, 1);
    generic_attention::mapped(access, 2, 0, 0, 1);
  }
  output->kernel.paramsNr = 0;
  return generic_attention::binary(output, &_binary_fp8_block_quant_o_start, &_binary_fp8_block_quant_o_end);
}

inline GlueCodeReturn instantiate_reduce(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& p = input->inputTensors[0].geometry;
  const auto width = p.maxSizes[0], rows = p.maxSizes[1], groups = p.maxSizes[2];
  if (!width || width > INT_MAX - 128 || width % 128 || !rows || rows > 4096 ||
      !groups || groups > INT_MAX ||
      !generic_attention::shape(input->inputTensors[0], DATA_F32, {width, rows, groups}) ||
      !generic_attention::shape(input->inputTensors[1], DATA_F32, {1, rows, groups}) ||
      !generic_attention::shape(input->inputTensors[2], DATA_F32, {groups, width / 128}) ||
      !generic_attention::shape(input->inputTensors[3], DATA_F32, {width}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_BF16, {width, rows}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  const unsigned row_tile = rows <= 4 ? 1 : rows <= 16 ? 2 : 4;
  output->indexSpaceRank = 4;
  output->indexSpaceGeometry[0] = width / 128;
  output->indexSpaceGeometry[1] = (rows + row_tile - 1) / row_tile;
  output->indexSpaceGeometry[2] = 1;
  output->indexSpaceGeometry[3] = 1;
  auto& partial = output->inputTensorAccessPattern[0];
  generic_attention::mapped(partial, 0, 0, 127, 128);
  generic_attention::mapped(partial, 1, 1, row_tile - 1, row_tile);
  generic_attention::mapped(partial, 2, 2, groups - 1, 0);
  auto& act_scale = output->inputTensorAccessPattern[1];
  generic_attention::mapped(act_scale, 0, 3, 0, 0);
  generic_attention::mapped(act_scale, 1, 1, row_tile - 1, row_tile);
  generic_attention::mapped(act_scale, 2, 2, groups - 1, 0);
  auto& weight_scale = output->inputTensorAccessPattern[2];
  generic_attention::mapped(weight_scale, 0, 2, groups - 1, 0);
  generic_attention::mapped(weight_scale, 1, 0, 0, 1);
  generic_attention::mapped(output->inputTensorAccessPattern[3], 0, 0, 127, 128);
  auto& result = output->outputTensorAccessPattern[0];
  generic_attention::mapped(result, 0, 0, 127, 128);
  generic_attention::mapped(result, 1, 1, row_tile - 1, row_tile);
  output->kernel.paramsNr = 0;
  return generic_attention::binary(output, &_binary_fp8_block_reduce_o_start, &_binary_fp8_block_reduce_o_end);
}
}  // namespace generic_fp8
