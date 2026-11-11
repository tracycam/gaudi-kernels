// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "generic_attention_glue.hpp"

extern "C" unsigned char _binary_fp8_block_quant_o_start, _binary_fp8_block_quant_o_end;
extern "C" unsigned char _binary_fp8_block_reduce_o_start, _binary_fp8_block_reduce_o_end;
extern "C" unsigned char _binary_fp8_token_quant_o_start, _binary_fp8_token_quant_o_end;
extern "C" unsigned char _binary_fp8_channel_finish_o_start, _binary_fp8_channel_finish_o_end;
extern "C" unsigned char _binary_fp8_quant_act_1k_o_start, _binary_fp8_quant_act_1k_o_end;
extern "C" unsigned char _binary_fp8_quant_act_k32_o_start, _binary_fp8_quant_act_k32_o_end;
extern "C" unsigned char _binary_fp8_dequant_weight_1k_o_start, _binary_fp8_dequant_weight_1k_o_end;
extern "C" unsigned char _binary_fp8_dequant_weight_k32_o_start, _binary_fp8_dequant_weight_k32_o_end;
extern "C" unsigned char _binary_fp8_epilogue_k32_o_start, _binary_fp8_epilogue_k32_o_end;
extern "C" unsigned char _binary_fp8_epilogue_k32_bias_o_start, _binary_fp8_epilogue_k32_bias_o_end;

namespace generic_fp8 {
using namespace tpc_lib_api;

constexpr const char* kQuant = "gk_fp8_block_quant_bf16_v1";
constexpr const char* kReduce = "gk_fp8_block_reduce_bf16_v1";
constexpr const char* kTokenQuant = "gk_fp8_token_quant_bf16_v1";
constexpr const char* kChannelFinish = "gk_fp8_channel_finish_bf16_v1";
constexpr const char* kQuantAct1k = "gk_fp8_quant_act_bf16_1k_v1";
constexpr const char* kQuantActK32 = "gk_fp8_quant_act_bf16_k32_v1";
constexpr const char* kDequantWeight1k = "gk_fp8_dequant_weight_bf16_1k_v1";
constexpr const char* kDequantWeightK32 = "gk_fp8_dequant_weight_bf16_k32_v1";
constexpr const char* kEpilogueK32 = "gk_fp8_epilogue_k32_bf16_v1";
constexpr const char* kEpilogueK32Bias = "gk_fp8_epilogue_k32_bias_bf16_v1";

inline GlueCodeReturn instantiate_token_quant(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& x = input->inputTensors[0].geometry;
  const auto width = x.maxSizes[0], rows = x.maxSizes[1];
  if (!width || width > INT_MAX || width % 128 || !rows || rows > 4096 ||
      !generic_attention::shape(input->inputTensors[0], DATA_BF16, {width, rows}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_F8_143, {width, rows}) ||
      !generic_attention::shape(input->outputTensors[1], DATA_F32, {1, rows}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = rows;
  output->indexSpaceGeometry[1] = 1;
  generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 1, width - 1, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 0, 0, 1);
  for (unsigned i = 0; i < 2; ++i) {
    generic_attention::mapped(output->outputTensorAccessPattern[i], 0, 1, i == 0 ? width - 1 : 0, 0);
    generic_attention::mapped(output->outputTensorAccessPattern[i], 1, 0, 0, 1);
  }
  output->kernel.paramsNr = 0;
  return generic_attention::binary(output, &_binary_fp8_token_quant_o_start, &_binary_fp8_token_quant_o_end);
}

inline GlueCodeReturn instantiate_channel_finish(HabanaKernelParams* input, HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& p = input->inputTensors[0].geometry;
  const auto width = p.maxSizes[0], rows = p.maxSizes[1];
  if (!width || width > INT_MAX - 128 || width % 128 || !rows || rows > 4096 ||
      !generic_attention::shape(input->inputTensors[0], DATA_F32, {width, rows}) ||
      !generic_attention::shape(input->inputTensors[1], DATA_F32, {1, rows}) ||
      !generic_attention::shape(input->inputTensors[2], DATA_F32, {width}) ||
      !generic_attention::shape(input->inputTensors[3], DATA_F32, {width}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_BF16, {width, rows}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = width / 128;
  output->indexSpaceGeometry[1] = rows;
  output->indexSpaceGeometry[2] = 1;
  generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 0, 127, 128);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 1, 0, 1);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 0, 2, 0, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 1, 1, 0, 1);
  for (unsigned i = 2; i < 4; ++i)
    generic_attention::mapped(output->inputTensorAccessPattern[i], 0, 0, 127, 128);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 0, 127, 128);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 1, 0, 1);
  output->kernel.paramsNr = 0;
  return generic_attention::binary(output, &_binary_fp8_channel_finish_o_start, &_binary_fp8_channel_finish_o_end);
}

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

inline GlueCodeReturn parse_tile_tag(const HabanaKernelParams* input, int32_t& tile_tag) {
  tile_tag = 0;
  if (input->nodeParams.nodeParamsSize == 0) return GLUE_SUCCESS;
  if (input->nodeParams.nodeParamsSize != sizeof(int32_t) || !input->nodeParams.nodeParams)
    return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  std::memcpy(&tile_tag, input->nodeParams.nodeParams, sizeof(int32_t));
  return tile_tag >= 0 ? GLUE_SUCCESS : GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
}

inline GlueCodeReturn instantiate_quant_act_bf16(HabanaKernelParams* input,
                                                 HabanaKernelInstantiation* output,
                                                 bool k32) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t tile_tag = 0;
  if (auto rc = parse_tile_tag(input, tile_tag); rc != GLUE_SUCCESS) return rc;
  const auto& x = input->inputTensors[0].geometry;
  const auto width = x.maxSizes[0], rows = x.maxSizes[1];
  if (!width || width > INT_MAX || width % 128 || !rows || rows > 4096 ||
      !generic_attention::shape(input->inputTensors[0], DATA_BF16, {width, rows}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_BF16, {width, rows}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  constexpr unsigned kGroupTile = 4;
  constexpr unsigned kColTile = 128 * kGroupTile;
  constexpr unsigned kRowTile = 8;
  const auto groups = width / 128;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = (groups + kGroupTile - 1) / kGroupTile;
  output->indexSpaceGeometry[1] = (rows + kRowTile - 1) / kRowTile;
  output->indexSpaceGeometry[2] = 1;
  generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 0, kColTile - 1, kColTile);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 1, kRowTile - 1, kRowTile);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 0, kColTile - 1, kColTile);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 1, kRowTile - 1, kRowTile);
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &tile_tag, sizeof(tile_tag));
  return k32 ? generic_attention::binary(output, &_binary_fp8_quant_act_k32_o_start,
                                         &_binary_fp8_quant_act_k32_o_end)
             : generic_attention::binary(output, &_binary_fp8_quant_act_1k_o_start,
                                         &_binary_fp8_quant_act_1k_o_end);
}

inline GlueCodeReturn instantiate_dequant_weight_bf16(HabanaKernelParams* input,
                                                      HabanaKernelInstantiation* output,
                                                      bool k32) {
  if (input->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t tile_tag = 0;
  if (auto rc = parse_tile_tag(input, tile_tag); rc != GLUE_SUCCESS) return rc;
  const auto& w = input->inputTensors[0].geometry;
  const auto width = w.maxSizes[1], groups = w.maxSizes[2];
  if (!width || width > INT_MAX - 128 || width % 128 || !groups || groups > INT_MAX / 128 ||
      !generic_attention::shape(input->inputTensors[0], DATA_F8_143, {128, width, groups}) ||
      !generic_attention::shape(input->inputTensors[1], DATA_F32, {groups, width / 128}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_BF16, {groups * 128, width}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = groups;
  output->indexSpaceGeometry[1] = width / 128;
  output->indexSpaceGeometry[2] = 1;
  generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 2, 127, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 1, 127, 128);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 2, 0, 0, 1);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 0, 0, 0, 1);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 1, 1, 0, 1);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 0, 127, 128);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 1, 127, 128);
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &tile_tag, sizeof(tile_tag));
  return k32 ? generic_attention::binary(output, &_binary_fp8_dequant_weight_k32_o_start,
                                         &_binary_fp8_dequant_weight_k32_o_end)
             : generic_attention::binary(output, &_binary_fp8_dequant_weight_1k_o_start,
                                         &_binary_fp8_dequant_weight_1k_o_end);
}

inline GlueCodeReturn instantiate_epilogue_k32(HabanaKernelParams* input,
                                               HabanaKernelInstantiation* output,
                                               bool has_bias) {
  if (input->inputTensorNr != (has_bias ? 3u : 2u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& p = input->inputTensors[0].geometry;
  const auto width = p.maxSizes[0], rows = p.maxSizes[1];
  if (!width || width > INT_MAX - 128 || width % 128 || !rows || rows > 4096 ||
      !generic_attention::shape(input->inputTensors[0], DATA_F32, {width, rows}) ||
      !generic_attention::shape(input->inputTensors[1], DATA_F32, {width, rows}) ||
      (has_bias && !generic_attention::shape(input->inputTensors[2], DATA_F32, {width})) ||
      !generic_attention::shape(input->outputTensors[0], DATA_BF16, {width, rows}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  constexpr unsigned kBlockTile = 4;
  constexpr unsigned kColTile = 128 * kBlockTile;
  constexpr unsigned kRowTile = 8;
  const auto blocks = width / 128;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = (blocks + kBlockTile - 1) / kBlockTile;
  output->indexSpaceGeometry[1] = (rows + kRowTile - 1) / kRowTile;
  output->indexSpaceGeometry[2] = 1;
  for (unsigned i = 0; i < 2; ++i) {
    generic_attention::mapped(output->inputTensorAccessPattern[i], 0, 0, kColTile - 1, kColTile);
    generic_attention::mapped(output->inputTensorAccessPattern[i], 1, 1, kRowTile - 1, kRowTile);
  }
  if (has_bias)
    generic_attention::mapped(output->inputTensorAccessPattern[2], 0, 0, kColTile - 1, kColTile);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 0, kColTile - 1, kColTile);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 1, kRowTile - 1, kRowTile);
  output->kernel.paramsNr = 0;
  return has_bias
             ? generic_attention::binary(output, &_binary_fp8_epilogue_k32_bias_o_start,
                                         &_binary_fp8_epilogue_k32_bias_o_end)
             : generic_attention::binary(output, &_binary_fp8_epilogue_k32_o_start,
                                         &_binary_fp8_epilogue_k32_o_end);
}
}  // namespace generic_fp8
