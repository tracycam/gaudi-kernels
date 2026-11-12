// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "generic_attention_glue.hpp"

extern "C" unsigned char _binary_fp8_block_quant_o_start, _binary_fp8_block_quant_o_end;
extern "C" unsigned char _binary_fp8_block_reduce_o_start, _binary_fp8_block_reduce_o_end;
extern "C" unsigned char _binary_fp8_token_quant_o_start, _binary_fp8_token_quant_o_end;
extern "C" unsigned char _binary_fp8_channel_finish_o_start, _binary_fp8_channel_finish_o_end;
extern "C" unsigned char _binary_fp8_act_row_scale_o_start, _binary_fp8_act_row_scale_o_end;
extern "C" unsigned char _binary_fp8_weight_blk_scale_o_start, _binary_fp8_weight_blk_scale_o_end;
extern "C" unsigned char _binary_fp8_quant_act_1k_o_start, _binary_fp8_quant_act_1k_o_end;
extern "C" unsigned char _binary_fp8_dequant_weight_1k_o_start, _binary_fp8_dequant_weight_1k_o_end;
extern "C" unsigned char _binary_fp8_epilogue_o_start, _binary_fp8_epilogue_o_end;
extern "C" unsigned char _binary_fp8_epilogue_bias_o_start, _binary_fp8_epilogue_bias_o_end;

namespace generic_fp8 {
using namespace tpc_lib_api;

constexpr const char* kQuant = "gk_fp8_block_quant_bf16_v1";
constexpr const char* kReduce = "gk_fp8_block_reduce_bf16_v1";
constexpr const char* kTokenQuant = "gk_fp8_token_quant_bf16_v1";
constexpr const char* kChannelFinish = "gk_fp8_channel_finish_bf16_v1";
constexpr const char* kActRowScale = "gk_fp8_act_row_scale_f32_v1";
constexpr const char* kWeightBlkScale = "gk_fp8_weight_blk_scale_f32_v1";
constexpr const char* kQuantAct1k = "gk_fp8_quant_act_fp16_1k_v1";
constexpr const char* kDequantWeight1k = "gk_fp8_dequant_weight_fp16_1k_v1";
constexpr const char* kEpilogue = "gk_fp8_epilogue_bf16_v1";
constexpr const char* kEpilogueBias = "gk_fp8_epilogue_bias_bf16_v1";

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

inline GlueCodeReturn parse_quant_act_params(const HabanaKernelParams* input,
                                             int32_t& col_offset, int32_t& tile_tag) {
  col_offset = 0;
  tile_tag = 0;
  if (input->nodeParams.nodeParamsSize == 0) return GLUE_SUCCESS;
  if (!input->nodeParams.nodeParams) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  if (input->nodeParams.nodeParamsSize == sizeof(int32_t)) {
    std::memcpy(&tile_tag, input->nodeParams.nodeParams, sizeof(int32_t));
    return tile_tag >= 0 ? GLUE_SUCCESS : GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  if (input->nodeParams.nodeParamsSize == 2 * sizeof(int32_t)) {
    int32_t raw[2]{};
    std::memcpy(raw, input->nodeParams.nodeParams, sizeof(raw));
    col_offset = raw[0];
    tile_tag = raw[1];
    return (col_offset >= 0 && col_offset % 128 == 0 && tile_tag >= 0)
               ? GLUE_SUCCESS
               : GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
}

inline GlueCodeReturn instantiate_act_row_scale(HabanaKernelParams* input,
                                                HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t tile_tag = 0;
  if (auto rc = parse_tile_tag(input, tile_tag); rc != GLUE_SUCCESS) return rc;
  const auto& x = input->inputTensors[0].geometry;
  const auto width = x.maxSizes[0], rows = x.maxSizes[1];
  if (!width || width > INT_MAX || width % 128 || !rows || rows > 4096 ||
      !generic_attention::shape(input->inputTensors[0], DATA_BF16, {width, rows}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_F32, {1, rows}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  constexpr unsigned kRowTile = 16;
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = (rows + kRowTile - 1) / kRowTile;
  output->indexSpaceGeometry[1] = 1;
  generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 1, width - 1, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 0, kRowTile - 1, kRowTile);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 1, 0, 0);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 0, kRowTile - 1, kRowTile);
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &tile_tag, sizeof(tile_tag));
  return generic_attention::binary(output, &_binary_fp8_act_row_scale_o_start,
                                   &_binary_fp8_act_row_scale_o_end);
}

inline GlueCodeReturn instantiate_weight_blk_scale(HabanaKernelParams* input,
                                                   HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t tile_tag = 0;
  if (auto rc = parse_tile_tag(input, tile_tag); rc != GLUE_SUCCESS) return rc;
  const auto& s = input->inputTensors[0].geometry;
  const auto groups = s.maxSizes[0], blocks = s.maxSizes[1];
  if (!groups || groups > INT_MAX || !blocks || blocks > INT_MAX ||
      !generic_attention::shape(input->inputTensors[0], DATA_F32, {groups, blocks}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_F32, {1, blocks}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  constexpr unsigned kBlockTile = 16;
  output->indexSpaceRank = 2;
  output->indexSpaceGeometry[0] = (blocks + kBlockTile - 1) / kBlockTile;
  output->indexSpaceGeometry[1] = 1;
  generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 1, groups - 1, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 0, kBlockTile - 1, kBlockTile);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 1, 0, 0);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 0, kBlockTile - 1, kBlockTile);
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &tile_tag, sizeof(tile_tag));
  return generic_attention::binary(output, &_binary_fp8_weight_blk_scale_o_start,
                                   &_binary_fp8_weight_blk_scale_o_end);
}

inline GlueCodeReturn instantiate_quant_act_1k(HabanaKernelParams* input,
                                               HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t col_offset = 0, tile_tag = 0;
  if (auto rc = parse_quant_act_params(input, col_offset, tile_tag); rc != GLUE_SUCCESS)
    return rc;
  const auto& x = input->inputTensors[0].geometry;
  const auto& y = input->outputTensors[0].geometry;
  const auto in_width = x.maxSizes[0], rows = x.maxSizes[1];
  const auto out_width = y.maxSizes[0];
  const auto out_groups = out_width / 128;
  if (!in_width || in_width > INT_MAX - 512 || in_width % 128 ||
      !out_width || out_width > in_width || out_width % 128 ||
      static_cast<uint64_t>(col_offset) + out_width > in_width ||
      !rows || rows > 4096 ||
      !generic_attention::shape(input->inputTensors[0], DATA_BF16, {in_width, rows}) ||
      !generic_attention::shape(input->inputTensors[1], DATA_F32, {1, rows}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_F16, {out_width, rows}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = (out_groups + 3) / 4;
  output->indexSpaceGeometry[1] = (rows + 15) / 16;
  output->indexSpaceGeometry[2] = 1;
  auto& in_access = output->inputTensorAccessPattern[0];
  in_access.allRequired = 0;
  in_access.mapping[0].indexSpaceDim = 0;
  in_access.mapping[0].a = 512;
  in_access.mapping[0].start_b = col_offset;
  in_access.mapping[0].end_b = col_offset + 511;
  generic_attention::mapped(in_access, 1, 1, 15, 16);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 0, 2, 0, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 1, 1, 15, 16);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 0, 511, 512);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 1, 15, 16);
  output->kernel.paramsNr = 2;
  const int32_t scalars[2]{col_offset, tile_tag};
  std::memcpy(output->kernel.scalarParams, scalars, sizeof(scalars));
  return generic_attention::binary(output, &_binary_fp8_quant_act_1k_o_start,
                                   &_binary_fp8_quant_act_1k_o_end);
}

inline GlueCodeReturn instantiate_dequant_weight_1k(HabanaKernelParams* input,
                                                    HabanaKernelInstantiation* output) {
  if (input->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  int32_t tile_tag = 0;
  if (auto rc = parse_tile_tag(input, tile_tag); rc != GLUE_SUCCESS) return rc;
  const auto& w = input->inputTensors[0].geometry;
  const auto width = w.maxSizes[1], groups = w.maxSizes[2];
  const auto blocks = width / 128;
  const auto out_w = groups * 128;
  if (!width || width > INT_MAX - 128 || width % 128 || !groups || out_w > INT_MAX ||
      !generic_attention::shape(input->inputTensors[0], DATA_F8_143, {128, width, groups}) ||
      !generic_attention::shape(input->inputTensors[1], DATA_F32, {groups, blocks}) ||
      !generic_attention::shape(input->inputTensors[2], DATA_F32, {1, blocks}) ||
      !generic_attention::shape(input->outputTensors[0], DATA_F16, {out_w, width}))
    return GLUE_INCOMPATIBLE_INPUT_SIZE;
  output->indexSpaceRank = 3;
  output->indexSpaceGeometry[0] = groups;
  output->indexSpaceGeometry[1] = blocks;
  output->indexSpaceGeometry[2] = 1;
  generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 2, 127, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 1, 127, 128);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 2, 0, 0, 1);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 0, 0, 0, 1);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 1, 1, 0, 1);
  generic_attention::mapped(output->inputTensorAccessPattern[2], 0, 2, 0, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[2], 1, 1, 0, 1);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 0, 127, 128);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 1, 127, 128);
  output->kernel.paramsNr = 1;
  std::memcpy(output->kernel.scalarParams, &tile_tag, sizeof(tile_tag));
  return generic_attention::binary(output, &_binary_fp8_dequant_weight_1k_o_start,
                                   &_binary_fp8_dequant_weight_1k_o_end);
}

inline GlueCodeReturn instantiate_epilogue(HabanaKernelParams* input,
                                           HabanaKernelInstantiation* output,
                                           bool has_bias) {
  if (input->inputTensorNr != (has_bias ? 4u : 3u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
  if (input->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  if (input->nodeParams.nodeParamsSize) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  const auto& p = input->inputTensors[0].geometry;
  const auto width = p.maxSizes[0], rows = p.maxSizes[1];
  if (!width || width > INT_MAX - 128 || width % 128 || !rows || rows > 4096 ||
      !generic_attention::shape(input->inputTensors[0], DATA_F32, {width, rows}) ||
      !generic_attention::shape(input->inputTensors[1], DATA_F32, {1, rows}) ||
      !generic_attention::shape(input->inputTensors[2], DATA_F32, {1, width / 128}) ||
      (has_bias && !generic_attention::shape(input->inputTensors[3], DATA_F32, {width})) ||
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
  generic_attention::mapped(output->inputTensorAccessPattern[0], 0, 0, kColTile - 1, kColTile);
  generic_attention::mapped(output->inputTensorAccessPattern[0], 1, 1, kRowTile - 1, kRowTile);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 0, 2, 0, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[1], 1, 1, kRowTile - 1, kRowTile);
  generic_attention::mapped(output->inputTensorAccessPattern[2], 0, 2, 0, 0);
  generic_attention::mapped(output->inputTensorAccessPattern[2], 1, 0, kBlockTile - 1, kBlockTile);
  if (has_bias)
    generic_attention::mapped(output->inputTensorAccessPattern[3], 0, 0, kColTile - 1, kColTile);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 0, 0, kColTile - 1, kColTile);
  generic_attention::mapped(output->outputTensorAccessPattern[0], 1, 1, kRowTile - 1, kRowTile);
  output->kernel.paramsNr = 0;
  return has_bias
             ? generic_attention::binary(output, &_binary_fp8_epilogue_bias_o_start,
                                         &_binary_fp8_epilogue_bias_o_end)
             : generic_attention::binary(output, &_binary_fp8_epilogue_o_start,
                                         &_binary_fp8_epilogue_o_end);
}
}  // namespace generic_fp8
