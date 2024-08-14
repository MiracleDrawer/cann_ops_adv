/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "fallback_comm.h"
#include "fallback.h"
#include "error/ops_error.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t QUERY_INDEX = 0;
static const size_t KEY_INDEX = 1;
static const size_t VALUE_INDEX = 2;
static const size_t PSE_SHIFT_INDEX = 3;
static const size_t ATTEN_MASK_INDEX = 4;
static const size_t ACTUAL_SEQ_Q_INDEX = 5;
static const size_t ACTUAL_SEQ_KV_INDEX = 6;
static const size_t DEQUANT_SCALE1_INDEX = 7;
static const size_t QUANT_SCALE1_INDEX = 8;
static const size_t DEQUANT_SCALE2_INDEX = 9;
static const size_t QUANT_SCALE2_INDEX = 10;
static const size_t QUANT_OFFSET2_INDEX = 11;
static const size_t ANTIQUANT_SCALE_INDEX = 12;
static const size_t ANTIQUANT_OFFSET_INDEX = 13;
static const size_t BLOCK_TABLE_INDEX = 14;
static const size_t QUERY_PADDING_INDEX = 15;
static const size_t KV_PADDING_INDEX = 16;
static const size_t KEY_ANTIQUANT_SCALE_INDEX = 17;
static const size_t KEY_ANTIQUANT_OFFSET_INDEX = 18;
static const size_t VALUE_ANTIQUANT_SCALE_INDEX = 19;
static const size_t VALUE_ANTIQUANT_OFFSET_INDEX = 20;
static const size_t KEY_SHARED_PREFIX_INDEX = 21;
static const size_t VALUE_SHARED_PREFIX_INDEX = 22;
static const size_t ACTUAL_SHARED_PREFIX_LEN_INDEX = 23;

static const size_t ATTR_N_INDEX = 0;
static const size_t ATTR_SCALE_INDEX = 1;
static const size_t ATTR_PRE_TOKEN_INDEX = 2;
static const size_t ATTR_NEXT_TOKEN_INDEX = 3;
static const size_t ATTR_INPUT_LAYOUT_INDEX = 4;
static const size_t ATTR_NUM_KV_HEADS_INDEX = 5;
static const size_t ATTR_SPARSE_MODE_INDEX = 6;
static const size_t ATTR_INNER_PRECISE_INDEX = 7;
static const size_t ATTR_BLOCK_SIZE_INDEX = 8;
static const size_t ATTR_ANTIQUANT_MODE_INDEX = 9;
static const size_t ATTR_SOFTMAX_LSE_FLAG_INDEX = 10;
static const size_t ATTR_KEY_ANTIQUANT_MODE_INDEX = 11;
static const size_t ATTR_VALUE_ANTIQUANT_MODE_INDEX = 12;

static const size_t ATTENTION_OUT_INDEX = 0;
static const size_t SOFTMAX_LSE_INDEX = 1;

graphStatus FusedInferHostExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OPS_ERR_IF(host_api_ctx == nullptr,
    OPS_LOG_E("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

  auto query = host_api_ctx->GetInputTensor(QUERY_INDEX);
  OPS_ERR_IF(query == nullptr,
    OPS_LOG_E("aclnnfallback", "query is null"), return GRAPH_FAILED);

  auto key = host_api_ctx->GetDynamicInputTensor(KEY_INDEX, 0);
  OPS_ERR_IF(key == nullptr,
    OPS_LOG_E("aclnnfallback", "key is null"), return GRAPH_FAILED);

  auto value = host_api_ctx->GetDynamicInputTensor(VALUE_INDEX, 0);
  OPS_ERR_IF(value == nullptr,
    OPS_LOG_E("aclnnfallback", "value is null"), return GRAPH_FAILED);

  auto output = host_api_ctx->GetOutputTensor(ATTENTION_OUT_INDEX);
  OPS_ERR_IF(output == nullptr,
    OPS_LOG_E("aclnnfallback", "output is null"), return GRAPH_FAILED);

  auto softmaxLse = host_api_ctx->GetOutputTensor(SOFTMAX_LSE_INDEX);
  OPS_ERR_IF(softmaxLse == nullptr,
    OPS_LOG_E("aclnnfallback", "softmaxLse is null"), return GRAPH_FAILED);

  std::vector<const gert::Tensor*> ge_tenserListKey;
  ge_tenserListKey.push_back(key);

  std::vector<const gert::Tensor*> ge_tenserListValue;
  ge_tenserListValue.push_back(value);

  auto pseShiftGe = host_api_ctx->GetOptionalInputTensor(PSE_SHIFT_INDEX);
  auto attenMaskGe = host_api_ctx->GetOptionalInputTensor(ATTEN_MASK_INDEX);
  auto actualSeqLengthsGe = host_api_ctx->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
  auto actualSeqLengthsGeKv = host_api_ctx->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
  auto deqScale1 = host_api_ctx->GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
  auto quantScale1 = host_api_ctx->GetOptionalInputTensor(QUANT_SCALE1_INDEX);
  auto deqScale2 = host_api_ctx->GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
  auto quantScale2 = host_api_ctx->GetOptionalInputTensor(QUANT_SCALE2_INDEX);
  auto quantOffset2 = host_api_ctx->GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
  auto antiquantScaleGe = host_api_ctx->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
  auto antiquantOffsetGe = host_api_ctx->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
  auto blocktableGe = host_api_ctx->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
  auto queryPaddingGe = host_api_ctx->GetOptionalInputTensor(QUERY_PADDING_INDEX);
  auto kvPaddingGe = host_api_ctx->GetOptionalInputTensor(KV_PADDING_INDEX);
  auto keyAntiquantScaleGe = host_api_ctx->GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
  auto keyAntiquantOffsetGe = host_api_ctx->GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
  auto valueAntiquantScaleGe = host_api_ctx->GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
  auto valueAntiquantOffsetGe = host_api_ctx->GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
  auto keySharedPrefixGe = host_api_ctx->GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
  auto valueSharedPrefixGe = host_api_ctx->GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
  auto actualSharedPrefixLenGe = host_api_ctx->GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);

  std::vector<int64_t> actSeqArray;
  if (actualSeqLengthsGe != nullptr) {
    const int64_t* actSeqData = actualSeqLengthsGe->GetData<int64_t>();
    const size_t len = static_cast<size_t>(actualSeqLengthsGe->GetShapeSize());
    for (size_t i = 0; i < len; i++) {
      actSeqArray.push_back(actSeqData[i]);
    }
  }

  std::vector<int64_t> actSeqArrayKv;
  if (actualSeqLengthsGeKv != nullptr) {
    const int64_t* actSeqData = actualSeqLengthsGeKv->GetData<int64_t>();
    const size_t len = static_cast<size_t>(actualSeqLengthsGeKv->GetShapeSize());
    for (size_t i = 0; i < len; i++) {
      actSeqArrayKv.push_back(actSeqData[i]);
    }
  }

  std::vector<int64_t> actSeqSharedPrefix;
  if (actualSharedPrefixLenGe != nullptr) {
    const int64_t* actSeqData = actualSharedPrefixLenGe->GetData<int64_t>();
    const size_t len = static_cast<size_t>(actualSharedPrefixLenGe->GetShapeSize());
    for (size_t i = 0; i < len; i++) {
      actSeqSharedPrefix.push_back(actSeqData[i]);
    }
  }

  auto attrs = host_api_ctx->GetAttrs();
  const uint32_t* get_num_heads = attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
  const float* scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
  const uint32_t* get_pre_tokens = attrs->GetAttrPointer<uint32_t>(ATTR_PRE_TOKEN_INDEX);
  const uint32_t* get_next_tokens = attrs->GetAttrPointer<uint32_t>(ATTR_NEXT_TOKEN_INDEX);
  const char* layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
  const uint32_t* get_kvHeadNum = attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
  const uint32_t* get_sparseMode = attrs->GetAttrPointer<uint32_t>(ATTR_SPARSE_MODE_INDEX);
  const uint32_t* get_innerPrecise = attrs->GetAttrPointer<uint32_t>(ATTR_INNER_PRECISE_INDEX);
  const uint32_t* get_blockSize = attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX);
  const uint32_t* get_antiquantMode = attrs->GetAttrPointer<uint32_t>(ATTR_ANTIQUANT_MODE_INDEX);
  const bool* get_softmaxLseFlag = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
  const uint32_t* get_keyAntiquantMode = attrs->GetAttrPointer<uint32_t>(ATTR_KEY_ANTIQUANT_MODE_INDEX);
  const uint32_t* get_valueAntiquantMode = attrs->GetAttrPointer<uint32_t>(ATTR_VALUE_ANTIQUANT_MODE_INDEX);

  int64_t num_heads = *get_num_heads;
  double dScaleValue = *scaleValue;
  int64_t pre_tokens = *get_pre_tokens;
  int64_t next_tokens = *get_next_tokens;
  int64_t kvHeadNum = *get_kvHeadNum;
  int64_t sparseMode = *get_sparseMode;
  int64_t innerPrecise = *get_innerPrecise;
  int64_t blockSize = *get_blockSize;
  int64_t antiquantMode = *get_antiquantMode;
  bool softmaxLseFlag = *get_softmaxLseFlag;
  int64_t keyAntiquantMode = *get_keyAntiquantMode;
  int64_t valueAntiquantMode = *get_valueAntiquantMode;

  if (innerPrecise < 0 || innerPrecise > 3) {   // innerPrecise = 2,3对应行无效的高精度和高性能
    OPS_LOG_E("aclnnfallback", "invalid innerPrecise(%ld). Only support 0~3 now.", innerPrecise);
    return GRAPH_FAILED;
  }
  OPS_LOG_D("aclnnFallback",
          "FusedInferAttentionScore fallback begin, num_heads = %ld, dScaleValue = %lf",
          num_heads, dScaleValue);
  OPS_LOG_D("aclnnFallback",
          "pre_tokens = %ld, next_tokens = %ld, kvHeadNum = %ld, sparseMode = %ld, innerPrecise = %ld",
          pre_tokens, next_tokens, kvHeadNum, sparseMode, innerPrecise);

  if (sparseMode >= 10 && sparseMode <= 14) {  // 10: min  14: max 
    innerPrecise = 0;
    sparseMode -= 10;  // subtract 10 to modify sparseMode
    OPS_LOG_D("aclnnFallback",
            "because sparseMode in range [10, 14], after modification, sparseMode = %ld, innerPrecise = %ld.",
            sparseMode, innerPrecise);
  }

  auto api_ret = EXEC_OPAPI_CMD(aclnnFusedInferAttentionScoreV2, query, ge_tenserListKey, ge_tenserListValue,
                                pseShiftGe, attenMaskGe, actSeqArray, actSeqArrayKv, deqScale1, quantScale1,
                                deqScale2, quantScale2, quantOffset2, antiquantScaleGe, antiquantOffsetGe,
                                blocktableGe, queryPaddingGe, kvPaddingGe,
                                keyAntiquantScaleGe, keyAntiquantOffsetGe, valueAntiquantScaleGe, valueAntiquantOffsetGe,
                                keySharedPrefixGe, valueSharedPrefixGe, actSeqSharedPrefix,
                                num_heads, dScaleValue, pre_tokens, next_tokens, layout, kvHeadNum, sparseMode,
                                innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, output, softmaxLse);

  OPS_ERR_IF(api_ret != GRAPH_SUCCESS,
    OPS_LOG_E("aclnnfallback", "api_ret faild:%u", api_ret), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPL_OP(FusedInferAttentionScore).OpExecuteFunc(FusedInferHostExecuteFunc).HostInputs({5, 6, 23});
}  // namespace fallback

#ifdef __cplusplus
}
#endif
