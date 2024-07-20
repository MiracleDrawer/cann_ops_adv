/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling.h"
#include "incre_flash_attention_tiling.h"
#include "prompt_flash_attention_tiling.h"
#include "error/ops_error.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
static ge::graphStatus ConvertContextToParamsPFA(gert::TilingContext* context, ContextParamsForPFATiling& contextKeyParams) {
    contextKeyParams.opName = context->GetNodeName();
    bool inputOutputIsNullPtr = (context->GetInputDesc(QUERY_INDEX) == nullptr) || (context->GetInputDesc(KEY_INDEX) == nullptr) ||
                                (context->GetInputDesc(VALUE_INDEX) == nullptr) || (context->GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
                                (context->GetInputShape(QUERY_INDEX) == nullptr) || (context->GetInputShape(KEY_INDEX) == nullptr) ||
                                (context->GetInputShape(VALUE_INDEX) == nullptr) || (context->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OPS_ERR_IF(inputOutputIsNullPtr,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "q, k, v or attenOut is nullptr!"),
                return ge::GRAPH_FAILED);

    contextKeyParams.isKvContinuous = 1;
    contextKeyParams.emptyTensor = 0;
    contextKeyParams.fromTilingSink = 0;
    contextKeyParams.fromFused = FROM_FUSED_FLAG;
    contextKeyParams.maxKVs = 0;
    contextKeyParams.pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask = context->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    OPS_ERR_IF((contextKeyParams.attentionMask != nullptr) &&
                    (context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_BOOL) &&
                    (context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_INT8) &&
                    (context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_UINT8),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Invalid attention mask datatype! Only support BOOL, INT8 and UINT8"),
                    return ge::GRAPH_FAILED);
    contextKeyParams.actualSeqenceLengthQ = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSeqenceLengthKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.antiquantScale = context->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffset = context->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.queryPaddingSize = context->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    contextKeyParams.kvPaddingSize = context->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    contextKeyParams.inputDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType = context->GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
        context->GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ? 
        context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.quantScale2Type = (context->GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType() : ge::DT_FLOAT;    
    contextKeyParams.outputDataType = context->GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context->GetInputShape(KEY_INDEX);
    contextKeyParams.valueInputShape = context->GetInputShape(VALUE_INDEX);
    contextKeyParams.pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.deqScale1Shape = context->GetOptionalInputShape(DEQUANT_SCALE1_INDEX);
    contextKeyParams.scale1Shape = context->GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape = context->GetOptionalInputShape(DEQUANT_SCALE2_INDEX);
    contextKeyParams.scale2Shape = context->GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape = context->GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.antiquantScaleShape = context->GetOptionalInputShape(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffsetShape = context->GetOptionalInputShape(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.outputShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    contextKeyParams.lseoutputShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Attributes returned from GetAttrs() is a nullptr"),
                  return ge::GRAPH_FAILED);
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_INDEX);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    contextKeyParams.preToken = attrs->GetAttrPointer<int32_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken = attrs->GetAttrPointer<int32_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1 : 0;
    contextKeyParams.softmaxLseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    contextKeyParams.isSoftMaxLseEnable = (contextKeyParams.softmaxLseFlag == nullptr) ? false : *contextKeyParams.softmaxLseFlag;

    const string layoutStr = string(contextKeyParams.layout);
    auto batchOfQ = 1;
    auto batchOfK = 1;
    if (layoutStr != "NSD") {
        batchOfQ = contextKeyParams.queryInputShape->GetStorageShape().GetDim(0);
        batchOfK = contextKeyParams.keyInputShape->GetStorageShape().GetDim(0);
    }

    if (batchOfQ != batchOfK) {
        int64_t validBatchOfK = 0;
        int64_t validBatchOfV = 0;
        int64_t cumulativeKeyS = 0;
        int64_t cumulativeValueS = 0;
        contextKeyParams.kTensorList.resize(batchOfQ);
        contextKeyParams.vTensorList.resize(batchOfQ);
        while (context->GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
            contextKeyParams.kTensorList[validBatchOfK] = context->GetDynamicInputShape(KEY_INDEX, validBatchOfK);
            OPS_ERR_IF(contextKeyParams.kTensorList[validBatchOfK]->GetStorageShape().GetDim(0) != 1,
                            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Batch value of Key is NOT 1 but should be 1 under tensorlist mode!"),
                            return ge::GRAPH_FAILED);
            validBatchOfK++;
        }

        while (context->GetDynamicInputShape(VALUE_INDEX, validBatchOfV) != nullptr) {
            contextKeyParams.vTensorList[validBatchOfV] = context->GetDynamicInputShape(VALUE_INDEX, validBatchOfV);
            OPS_ERR_IF(contextKeyParams.vTensorList[validBatchOfV]->GetStorageShape().GetDim(0) != 1,
                            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Batch value of Value is NOT 1 but should be 1 under tensorlist mode!"),
                            return ge::GRAPH_FAILED);
            validBatchOfV++;
        }

        if (layoutStr == "BSH") {                                 // check all H across batches and KVs are the same under BSH layout
            auto standardH = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(2);
            for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
                if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(2) != standardH) || (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(2) != standardH)) {
                    OPS_LOG_E("FusedInferAttentionScore", "D is not the same across batch and Key Value under tensorlist mode!");
                    return ge::GRAPH_FAILED;
                }
                if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) != contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
                    OPS_LOG_E("FusedInferAttentionScore", "S from Key and Value does NOT equal but they should!");
                    return ge::GRAPH_FAILED;
                }
                if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) == 0) {
                    contextKeyParams.emptyTensor = 1;
                }
                cumulativeKeyS += contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1);
                cumulativeValueS += contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1);
                contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs, uint32_t(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
            }
        } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND") {                         // check N and D, respectively, are the same across batches and KVs under BNSD/BNSD_BSND
            auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(1);
            auto standardD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(3);
            int64_t tmpNKv = (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads : *contextKeyParams.headsNumber;
            if (tmpNKv != standardN) {
                OPS_LOG_E("FusedInferAttentionScore", "kvN from tensorlist does NOT EQUAL kvN from attribute!");
                return ge::GRAPH_FAILED;
            }

            for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
                if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) != standardN) || (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1) != standardN)) {
                    OPS_LOG_E("FusedInferAttentionScore", "N is not the same across batch and Key Value under tensorlist mode!");
                    return ge::GRAPH_FAILED;
                }
                if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(3) != standardD) || (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(3) != standardD)) {
                    OPS_LOG_E("FusedInferAttentionScore", "D is not the same across batch and Key Value under tensorlist mode!");
                    return ge::GRAPH_FAILED;
                }
                if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(2) != contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(2)) {
                    OPS_LOG_E("FusedInferAttentionScore", "S from Key and Value does NOT equal but they should!");
                    return ge::GRAPH_FAILED;
                }
                if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(2) == 0) {
                    contextKeyParams.emptyTensor = 1;
                }
                cumulativeKeyS += contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(2);
                cumulativeValueS += contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(2);
                contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs, uint32_t(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(2)));
            }
        } else {                                                  // check N and D, respectively, are the same across batches and KVs under BSND
            auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(2);
            auto standardD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(3);
            int64_t tmpNKv = (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads : *contextKeyParams.headsNumber;
            if (tmpNKv != standardN) {
                OPS_LOG_E("FusedInferAttentionScore", "kvN from tensorlist does NOT EQUAL kvN from attribute!");
                return ge::GRAPH_FAILED;
            }

            for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
                if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(2) != standardN) || (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(2) != standardN)) {
                    OPS_LOG_E("FusedInferAttentionScore", "N is not the same across batch and Key Value under tensorlist mode!");
                    return ge::GRAPH_FAILED;
                }
                if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(3) != standardD) || (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(3) != standardD)) {
                    OPS_LOG_E("FusedInferAttentionScore", "D is not the same across batch and Key Value under tensorlist mode!");
                    return ge::GRAPH_FAILED;
                }
                if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) != contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
                    OPS_LOG_E("FusedInferAttentionScore", "S from Key and Value does NOT equal but they should!");
                    return ge::GRAPH_FAILED;
                }
                if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) == 0) {
                    contextKeyParams.emptyTensor = 1;
                }
                cumulativeKeyS += contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1);
                cumulativeValueS += contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1);
                contextKeyParams.maxKVs = std::max(contextKeyParams.maxKVs, uint32_t(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
            }
        }

        OPS_ERR_IF((batchOfQ != validBatchOfK) || (validBatchOfK != validBatchOfV),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Batch of Query, Key and Value do NOT equal but should equal under tensorlist mode!"),
                    return ge::GRAPH_FAILED);

        OPS_ERR_IF((contextKeyParams.emptyTensor == 1) && (cumulativeKeyS != 0) && (cumulativeValueS != 0),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Got empty tensor in key and value which is not continuous.!!"),
                    return ge::GRAPH_FAILED);
        contextKeyParams.isKvContinuous = 0;
    }

    OPS_ERR_IF(((contextKeyParams.isKvContinuous == 0) &&
                ((contextKeyParams.queryPaddingSize != nullptr) ||
                (context->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX) != nullptr))),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "when tensorlist is used, left padding is not supported!"),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(((contextKeyParams.queryPaddingSize != nullptr) &&
                (contextKeyParams.queryPaddingSize->GetStorageShape().GetShapeSize() !=1 ||
                contextKeyParams.queryPaddingSize->GetStorageShape().GetDimNum() != 1)),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Query PaddingSize input is invalid!"),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(((contextKeyParams.kvPaddingSize != nullptr) &&
                (contextKeyParams.kvPaddingSize->GetStorageShape().GetShapeSize() != 1 ||
                contextKeyParams.kvPaddingSize->GetStorageShape().GetDimNum() != 1)),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "KV PaddingSize input is invalid!"),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(((contextKeyParams.queryPaddingSize != nullptr) &&
                (contextKeyParams.actualSeqenceLengthQ == nullptr)),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "if Query has leftpadding, actual_seq_lengths are required!"),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(((contextKeyParams.kvPaddingSize != nullptr) &&
                (contextKeyParams.actualSeqenceLengthKV == nullptr)),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "if KV has leftpadding, actual_seq_lengths_kv are required!"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToParamsIFA(gert::TilingContext& context,
                                                 IncreFlashAttentionContext& ifaContext) {
  if (context.GetNodeName() == nullptr) {
    OPS_LOG_E("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
    return ge::GRAPH_FAILED;
  }
  ifaContext.opName = context.GetNodeName();
  ifaContext.platformInfo = context.GetPlatformInfo();
  ifaContext.query.desc = context.GetInputDesc(QUERY_INDEX);
  ifaContext.query.shape = context.GetInputShape(QUERY_INDEX);
  ifaContext.key.desc = context.GetInputDesc(KEY_INDEX);
  ifaContext.key.shape = context.GetInputShape(KEY_INDEX);
  OPS_ERR_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
                  OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "shape of query of shape of key is null."),
                  return ge::GRAPH_FAILED);
  auto batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(0);
  auto batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(0);
  if (batchOfQuery != batchOfKey) {
    ifaContext.kCache.resize(batchOfQuery);
    ifaContext.vCache.resize(batchOfQuery);
    for (int64_t size = 0; size < batchOfQuery; ++size) {
      ifaContext.kCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, size));
      ifaContext.vCache[size] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, size));
    }
  } else {
    ifaContext.kCache.resize(1);
    ifaContext.vCache.resize(1);
    ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INDEX, 0));
    ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INDEX, 0));
  }

  ifaContext.value.desc = context.GetInputDesc(VALUE_INDEX);
  ifaContext.value.shape = context.GetInputShape(VALUE_INDEX);
  ifaContext.pseShift.desc = context.GetOptionalInputDesc(PSE_SHIFT_INDEX);
  ifaContext.pseShift.tensor = context.GetOptionalInputTensor(PSE_SHIFT_INDEX);

  ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INDEX);
  ifaContext.attenMask.tensor= context.GetOptionalInputTensor(ATTEN_MASK_INDEX);
  ifaContext.attenOut.desc = context.GetOutputDesc(ATTENTION_OUT_INDEX);
  ifaContext.attenOut.shape = context.GetOutputShape(ATTENTION_OUT_INDEX);

  ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
  ifaContext.deqScale1.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE1_INDEX);
  ifaContext.quantScale1.tensor = context.GetOptionalInputTensor(QUANT_SCALE1_INDEX);
  ifaContext.deqScale2.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE2_INDEX);
  ifaContext.quantScale2.tensor = context.GetOptionalInputTensor(QUANT_SCALE2_INDEX);
  ifaContext.quantOffset2.tensor = context.GetOptionalInputTensor(QUANT_OFFSET2_INDEX);
  ifaContext.quantScale2.desc = context.GetOptionalInputDesc(QUANT_SCALE2_INDEX);
  ifaContext.quantOffset2.desc = context.GetOptionalInputDesc(QUANT_OFFSET2_INDEX);
  ifaContext.antiquantScale.tensor = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
  ifaContext.antiquantScale.desc = context.GetOptionalInputDesc(ANTIQUANT_SCALE_INDEX);
  ifaContext.antiquantOffset.tensor = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
  ifaContext.antiquantOffset.desc = context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
  ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
  ifaContext.kvPaddingSize.tensor = context.GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
  ifaContext.keyAntiquantScale.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
  ifaContext.keyAntiquantScale.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX);
  ifaContext.keyAntiquantOffset.tensor = context.GetOptionalInputTensor(KEY_ANTIQUANT_OFFSET_INDEX);
  ifaContext.keyAntiquantOffset.desc = context.GetOptionalInputDesc(KEY_ANTIQUANT_OFFSET_INDEX);
  ifaContext.valueAntiquantScale.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
  ifaContext.valueAntiquantScale.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX);
  ifaContext.valueAntiquantOffset.tensor = context.GetOptionalInputTensor(VALUE_ANTIQUANT_OFFSET_INDEX);
  ifaContext.valueAntiquantOffset.desc = context.GetOptionalInputDesc(VALUE_ANTIQUANT_OFFSET_INDEX);
  ifaContext.keySharedPrefix.tensor = context.GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
  ifaContext.keySharedPrefix.desc = context.GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX);
  ifaContext.valueSharedPrefix.tensor = context.GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
  ifaContext.valueSharedPrefix.desc = context.GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX);
  ifaContext.actualSharedPrefixLen.tensor = context.GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);

  auto attrs = context.GetAttrs();
  OPS_ERR_IF(attrs == nullptr,
                  OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "attrs got from ge is nullptr"),
                  return ge::GRAPH_FAILED);

  ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
  ifaContext.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
  ifaContext.layOut = attrs->GetStr(ATTR_INPUT_LAYOUT_INDEX);
  ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX);
  ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX);
  ifaContext.antiquantMode = attrs->GetAttrPointer<uint32_t>(ANTIQUANT_MODE_INDEX);
  ifaContext.softmaxLseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
  ifaContext.keyAntiquantMode = attrs->GetAttrPointer<uint32_t>(KEY_ANTIQUANT_MODE_INDEX);
  ifaContext.valueAntiquantMode = attrs->GetAttrPointer<uint32_t>(VALUE_ANTIQUANT_MODE_INDEX);

  OPS_ERR_IF(context.GetWorkspaceSizes(1) == nullptr,
                  OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
                  return ge::GRAPH_FAILED);
  ifaContext.workSpaces = context.GetWorkspaceSizes(1);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DoOpTilingFusedInferAttentionScore(gert::TilingContext* context) {
    if (context == nullptr) {
        OPS_LOG_E("FusedInferAttentionScore", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    auto tempQ = context->GetInputShape(QUERY_INDEX);
    auto tempOut = context->GetOutputShape(ATTENTION_OUT_INDEX);
    auto tempLse = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    uint32_t maxDlimit = 512;
    uint32_t tempD = 1;
    OPS_ERR_IF((tempQ == nullptr),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Query input is null pointer!"),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((tempQ->GetStorageShape().GetShapeSize() == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Query input is empty!"),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((tempQ->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Query input dims are invalid!"),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((tempOut == nullptr),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Attention_Out is null pointer!"),
                    return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Attributes returned from GetAttrs() is a nullptr"),
                  return ge::GRAPH_FAILED);
    uint32_t tempN = *attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX);
    OPS_ERR_IF(tempN == 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Q numhead is 0!"), return ge::GRAPH_FAILED);
    const string inputLayoutStr = string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    int64_t s = 0;
    int64_t b = tempQ->GetStorageShape().GetDim(0);
    bool lseFlag = *attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_INDEX);
    bool usingIFA = false;
    if (inputLayoutStr == "SH") {
        OPS_LOG_E(context->GetNodeName(), "SH layout is not supported!");
        return ge::GRAPH_FAILED;
    }
    else if (inputLayoutStr == "BNSD" || inputLayoutStr == "BNSD_BSND") {
        s = tempQ->GetStorageShape().GetDim(2);
    }
    else {
        s = tempQ->GetStorageShape().GetDim(1);
    }
    if (inputLayoutStr == "NSD") {
        b = 1;
        OPS_ERR_IF((tempQ->GetStorageShape().GetDimNum() != 3),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "input shape dim should be 3!"),
                    return ge::GRAPH_FAILED);
        tempD = tempQ->GetStorageShape().GetDim(2);
        OPS_ERR_IF((tempQ->GetStorageShape() != tempOut->GetStorageShape()),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is NSD and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, %ld]!",
                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(2),
                    tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(2)),
                    return ge::GRAPH_FAILED);
    } else if (inputLayoutStr == "BSH") {
        OPS_ERR_IF((tempQ->GetStorageShape().GetDimNum() != 3),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "input shape dim should be 3!"),
                    return ge::GRAPH_FAILED);
        tempD = tempQ->GetStorageShape().GetDim(2) / tempN;
        OPS_ERR_IF((tempQ->GetStorageShape() != tempOut->GetStorageShape()),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is BSH and Query shape size[%ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, %ld]!",
                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(2),
                    tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(2)),
                    return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF((tempQ->GetStorageShape().GetDimNum() != 4),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "input shape dim should be 4!"),
                    return ge::GRAPH_FAILED);
        tempD = tempQ->GetStorageShape().GetDim(3);
        if (inputLayoutStr == "BNSD_BSND") {
            OPS_ERR_IF(((tempQ->GetStorageShape().GetDim(0) != tempOut->GetStorageShape().GetDim(0)) ||
                        (tempQ->GetStorageShape().GetDim(1) != tempOut->GetStorageShape().GetDim(2)) ||
                        (tempQ->GetStorageShape().GetDim(2) != tempOut->GetStorageShape().GetDim(1)) ||
                        (tempQ->GetStorageShape().GetDim(3) != tempOut->GetStorageShape().GetDim(3))
                    ),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is BNSD_BSND and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, %ld, %ld]!",
                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(2), tempQ->GetStorageShape().GetDim(3),
                    tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(2), tempOut->GetStorageShape().GetDim(3)),
                    return ge::GRAPH_FAILED);
        } else if (inputLayoutStr == "BNSD") {
            OPS_ERR_IF((tempQ->GetStorageShape() != tempOut->GetStorageShape()),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Layout is BNSD and Query shape size[%ld, %ld, %ld, %ld] does NOT match Attention Out shape size[%ld, %ld, %ld, %ld]!",
                    tempQ->GetStorageShape().GetDim(0), tempQ->GetStorageShape().GetDim(1), tempQ->GetStorageShape().GetDim(2), tempQ->GetStorageShape().GetDim(3),
                    tempOut->GetStorageShape().GetDim(0), tempOut->GetStorageShape().GetDim(1), tempOut->GetStorageShape().GetDim(2), tempOut->GetStorageShape().GetDim(3)),
                    return ge::GRAPH_FAILED);
        }
    }
    OPS_ERR_IF((tempD > maxDlimit),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "D should be less than or equal to 512 of Q/KV shape!"),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(((s == 1) && (inputLayoutStr == "BNSD_BSND")),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "BNSD_BSND layout is not supported when S is 1!"),
                return ge::GRAPH_FAILED);

    if ((s == 1) && ((inputLayoutStr == "BSH") || (inputLayoutStr == "BNSD") || (inputLayoutStr == "BSND"))) {
        usingIFA = true;
    }

    if (usingIFA) {
        // IFA tiling path
        IncreFlashAttentionTilingDataV2 ifaTilingData;
        IncreFlashAttentionContext ifaContext{0};
        auto ret = ConvertContextToParamsIFA(*context, ifaContext);
        if (ret != ge::GRAPH_SUCCESS) {
          OPS_LOG_E(context->GetNodeName(), "Error occored while convert tilingContext to ifa context");
          return ret;
        }

        return TilingIncreFlashAttentionAdapter(context, ifaContext, ifaTilingData);
    }
    else {
        // PFA tiling process
        PromptFlashAttentionTilingData pfaTilingData;
        PromptFlashAttentionTiling pfa_tiling(nullptr);
        ContextParamsForPFATiling contextParamsForPFATiling = {0};
        PromptFlashAttentionCompileInfo tempCompileInfoPtr = {0};

        OPS_ERR_IF((attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX) != nullptr) &&
                   (*attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX) != 0),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "block_size is not supported!"),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF((context->GetOptionalInputTensor(BLOCK_TABLE_INDEX) != nullptr),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "block_table is not supported!"),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF((attrs->GetAttrPointer<uint64_t>(ANTIQUANT_MODE_INDEX) != nullptr) &&
                   (*attrs->GetAttrPointer<uint64_t>(ANTIQUANT_MODE_INDEX) != 0),
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "antiquant_mode is not supported!"),
                        return ge::GRAPH_FAILED);

        auto platformInfoPtr = context->GetPlatformInfo();
        OPS_ERR_IF(platformInfoPtr == nullptr,
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                  return ge::GRAPH_FAILED);
        
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        tempCompileInfoPtr.aivNum = ascendcPlatform.GetCoreNumAiv();
        tempCompileInfoPtr.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tempCompileInfoPtr.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, tempCompileInfoPtr.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, tempCompileInfoPtr.l0CSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, tempCompileInfoPtr.l0ASize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, tempCompileInfoPtr.l0BSize);
        tempCompileInfoPtr.socShortName = ascendcPlatform.GetSocVersion();
        if (tempCompileInfoPtr.socShortName == platform_ascendc::SocVersion::ASCEND310P) {
            // sys workspace size default value
            tempCompileInfoPtr.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        } else {
            tempCompileInfoPtr.defaultSysWorkspaceSize = 0;
        }

        contextParamsForPFATiling.compileInfoPtr = &tempCompileInfoPtr;
        auto ret = ConvertContextToParamsPFA(context, contextParamsForPFATiling);
        if (ret != ge::GRAPH_SUCCESS) {
          OPS_LOG_E(context->GetNodeName(), "Error occored while convert tilingContext to PFA context");
          return ret;
        }
        if (lseFlag != false) {
            if (pfa_tiling.CheckNonEmptyShapeExceptions(contextParamsForPFATiling, contextParamsForPFATiling.lseoutputShape, "softmaxLse")) {
                return ge::GRAPH_FAILED;
            }
            OPS_ERR_IF(((lseFlag != false) && (tempLse->GetStorageShape().GetDimNum() != 4)),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SoftmaxLse shape dim should be 4!"),
                    return ge::GRAPH_FAILED);
            OPS_ERR_IF(((lseFlag != false) && ((tempLse->GetStorageShape().GetDim(0) != b) ||
                                                (tempLse->GetStorageShape().GetDim(1) != tempN) ||
                                                (tempLse->GetStorageShape().GetDim(2) != s) ||
                                                (tempLse->GetStorageShape().GetDim(3) != 1))),
                            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SoftmaxLse shape size[%ld, %ld, %ld, %ld] does not match BNS1[%ld, %u, %ld, 1]!",
                            tempLse->GetStorageShape().GetDim(0), tempLse->GetStorageShape().GetDim(1), tempLse->GetStorageShape().GetDim(2), tempLse->GetStorageShape().GetDim(3),
                            b, tempN, s),
                            return ge::GRAPH_FAILED);
        }
        const string inputLayout = string(contextParamsForPFATiling.layout);
        OPS_ERR_IF((((contextParamsForPFATiling.inputDataType == ge::DT_INT8) || (contextParamsForPFATiling.kDataType == ge::DT_INT8) ||
                    (contextParamsForPFATiling.outputDataType == ge::DT_INT8)) && (tempD % D_ALIGN_32 != 0)),
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "D should be 32 elements aligned when int8 is involved!!"),
                  return ge::GRAPH_FAILED);
        OPS_ERR_IF((tempD % D_ALIGN_16 != 0),
                  OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "D should be 16 elements aligned when with FP16/BF16 dtype!"),
                  return ge::GRAPH_FAILED);
        uint64_t tilingKey = 7U;
        uint32_t blockDimToBeSet;
        ret = pfa_tiling.RunBigKernelTilingWithParams(contextParamsForPFATiling, tilingKey, blockDimToBeSet, pfaTilingData);
        tilingKey += BENCHMARK_TILING_KEY;
        OPS_LOG_D(contextParamsForPFATiling.opName, "The final tiling key is: %lu", tilingKey);
        context->SetTilingKey(tilingKey);
        context->SetBlockDim(blockDimToBeSet);
        pfa_tiling.PromptFlashAttentionSetTilingData(context, pfaTilingData);
        return ret;
    }
}

extern "C" {
    __attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext* context) {
        return TilingIncreFlashAttention(context);
    }
    __attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingFusedInferAttentionScore(gert::TilingContext* context) {
        return DoOpTilingFusedInferAttentionScore(context);
    }
}

} // namespace optiling