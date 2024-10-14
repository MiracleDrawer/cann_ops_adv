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
 * \file incre_flash_attention_tiling.cc
 * \brief
 */

#include "incre_flash_attention_tiling.h"

#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

template <typename T> inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

static int64_t CeilDivision(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

// 获取公约数
static uint32_t increGcd(uint32_t a, uint32_t b)
{
    if (a % b == 0) {
        return b;
    }
    return increGcd(b, a % b);
}

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10U * RecursiveSum(templateIds...);
}
constexpr uint64_t IFA_TILINGKEYOFFSET = uint64_t(10000000000000000UL);          // 10^16
constexpr uint64_t IFA_PERF_MODE_TILINGKEYOFFSET = uint64_t(1000000000000000UL); // 10^15
template <typename... Args> constexpr uint64_t IFA_GET_TILINGKEY(Args... templateIds)
{
    return RecursiveSum(templateIds...);
}

ge::graphStatus IFATiling::GetNpuInfo()
{
    OPS_ERR_IF(context_->platformInfo == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aicNum_ = ascendcPlatform.GetCoreNumAic();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        socVersion_ = IfaSocVersion::SOC_ASCEND_310P;
        coreNum_ = aicNum_; // use aic num in 310p
    } else {
        socVersion_ = IfaSocVersion::SOC_ASCEND_910B;
        coreNum_ = aivNum_; // default aiv num
    }

    OPS_ERR_IF(aicNum_ == 0 || aivNum_ == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::PreProcess()
{
    if (ProcessBaseInputs()) {
        return ge::GRAPH_FAILED;
    }
    bool ret = CheckIfRollBack();
    if (ret) {
        passToOldTiling_ = true;
        return ge::GRAPH_FAILED;
    }

    if (ProcessOptionalTensors()) {
        return ge::GRAPH_FAILED;
    }

    SetupPerfMode();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SetL2CacheFlag()
{
    uint32_t kvTypeSize = NUM_BYTES_FLOAT16;
    auto kDType = context_->key.desc->GetDataType();
    switch (kDType) {
        case ge::DT_FLOAT:
            kvTypeSize = NUM_BYTES_FLOAT;
            break;
        case ge::DT_FLOAT16:
            kvTypeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            kvTypeSize = NUM_BYTES_BF16;
            break;
        case ge::DT_BOOL:
            kvTypeSize = NUM_BYTES_BOOL;
            break;
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_INT4:
            kvTypeSize = NUM_BYTES_INT8;
            break;
        default:
            OPS_LOG_E(context_->opName, "Data type %s is not currently supported.",
                      DataTypeToSerialString(kDType).c_str());
            return ge::GRAPH_FAILED;
    }

    uint64_t kvSize = 0;
    auto batchOfQuery = context_->query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = context_->key.shape->GetStorageShape().GetDim(0);
    if (context_->blockTable.tensor != nullptr) {
        kvSize = context_->key.shape->GetStorageShape().GetShapeSize();
    } else if (batchOfQuery != batchOfKey) { /* kv noncontinuous */
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            auto keyTensorInList = context_->kCache[size];
            kvSize += keyTensorInList->GetStorageShape().GetShapeSize();
        }
    } else {
        kvSize = context_->key.shape->GetStorageShape().GetShapeSize();
    }

    uint64_t l2CacheSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    // ×2考虑K、V，1.2为关闭L2Cache的系数
    if (kvSize * kvTypeSize * 2 >= l2CacheSize * 1.2) {
        OPS_LOG_D(context_->opName, "L2 cache off");
        l2CacheOffFlag_ = 1;
    }

    OPS_LOG_D(context_->opName, "l2CacheOffFlag_:%u, kvSize:%llu, kvTypeSize:%u, l2CacheSize:%u", l2CacheOffFlag_,
              kvSize, kvTypeSize, l2CacheSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckPABlockSize()
{
    OPS_ERR_IF(
        blockSize_ == 0,
        OPS_LOG_E(context_->opName, "When Page Attention is enabled, input attribute blocksize can not be 0."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockSize_ > MAX_BLOCK_SIZE,
                OPS_LOG_E(context_->opName,
                            "When Page Attention is enabled, input attribute blocksize %u can not be larger than %u.",
                            blockSize_, MAX_BLOCK_SIZE),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(((inputKvType_ == ge::DT_INT8) && (blockSize_ % 32 != 0)),
                OPS_LOG_E(context_->opName, "When Page Attention is enabled, if kv cache dtype is int8, input attr "
                                            "blocksize[%u] should be 32 aligned.",blockSize_),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(((inputKvType_ == ge::DT_FLOAT16) || (inputKvType_ == ge::DT_BF16)) && (blockSize_ % 16 != 0),
                OPS_LOG_E(context_->opName,
                            "When Page Attention is enabled, "
                            "if kv cache dtype is float16/bfloat16, input attr blocksize should be 16 aligned"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckBaseInputsNull() {
    // Check base input tensors
    OPS_ERR_IF(context_->query.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->query.shape->GetStorageShape().GetShapeSize() == 0,
               OPS_LOG_E(context_->opName, "Tensor q is empty cause shapesize is 0."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->query.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->key.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->key.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->value.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->value.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->attenOut.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->attenOut.shape == nullptr, OPS_LOG_E(context_->opName, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);

    // Check base input attrs
    OPS_ERR_IF(context_->numHeads == nullptr, OPS_LOG_E(context_->opName, "attr numHeads is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->scaleValue == nullptr, OPS_LOG_E(context_->opName, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->kvHeadNums == nullptr, OPS_LOG_E(context_->opName, "attr kvHeadNums is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->layOut == nullptr, OPS_LOG_E(context_->opName, "attr layOut is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->blockSize == nullptr, OPS_LOG_E(context_->opName, "attr blockSize is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::QKVPreProcess() {
    OPS_ERR_IF(context_->key.desc->GetDataType() != context_->value.desc->GetDataType(),
               OPS_LOG_E(context_->opName, "datatype of k tensor and value tensor is different"), return ge::GRAPH_FAILED);
    batchSizeQ_ = batchSize_ = context_->query.shape->GetStorageShape().GetDim(0);
    inputQType_ = context_->query.desc->GetDataType();
    inputKvType_ = context_->key.desc->GetDataType();
    outputType_ = context_->attenOut.desc->GetDataType();

    numHeads_ = *context_->numHeads;
    numKvHeads_ = *context_->kvHeadNums;
    scaleValue_ = *context_->scaleValue;
    blockSize_ = *context_->blockSize;

    OPS_ERR_IF(numHeads_ == 0, OPS_LOG_E(context_->opName, "numHeads is zero"), return ge::GRAPH_FAILED);
    if (numKvHeads_ == 0) {
        numKvHeads_ = numHeads_;
    }
    OPS_ERR_IF(((numKvHeads_ > numHeads_) || (numHeads_ % numKvHeads_ != 0)),
               OPS_LOG_E(context_->opName, "Attr num_key_value_heads is invalid, n: %u, kvHeadNum: %u", numHeads_,
                         numKvHeads_),
               return ge::GRAPH_FAILED);
    nNumOfQInOneGroup_ = numHeads_ / numKvHeads_;

    std::string layout(context_->layOut);
    uint32_t sOfQuery = 0;
    if (layout == "BSH") {
        inputLayout_ = IfaLayout::BSH_BSND;
        OPS_ERR_IF(context_->query.shape->GetStorageShape().GetDim(2) % numHeads_ != 0,
                   OPS_LOG_E(context_->opName, "H should be an interger multiple of numHeads"),
                   return ge::GRAPH_FAILED);
        sOfQuery = context_->query.shape->GetStorageShape().GetDim(1);
        headDim_ = context_->query.shape->GetStorageShape().GetDim(2) / numHeads_; // 2, dim of H
    } else if (layout == "BSND") {
        inputLayout_ = IfaLayout::BSH_BSND;
        sOfQuery = context_->query.shape->GetStorageShape().GetDim(1);
        headDim_ = context_->query.shape->GetStorageShape().GetDim(3); // 3, dim of D
    } else if (layout == "BNSD") {
        inputLayout_ = IfaLayout::BNSD;
        sOfQuery = context_->query.shape->GetStorageShape().GetDim(2); // 2, dim of S
        headDim_ = context_->query.shape->GetStorageShape().GetDim(3); // 3, dim of D
    } else {
        OPS_LOG_E(context_->opName, "Only support input_layout(BSH, BNSD, BSND), actually is %s", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (inputKvType_ == ge::DT_INT4 && headDim_ % KVINT4_BYTE_BLOCK != 0) {
        OPS_LOG_E(context_->opName, "Number of heads must be a multiple of %u, current dim of D is %u.", KVINT4_BYTE_BLOCK,
                  headDim_);
        return ge::GRAPH_FAILED;
    }
    if (inputKvType_ == ge::DT_INT4) {
        headDimAlign_ = Align(headDim_, KVINT4_BYTE_BLOCK); // 元素个数按照基本块大小对齐
    } else {
        headDimAlign_ = Align(headDim_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    }
    
    OPS_ERR_IF(sOfQuery != 1, OPS_LOG_E(context_->opName, " S of Query:%u is invalid, it should be 1", sOfQuery),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::InputAttrsPreProcess() {
    const uint32_t *innerPrecisePtr = context_->innerPrecise;
    innerPrecise_ = innerPrecisePtr ? *innerPrecisePtr : IFA_HIGH_PERFORMANCE; // 910B默认高性能
    OPS_ERR_IF(((innerPrecise_ != IFA_HIGH_PERFORMANCE) && (innerPrecise_ != IFA_HIGH_PRECISION)),
               OPS_LOG_E(context_->opName, "precision mode[%u] should be 0 or 1", innerPrecise_),
               return ge::GRAPH_FAILED); // 当前只支持高精度0和高性能1
    OPS_LOG_D(context_->opName, "innerPrecise is %u", innerPrecise_);

    blockTypeSize_ = sizeof(float); // 默认按照float计算
    pageAttentionFlag_ = context_->blockTable.tensor != nullptr;

    if (!pageAttentionFlag_) {
        uint32_t kvBatch = context_->key.shape->GetStorageShape().GetDim(0);
        batchContinuousFlag_ = (batchSize_ == kvBatch);
    } else {
        OPS_ERR_IF(inputKvType_ == ge::DT_INT4,
                   OPS_LOG_E(context_->opName,
                             "IFA don't support PageAttenion if the KV Inputtype is INT4 or INT32 currently."),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(context_->blockTable.tensor->GetStorageShape().GetShapeSize() == 0,
                   OPS_LOG_E(context_->opName, "check blockTable shape failed, blockTable shapeSize is zero."),
                   return ge::GRAPH_FAILED);
    }

    if (context_->softmaxLseFlag != nullptr) {
        softmaxLseFlag_ = *context_->softmaxLseFlag;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SetQuantFlag() {
    antiQuantFlag_ = (inputQType_ != inputKvType_) && (inputKvType_ == ge::DT_INT8 || inputKvType_ == ge::DT_INT4);
    if (antiQuantFlag_) {
        if (innerPrecise_ == IFA_HIGH_PRECISION) {
            msdIterNum_ = inputKvType_ == ge::DT_INT4 ? KVINT4_ITER_NUM : HIGH_PRECISION_ITER_NUM;
        } else {
            msdIterNum_ = inputKvType_ == ge::DT_INT4 ? KVINT4_ITER_NUM : ITER_NUM;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessBaseInputs()
{
    if ((CheckBaseInputsNull() != ge::GRAPH_SUCCESS) || (QKVPreProcess() != ge::GRAPH_SUCCESS) ||
        (InputAttrsPreProcess() != ge::GRAPH_SUCCESS) || (KvShapePostProcess() != ge::GRAPH_SUCCESS) ||
        (CheckQKOutShape() != ge::GRAPH_SUCCESS) || (CheckInputFormatAndLimits() != ge::GRAPH_SUCCESS) ||
        (SetL2CacheFlag() != ge::GRAPH_SUCCESS) || (SetQuantFlag() != ge::GRAPH_SUCCESS) ||
        (InitInOutMode() != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
    return ge::GRAPH_SUCCESS;
}

bool IFATiling::EnableAllVec()
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return true;
    }
    if (pageAttentionFlag_) {
        return false;
    }
    if (sysPrefixFlag_) {
        return false;
    }
    if (nNumOfQInOneGroup_ > 1) {
        return false;
    }
    if (headDim_ > 512) { // 全VEC模板仅支持headDim_不大于512
        return false;
    }
    return (inputQType_ == ge::DT_FLOAT16) && (inputKvType_ == ge::DT_FLOAT16) && (outputType_ == ge::DT_FLOAT16);
}

void IFATiling::SetupPerfMode()
{
    // 310P
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        perfMode_ = IfaPerfMode::BMM_ALL_BY_VEC;
    } else {
        if (EnableAllVec()) {
            perfMode_ = IfaPerfMode::BMM_ALL_BY_VEC;
        }
    }
}

bool IFATiling::EnableC1V1()
{
    if (splitKVFlag_) {
        return false;
    }
    if (sysPrefixFlag_) {
        return false;
    }
    // 2:核数不超过vector总核数一半，可以按1:1启动cube和vector
    return (perfMode_ == IfaPerfMode::NORMAL) && (batchSize_ * numKvHeads_ * 2 <= aivNum_);
}

void IFATiling::UpdatePerfMode()
{
    if (EnableC1V1()) {
        perfMode_ = IfaPerfMode::C1_V1;
    }
}

ge::graphStatus IFATiling::CheckInputParameterFormat()
{
    auto qFormat = context_->query.desc->GetOriginFormat();
    auto kFormat = context_->key.desc->GetOriginFormat();
    auto vFormat = context_->value.desc->GetOriginFormat();

    OPS_ERR_IF((qFormat != ge::FORMAT_ND && qFormat != ge::FORMAT_NCHW && qFormat != ge::FORMAT_NHWC),
               OPS_LOG_E(context_->opName, "Query format %u should be ND/NCHW/NHWC", qFormat), return ge::GRAPH_FAILED);
    OPS_ERR_IF((kFormat != ge::FORMAT_ND && kFormat != ge::FORMAT_NCHW && kFormat != ge::FORMAT_NHWC),
               OPS_LOG_E(context_->opName, "Key format %u should be ND/NCHW/NHWC", kFormat), return ge::GRAPH_FAILED);
    OPS_ERR_IF((vFormat != ge::FORMAT_ND && vFormat != ge::FORMAT_NCHW && vFormat != ge::FORMAT_NHWC),
               OPS_LOG_E(context_->opName, "Value format %u should be ND/NCHW/NHWC", vFormat), return ge::GRAPH_FAILED);
  if(context_->attenMask.desc != nullptr){
    auto mFormat = context_->attenMask.desc->GetOriginFormat();
    OPS_ERR_IF((mFormat != ge::FORMAT_ND && mFormat != ge::FORMAT_NCHW && mFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "atten_mask format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->kvPaddingSize.desc != nullptr){
    auto kvPaddingFormat = context_->kvPaddingSize.desc->GetOriginFormat();
    OPS_ERR_IF((kvPaddingFormat != ge::FORMAT_ND && kvPaddingFormat != ge::FORMAT_NCHW && kvPaddingFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "padding_size format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->keySharedPrefix.desc != nullptr){
    auto kPrefixFormat = context_->keySharedPrefix.desc->GetOriginFormat();
    OPS_ERR_IF((kPrefixFormat != ge::FORMAT_ND && kPrefixFormat != ge::FORMAT_NCHW && kPrefixFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "k_prefix format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->valueSharedPrefix.desc != nullptr){
    auto vPrefixFormat = context_->valueSharedPrefix.desc->GetOriginFormat();
    OPS_ERR_IF((vPrefixFormat != ge::FORMAT_ND && vPrefixFormat != ge::FORMAT_NCHW && vPrefixFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "v_prefix format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckInputAntiquantFormat() {
  if(context_->antiquantScale.desc != nullptr){
    auto aScaleFormat = context_->antiquantScale.desc->GetOriginFormat();
    OPS_ERR_IF((aScaleFormat != ge::FORMAT_ND && aScaleFormat != ge::FORMAT_NCHW && aScaleFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "antiquantScale format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->antiquantOffset.desc != nullptr){
    auto aOffsetFormat = context_->antiquantOffset.desc->GetOriginFormat();
    OPS_ERR_IF((aOffsetFormat != ge::FORMAT_ND && aOffsetFormat != ge::FORMAT_NCHW && aOffsetFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "antiquantOffset format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->keyAntiquantScale.desc != nullptr){
  auto kScaleFormat = context_->keyAntiquantScale.desc->GetOriginFormat();
    OPS_ERR_IF((kScaleFormat != ge::FORMAT_ND && kScaleFormat != ge::FORMAT_NCHW && kScaleFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "keyAntiquantScale format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->keyAntiquantOffset.desc != nullptr){
  auto kOffsetFormat = context_->keyAntiquantOffset.desc->GetOriginFormat();
    OPS_ERR_IF((kOffsetFormat != ge::FORMAT_ND && kOffsetFormat != ge::FORMAT_NCHW && kOffsetFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "keyAntiquantOffset format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->valueAntiquantScale.desc != nullptr){
  auto vScaleFormat = context_->valueAntiquantScale.desc->GetOriginFormat();
    OPS_ERR_IF((vScaleFormat != ge::FORMAT_ND && vScaleFormat != ge::FORMAT_NCHW && vScaleFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "valueAntiquantScale format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  if(context_->valueAntiquantOffset.desc != nullptr){
  auto vOffsetFormat = context_->valueAntiquantOffset.desc->GetOriginFormat();
    OPS_ERR_IF((vOffsetFormat != ge::FORMAT_ND && vOffsetFormat != ge::FORMAT_NCHW && vOffsetFormat != ge::FORMAT_NHWC),
             OPS_LOG_E(context_->opName, "valueAntiquantOffset format should be ND/NCHW/NHWC"),
             return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckInputFormatAndLimits() {  
  if(CheckInputParameterFormat() != ge::GRAPH_SUCCESS || CheckInputAntiquantFormat() != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
  }
    OPS_ERR_IF(
        (nNumOfQInOneGroup_ > 64),
        OPS_LOG_E(context_->opName, "numHeads_ / numKvHeads_ = %u, cannot be greater than 64", nNumOfQInOneGroup_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF((inputQType_ == ge::DT_INT8 && inputKvType_ == ge::DT_INT8),
               OPS_LOG_E(context_->opName, "IFA not support qkv datatype all int8."), return ge::GRAPH_FAILED);

    OPS_ERR_IF(((inputQType_ == ge::DT_FLOAT16) && (inputKvType_ != ge::DT_FLOAT16 && inputKvType_ != ge::DT_INT8 && inputKvType_ != ge::DT_INT4)),
               OPS_LOG_E(context_->opName, "when input Q type is fp16, KV type %u should be fp16 or int8 or int4", inputKvType_),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(((inputQType_ == ge::DT_BF16) && (inputKvType_ != ge::DT_BF16 && inputKvType_ != ge::DT_INT8 && inputKvType_ != ge::DT_INT4)),
               OPS_LOG_E(context_->opName, "when input Q type is bf16, KV type %u should be bf16 or int8 or int4", inputKvType_),
               return ge::GRAPH_FAILED);

    if (pageAttentionFlag_) {
        OPS_ERR_IF(
            (inputKvType_ == ge::DT_FLOAT16 || inputKvType_ == ge::DT_BF16) && (blockSize_ % 16 != 0),
            OPS_LOG_E(context_->opName, "blockSize=%u, it need align to 16 when kv dtype is fp16/bf16.", blockSize_),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputKvType_ == ge::DT_INT8) && (blockSize_ % 32 != 0),
                   OPS_LOG_E(context_->opName, "blockSize=%u, it need align to 32 when kv dtype is int8.", blockSize_),
                   return ge::GRAPH_FAILED);
    }

    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        OPS_ERR_IF((numHeads_ != numKvHeads_), // unsupport gqa
                   OPS_LOG_E(context_->opName, "numHeads:%u of key must be equal to numHeads:%u of kv when 310P.",
                             numHeads_, numKvHeads_),
                   return ge::GRAPH_FAILED);

        OPS_ERR_IF((batchSize_ > 256),
                   OPS_LOG_E(context_->opName, "batch size:%u cannot be greater than 256 when 310P.", batchSize_),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF((sMax_ > 65536),
                   OPS_LOG_E(context_->opName, "sMax:%u cannot be greater than 65536 when 310P.", sMax_),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF((headDim_ % 16 != 0), OPS_LOG_E(context_->opName, "in 310P, headDim:%u need align to 16.", headDim_),
                   return ge::GRAPH_FAILED);

        OPS_ERR_IF((antiQuantFlag_ && (headDim_ % 32 != 0)),
                   OPS_LOG_E(context_->opName, "in 310P, headDim:%u need align to 32 when kv dtype is int8.", headDim_),
                   return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF((batchSize_ > 65536),
                   OPS_LOG_E(context_->opName, "batch size:%u cannot be greater than 65536.", batchSize_),
                   return ge::GRAPH_FAILED);
    }

    OPS_ERR_IF((headDim_ > 512), OPS_LOG_E(context_->opName, "headDim:%u cannot be greater than 512.", headDim_),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((numKvHeads_ > 256),
               OPS_LOG_E(context_->opName, "numHead of key and value:%u cannot be greater than 256.", numKvHeads_),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVHeadNum(gert::StorageShape *inputShape)
{
    uint32_t tmpNumHeads = 0;
    std::string layOutStr = context_->layOut;
    if (layOutStr == "BSH") {
        auto H = inputShape->GetStorageShape().GetDim(2);
        tmpNumHeads = H / headDim_;
    } else if (layOutStr == "BNSD") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(1);
    } else if (layOutStr == "BSND") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(2);
    }
    OPS_ERR_IF(tmpNumHeads != numKvHeads_,
               OPS_LOG_E(context_->opName, "IFA check input param failed, tensor in list head num(%u) should be %u.",
                         tmpNumHeads, numKvHeads_),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus IFATiling::CheckKVShape()
{
    if (pageAttentionFlag_) {
        return ge::GRAPH_SUCCESS; // page_attention don't check this place
    }
    auto batchOfQuery = context_->query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = context_->key.shape->GetStorageShape().GetDim(0);
    /* kv continuous */
    if (batchOfQuery == batchOfKey) {
        return ge::GRAPH_SUCCESS;
    }
    /* kv not continuous */
    for (int64_t size = 0; size < batchOfQuery; ++size) {
        auto keyTensorInList = context_->kCache[size];
        auto valueTensorInList = context_->vCache[size];
        if ((keyTensorInList == nullptr) || (valueTensorInList == nullptr)) {
            OPS_LOG_E("IncreFlashAttention",
                      "kv tensor list length should be greater than or equal to q batch, "
                      "kv tensor list index[%ld] is null, q batch: %ld",
                      size, batchOfQuery);
            return ge::GRAPH_FAILED;
        }
        std::string layOutStr = context_->layOut;
        if (layOutStr == "BSH") {
            OPS_ERR_IF((keyTensorInList->GetStorageShape().GetDimNum() != DIM_BSH) ||
                           (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BSH),
                       OPS_LOG_E(context_->opName,
                                 "IFA check input param failed, tensor in list dim num should be 3, k: %lu, v: %lu.",
                                 keyTensorInList->GetStorageShape().GetDimNum(),
                                 valueTensorInList->GetStorageShape().GetDimNum()),
                       return ge::GRAPH_FAILED);
        }
        if ((layOutStr == "BNSD") || (layOutStr == "BSND")) {
            OPS_ERR_IF((keyTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BNSD) ||
                           (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BNSD),
                       OPS_LOG_E(context_->opName,
                                 "IFA check input param failed, tensor in list dim num should be 4, k: %lu, v: %lu.",
                                 keyTensorInList->GetStorageShape().GetDimNum(),
                                 valueTensorInList->GetStorageShape().GetDimNum()),
                       return ge::GRAPH_FAILED);
        }
        OPS_ERR_IF(
            keyTensorInList->GetStorageShape().GetDim(0) != 1,
            OPS_LOG_E(
                context_->opName,
                "IFA check input param failed, b of tensor in tensor list should be 1, now b is: %ld, list index: %ld",
                keyTensorInList->GetStorageShape().GetDim(0), size),
            return ge::GRAPH_FAILED);
        if (CheckKVHeadNum(keyTensorInList) != ge::GRAPH_SUCCESS ||
            CheckKVHeadNum(valueTensorInList) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQKOutShape()
{
    if (pageAttentionFlag_) { // page_attention don't check this place
        return ge::GRAPH_SUCCESS;
    }
    // queryShape (b, 1, h)
    const gert::StorageShape *queryShape = context_->query.shape;
    const gert::StorageShape *keyShape = context_->kCache[0];
    const std::string inputLayoutStr = context_->layOut;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = context_->attenOut.shape->GetStorageShape().GetDimNum();
    if (inputLayoutStr == "BSH") {
        OPS_ERR_IF(
            (dimOfQ != DIM_BSH) || (dimOfK != DIM_BSH) || (dimOfOut != DIM_BSH),
            OPS_LOG_E("[IFA]",
                      "When input layout is BSH, the dimension should be 3, dimOfQ: %lu, dimOfK: %lu, dimOfOut: %lu",
                      dimOfQ, dimOfK, dimOfOut),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(queryShape->GetStorageShape().GetDim(1) != 1,
                   OPS_LOG_E("[IFA]", "When input layout is BSH, the 2nd dimOfQ should be 1,the 2nd dimOfQ: %ld",
                             queryShape->GetStorageShape().GetDim(1)),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(queryShape->GetStorageShape().GetDim(2) / numHeads_ !=
                       keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
                   OPS_LOG_E("[IFA]",
                             "When input layout is BSH, the 3rd dimOfQ/numHeads not equal the 3rd dimOfK/numKvHeads, "
                             "3rd dimOfQ/numHeads: %ld, 3rd dimOfK/numKvHeads: %ld",
                             queryShape->GetStorageShape().GetDim(2) / numHeads_,
                             keyShape->GetStorageShape().GetDim(2) / numKvHeads_),
                   return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(
            (dimOfQ != DIM_BNSD_OR_BNSD) || (dimOfK != DIM_BNSD_OR_BNSD) || (dimOfOut != DIM_BNSD_OR_BNSD),
            OPS_LOG_E("[IFA]",
                      "When input layout is BNSD/BSND, the dim should be 4, 4th dimOfQ: %lu, 4th dimOfK: %lu, fourth dimOfOut: %lu",
                      dimOfQ, dimOfK, dimOfOut),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            queryShape->GetStorageShape().GetDim(3) != keyShape->GetStorageShape().GetDim(3),
            OPS_LOG_E(
                "[IFA]",
                "When input layout is BNSD/BSND, the 4th dimOfQ not be equal the 4th dimOfK, dimOfQ: %ld, dimOfK: %ld",
                queryShape->GetStorageShape().GetDim(3), keyShape->GetStorageShape().GetDim(3)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::KvShapePostProcess()
{
    if (pageAttentionFlag_) {
        maxBlockNumPerBatch_ = context_->blockTable.tensor->GetStorageShape().GetDim(1);
        sMax_ = maxBlockNumPerBatch_ * blockSize_;
        seqSize_ = sMax_;
        uint32_t kDimNum = context_->key.shape->GetStorageShape().GetDimNum();
        if (kDimNum == 3U) { // BSH
            inputLayout_ = IfaLayout::BSH_BSND;
        } else { // BNSD
            inputLayout_ = IfaLayout::BNSD;
        }
        const std::string inputLayoutStr = context_->layOut;
        OPS_ERR_IF((kDimNum == DIM_BNSD && inputLayoutStr != "BNSD"),
                   OPS_LOG_E(context_->opName, "when Page Attention scene, kvcache is BNBD, query layout must be BNSD"),
                   return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    for (size_t i = 0; i < context_->kCache.size(); i++) {
        auto keyShape = context_->kCache[i];
        auto valueShape = context_->vCache[i];

        OPS_ERR_IF((keyShape == nullptr || valueShape == nullptr),
                   OPS_LOG_E(context_->opName, "tensor shape of list[%zu] is nullptr", i), return ge::GRAPH_FAILED);

        if (!ShapeEqual(keyShape->GetStorageShape(), valueShape->GetStorageShape())) {
            OPS_LOG_E(context_->opName, "k v shape shoud be same ");
            return ge::GRAPH_FAILED;
        }

        if (CheckKVShape() != ge::GRAPH_SUCCESS ||
            CheckKeyShapeTensor(keyShape->GetStorageShape()) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        uint32_t seqSize;
        if (inputLayout_ == IfaLayout::BSH_BSND) {
            seqSize = keyShape->GetStorageShape().GetDim(1);
        } else {
            seqSize = keyShape->GetStorageShape().GetDim(2); // 2, dim of S
        }

        /* 原则上空tensor为S=0，兼容ShapeSize=0场景 */
        if (seqSize != 0 && keyShape->GetStorageShape().GetShapeSize() == 0) {
            seqSize = 0;
        }

        sMax_ = std::max(seqSize, sMax_);
        kvListSeqLens_.push_back(seqSize);
    }
    seqSize_ = sMax_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ZeroTensorProcess()
{
    if (sMax_ == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        seqSize_ = 1024;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKeyShapeTensor(const gert::Shape &aShape)
{
    auto firstKeyShape = context_->kCache[0];
    std::string layOutStr = context_->layOut;
    for (size_t idx = 0; idx < aShape.GetDimNum(); idx++) {
        if (((layOutStr == "BNSD") && (idx == 2)) || // BNSD s index is 2
            ((layOutStr == "BSND") && (idx == 1)) || // BSND s index is 1
            ((layOutStr == "BSH") && (idx == 1))) {  // BSH s index is 1
            continue;                                // s can be different
        }
        OPS_ERR_IF(firstKeyShape->GetStorageShape().GetDim(idx) != aShape.GetDim(idx),
                   OPS_LOG_E(context_->opName,
                             "IFA check input param failed, tensor in keyShape except S must be same, index:[%lu] is "
                             "not same, k0: %ld, k: %ld",
                             idx, firstKeyShape->GetStorageShape().GetDim(idx), aShape.GetDim(idx)),
                   return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool IFATiling::ShapeEqual(const gert::Shape &aShape, const gert::Shape &bShape)
{
    if (aShape.GetDimNum() != bShape.GetDimNum()) {
        return false;
    }

    for (size_t idx = 0; idx < aShape.GetDimNum(); idx++) {
        if (aShape.GetDim(idx) != bShape.GetDim(idx)) {
            return false;
        }
    }

    return true;
}

bool IFATiling::CanChangeToNew()
{
    if (inOutMode_ == TilingInOutMode::BF16_BF16) {
        return true;
    }
    if (inOutMode_ == TilingInOutMode::BF16_INT8) {
        return true;
    }

    if (inOutMode_ == TilingInOutMode::FP16_FP16 || inOutMode_ == TilingInOutMode::FP16_INT8) {
        return true;
    }
    return false;
}

bool IFATiling::CheckIfRollBack()
{
    if (sMax_ == 0) {
        return false; // 空tensor由新模板处理
    }

    if (socVersion_ != IfaSocVersion::SOC_ASCEND_310P) {
        // 1 page attention
        if (context_->blockTable.tensor != nullptr) {
            return false;
        }
    }

    // 2 qkv_quant
    if (inputQType_ == ge::DT_INT8) {
        return true;
    }

    // 4 D>=1024
    if (headDim_ >= 1024) {
        return true;
    }

    if (CanChangeToNew()) {
        return false;
    }

    return true;
}

ge::graphStatus IFATiling::InitInOutMode()
{
    if (inputQType_ == ge::DT_INT8 && outputType_ == ge::DT_INT8) {
        inOutMode_ = TilingInOutMode::INT8_INT8;
    } else if (inputQType_ == ge::DT_INT8 && outputType_ == ge::DT_FLOAT16) {
        inOutMode_ = TilingInOutMode::INT8_FP16;
    } else if (inputQType_ == ge::DT_FLOAT16 && outputType_ == ge::DT_INT8) {
        inOutMode_ = TilingInOutMode::FP16_INT8;
    } else if (inputQType_ == ge::DT_FLOAT16 && outputType_ == ge::DT_FLOAT16) {
        inOutMode_ = TilingInOutMode::FP16_FP16;
    } else if (inputQType_ == ge::DT_BF16 && outputType_ == ge::DT_BF16) {
        inOutMode_ = TilingInOutMode::BF16_BF16;
    } else if (inputQType_ == ge::DT_BF16 && outputType_ == ge::DT_INT8) {
        inOutMode_ = TilingInOutMode::BF16_INT8;
    } else if (inputQType_ == ge::DT_FLOAT && outputType_ == ge::DT_FLOAT) {
        inOutMode_ = TilingInOutMode::FP32_FP32;
    } else {
        OPS_LOG_E(context_->opName, "input dtype %d with output dtype %d is not currently supported.", inputQType_,
                  outputType_);
        return ge::GRAPH_FAILED;
    }
    if ((socVersion_ == IfaSocVersion::SOC_ASCEND_310P) && (inOutMode_ != TilingInOutMode::FP16_FP16)) {
        OPS_LOG_E(context_->opName,
                  "input dtype float16 with output dtype float16 is currently supported when 310P, but "
                  "current input dtype is %d and output dtype is %d",
                  inputQType_, outputType_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessOptionalTensors()
{
    if ((ProcessActualSeqLen() != ge::GRAPH_SUCCESS) || (ProcessPseShift() != ge::GRAPH_SUCCESS) ||
        (ProcessAttenMask() != ge::GRAPH_SUCCESS) || (ProcessQuant1() != ge::GRAPH_SUCCESS) ||
        (ProcessQuant2() != ge::GRAPH_SUCCESS) || (ProcessDequant1() != ge::GRAPH_SUCCESS) ||
        (ProcessDequant2() != ge::GRAPH_SUCCESS) || (ProcessAntiQuant() != ge::GRAPH_SUCCESS) ||
        (ProcessBlockTable() != ge::GRAPH_SUCCESS) || (ProcessKVPaddingSize() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    // for kv shared prefix
    if ((ProcessSharedPrefix() != ge::GRAPH_SUCCESS) || (ProcessSharedPrefixLen() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessPseShift()
{
    // get pse shift data
    auto pseShiftInput = context_->pseShift.tensor;
    if (pseShiftInput == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OPS_ERR_IF(context_->pseShift.desc == nullptr, OPS_LOG_E(context_->opName, "Desc of pse shift tensor is null."),
               return ge::GRAPH_FAILED);

    auto pseShiftDataType = context_->pseShift.desc->GetDataType();
    if (pseShiftDataType != ge::DT_FLOAT16 && pseShiftDataType != DT_BF16) {
        OPS_LOG_E(context_->opName, "Data type of pse shift is %s, which is not supported.",
                  DataTypeToSerialString(pseShiftDataType).c_str());
        return ge::GRAPH_FAILED;
    }

    switch (pseShiftDataType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            OPS_ERR_IF((inputQType_ != ge::DT_INT8) && (inputQType_ != pseShiftDataType),
                       OPS_LOG_E(context_->opName,
                                 "Data type of pse is %s, which does not match data type of query: %s.",
                                 DataTypeToSerialString(pseShiftDataType).c_str(),
                                 DataTypeToSerialString(inputQType_).c_str()),
                       return ge::GRAPH_FAILED);
            break;
        default:
            OPS_LOG_E(context_->opName, "Data type of pse %s is not currently supported.",
                      DataTypeToSerialString(pseShiftDataType).c_str());
            return ge::GRAPH_FAILED;
    }

    // check pse shift shape (B/1, N, 1, Si)
    const gert::Shape pseShiftShape = pseShiftInput->GetStorageShape();
    uint32_t pseShiftDimNum = pseShiftShape.GetDimNum();
    OPS_ERR_IF(pseShiftDimNum != 4,
               OPS_LOG_E(context_->opName, "The input shape of pse shift must have 4 dims, current dim num is %u.",
                         pseShiftDimNum),
               return GRAPH_FAILED);
    pseShiftBatch_ = pseShiftShape.GetDim(PSE_SHIFT_B);
    uint32_t pseShiftN = pseShiftShape.GetDim(PSE_SHIFT_N);
    uint32_t pseShiftS0 = pseShiftShape.GetDim(PSE_SHIFT_S0);
    pseShiftS1_ = pseShiftShape.GetDim(PSE_SHIFT_S1);

    OPS_ERR_IF(
        (pseShiftBatch_ != 1 && pseShiftBatch_ != batchSize_) || (pseShiftN != numHeads_) || (pseShiftS0 != 1),
        OPS_LOG_E(context_->opName,
                  "The shape of pse shift is (%u, %u, %u, %u), which does not match (B, N, 1, S) or (1, N, 1, S).",
                  pseShiftBatch_, pseShiftN, pseShiftS0, pseShiftS1_),
        return ge::GRAPH_FAILED);

    if (pseShiftS1_ < seqSize_) {
        OPS_LOG_E(context_->opName,
                  "The shape of pse shift is (%u, %u, %u, %u), the 3rd dim S[%u] shouldn't be less than sMax[%u]."
                  "When Page Attention is enabled, sMax is maxBlockNumPerBatch * blockSize.",
                  pseShiftBatch_, pseShiftN, pseShiftS0, pseShiftS1_, pseShiftS1_, seqSize_);
        return GRAPH_FAILED;
    }

    // pse shift D is not 16 aligned
    OPS_ERR_IF(headDim_ % 16 != 0, OPS_LOG_E(context_->opName, "When Pse shift is enabled, D should be 16 aligned."),
               return ge::GRAPH_FAILED);

    pseShiftFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessAttenMask()
{
    auto maskShape = context_->attenMask.tensor; // input shape = 4
    if (maskShape == nullptr) {
        attenMaskFlag_ = false;
        return ge::GRAPH_SUCCESS;
    }

    if (maskShape->GetStorageShape().GetShapeSize() == 0) {
        attenMaskFlag_ = false;
        OPS_LOG_W(context_->opName, "atten_mask tensor exist, but atten_mask shape size is 0.");
        return ge::GRAPH_SUCCESS;
    }

    uint32_t batchSizeOfMask = maskShape->GetStorageShape().GetDim(0);
    if (batchSizeOfMask != batchSize_) {
        OPS_LOG_E(context_->opName, "batchSize[%u] of atten_mask must be equal to batchSize[%u] of query.",
                  batchSizeOfMask, batchSize_);
        return ge::GRAPH_FAILED;
    }

    ge::DataType attenMaskType = context_->attenMask.desc->GetDataType();
    if (attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_INT8 && attenMaskType != ge::DT_UINT8) {
        OPS_LOG_E(context_->opName, "not support atten_mask type %d, only support bool, int8 and uint8.",
                  attenMaskType);
        return ge::GRAPH_FAILED;
    }

    auto dimNumOfMask = maskShape->GetStorageShape().GetDimNum();
    attenMaskSize_ = maskShape->GetStorageShape().GetDim(dimNumOfMask - 1);
    uint32_t minAttenMaskSize = pageAttentionFlag_ ? sMax_ : maxActualseq_;
    if (attenMaskSize_ < minAttenMaskSize) {
        OPS_LOG_E(context_->opName, "s Size[%u] of atten_mask must be greater than or equal to sMax[%u].",
                  attenMaskSize_, minAttenMaskSize);
        return ge::GRAPH_FAILED;
    }

    attenMaskFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessActualSeqLen()
{
    if (context_->actualSeqLengths.tensor == nullptr) {
        maxActualseq_ = sMax_;

        // pa场景必须带actual_seq_lens；第1次tiling调用时(isWorkspace为true)
        // actualSeqLengths会被强制置None，需要跳过校验
        OPS_LOG_D(context_->opName, "isWorkspace: %d", isWorkspace_);
        if (pageAttentionFlag_ && (!isWorkspace_)) {
            OPS_LOG_E(context_->opName,
                      "actual_seq_lengths is null, but actual_seq_lengths must exist in pageAttention scene");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    actualSeqLenFlag_ = true;
    actualLenDims_ = context_->actualSeqLengths.tensor->GetShapeSize();
    if (actualLenDims_ == 0) {
        // pa场景必须带actual_seq_lens
        if (pageAttentionFlag_) {
            OPS_LOG_E(context_->opName, "actual_seq_lengths size[%u] can not be zero in pageAttention scene",
                      actualLenDims_);
            return ge::GRAPH_FAILED;
        }
        maxActualseq_ = sMax_;
        return ge::GRAPH_SUCCESS;
    }

    OPS_ERR_IF(actualLenDims_ != 1 && actualLenDims_ < batchSize_,
               OPS_LOG_E(context_->opName,
                         "actual_seq_lengths size[%u] should be greater than q batch[%u] or equal to 1.",
                         actualLenDims_, batchSize_),
               return ge::GRAPH_FAILED);

    actualLenDims_ = std::min(actualLenDims_, batchSize_);

    const int64_t *actualLenData = context_->actualSeqLengths.tensor->GetData<int64_t>();
    if (actualLenData != nullptr) {
        uint32_t loop = ((actualLenDims_ == 1) && (kvListSeqLens_.size() == 1)) ? 1 : batchSize_;
        for (uint32_t i = 0; i < loop; i++) {
            int64_t actLen = (actualLenDims_ == 1) ? actualLenData[0] : actualLenData[i];
            OPS_ERR_IF(
                actLen < 0, // actualSeqLengths必须大于0
                OPS_LOG_E(context_->opName,
                          "the value of actual_seq_lengths[%u] must be greater than or equal to 0, but it is %ld", i,
                          actLen),
                return ge::GRAPH_FAILED);
            if (!pageAttentionFlag_) {
                uint32_t seqSize = (kvListSeqLens_.size() == 1) ? kvListSeqLens_[0] : kvListSeqLens_[i];
                OPS_ERR_IF(static_cast<uint32_t>(actLen) > seqSize,
                           OPS_LOG_E(context_->opName,
                                     "actual_seq_lengths[%u](%ld) cannot be greater than seq_length(%u) in input key.",
                                     i, actLen, seqSize),
                           return ge::GRAPH_FAILED);
            }
            maxActualseq_ =
                maxActualseq_ < static_cast<uint32_t>(actLen) ? static_cast<uint32_t>(actLen) : maxActualseq_;
        }
    } else {
        // pa场景必须带actual_seq_lens
        if (pageAttentionFlag_ && (!isWorkspace_)) {
            OPS_LOG_E(context_->opName, "data of actual_seq_lengths can not be nullptr in pageAttention scene");
            return ge::GRAPH_FAILED;
        }
        maxActualseq_ = sMax_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessQuant1()
{
    auto dqtScale1 = context_->deqScale1.tensor;
    auto qtScale1 = context_->quantScale1.tensor;
    auto dqtScale2 = context_->deqScale2.tensor;

    if (inputQType_ == ge::DT_INT8) {
        OPS_ERR_IF(dqtScale1 == nullptr,
                   OPS_LOG_E(context_->opName, "when input type is int8, dqtScale1 should not be  null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(qtScale1 == nullptr,
                   OPS_LOG_E(context_->opName, "when input type is int8, qtScale1 should not be null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(dqtScale2 == nullptr,
                   OPS_LOG_E(context_->opName, "when input type is int8, dqtScale2 should not be null"),
                   return ge::GRAPH_FAILED);

        if (dqtScale1->GetShapeSize() != 1 || qtScale1->GetShapeSize() != 1 || dqtScale2->GetShapeSize() != 1) {
            OPS_LOG_E(
                context_->opName,
                "input type is int8, dqtScale1/qtScale1/dqtScale2 size should be 1. But their size are %ld, %ld, %ld",
                dqtScale1->GetShapeSize(), qtScale1->GetShapeSize(), dqtScale2->GetShapeSize());
            return ge::GRAPH_FAILED;
        }
    }

    if (inputQType_ != ge::DT_INT8) {
        OPS_ERR_IF(dqtScale1 != nullptr,
                   OPS_LOG_E(context_->opName, "when input type is not int8, dqtScale1 should be null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(qtScale1 != nullptr,
                   OPS_LOG_E(context_->opName, "when input type is not int8, qtScale1 should be null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(dqtScale2 != nullptr,
                   OPS_LOG_E(context_->opName, "when input type is not int8, dqtScale2 should be null"),
                   return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQuant2Shape(const gert::Shape &inputParaShape)
{
    auto headsize = headDim_; // D
    auto headnum = numHeads_; // Q's N
    gert::Shape expectParamShapeBNSD = gert::Shape({1, headnum, 1, headsize});
    gert::Shape expectParamShapeBNSD_2 = gert::Shape({headnum, 1, headsize});
    gert::Shape expectParamShapeBNSD_3 = gert::Shape({headnum, headsize});
    gert::Shape expectParamShapeBSND = gert::Shape({1, 1, headnum, headsize});
    gert::Shape expectParamShapeBSND_2 = gert::Shape({1, headnum, headsize});
    gert::Shape expectParamShapeBSND_3 = gert::Shape({headnum, headsize});
    gert::Shape expectParamShapeBH = gert::Shape({1, headnum * headsize});
    gert::Shape expectParamShapeBH_2 = gert::Shape({1, 1, headnum * headsize});
    gert::Shape expectParamShapeBH_3 = gert::Shape({headnum * headsize});

    bool validShape = (inputParaShape == expectParamShapeBNSD) || (inputParaShape == expectParamShapeBSND) ||
                      (inputParaShape == expectParamShapeBH) || (inputParaShape == expectParamShapeBNSD_2) ||
                      (inputParaShape == expectParamShapeBSND_2) || (inputParaShape == expectParamShapeBH_2) ||
                      (inputParaShape == expectParamShapeBNSD_3) || (inputParaShape == expectParamShapeBSND_3) ||
                      (inputParaShape == expectParamShapeBH_3);

    if (!validShape && inputParaShape.GetDimNum() == DIM_BNSD) {
        OPS_LOG_E(context_->opName,
                  "The shape of postquant parameter[%ld, %ld, %ld, %ld] is not expected shape."
                  "Expect [1, %u, 1, %u] or [1, 1, %u, %u]",
                  inputParaShape.GetDim(BNSD_B_IDX), inputParaShape.GetDim(BNSD_N_IDX),
                  inputParaShape.GetDim(BNSD_S_IDX), inputParaShape.GetDim(BNSD_D_IDX), headnum, headsize, headnum,
                  headsize);
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == 3) { // dim is 3
        OPS_LOG_E(context_->opName,
                  "The shape of postquant parameter[%ld, %ld, %ld] is not expected shape."
                  "Expect [%u, 1, %u], [1, %u, %u] or [1, 1, %u].",
                  inputParaShape.GetDim(BNSD_B_IDX), inputParaShape.GetDim(BNSD_N_IDX),
                  inputParaShape.GetDim(BNSD_S_IDX), headnum, headsize, headnum, headsize, headnum * headsize);
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == DIM_BH) {
        OPS_LOG_E(context_->opName, "The shape of postquant parameter[%ld, %ld] is not expected[1, %u] or [%u, %u].",
                  inputParaShape.GetDim(BH_B_IDX), inputParaShape.GetDim(BH_H_IDX), headnum * headsize, headnum,
                  headsize);
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == 1) {
        OPS_LOG_E(context_->opName, "The shape of postquant parameter[%ld] is not expected[%u].",
                  inputParaShape.GetDim(BH_B_IDX), headnum * headsize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessQuant2Dtype()
{
    if (outputType_ == ge::DT_INT8) {
        OPS_ERR_IF(context_->quantScale2.tensor == nullptr,
                   OPS_LOG_E(context_->opName, "output data type is int8, but input tensor quantScale2 is null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(context_->quantScale2.desc == nullptr,
                   OPS_LOG_E(context_->opName, "Desc of quantScale2 input tensor is null."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(context_->quantScale2.desc->GetDataType() != ge::DT_BF16 &&
                       context_->quantScale2.desc->GetDataType() != ge::DT_FLOAT,
                   OPS_LOG_E(context_->opName, "quantScale2 type(%d) should be bf16 or fp32",
                             context_->quantScale2.desc->GetDataType()),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(context_->quantOffset2.desc != nullptr &&
                       context_->quantScale2.desc->GetDataType() != context_->quantOffset2.desc->GetDataType(),
                   OPS_LOG_E(context_->opName, "quantScale2 dtype(%d) and quantOffset2 dtype(%d) are not the same",
                             context_->quantScale2.desc->GetDataType(), context_->quantOffset2.desc->GetDataType()),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(inputQType_ != ge::DT_BF16 && context_->quantScale2.desc->GetDataType() == ge::DT_BF16,
                   OPS_LOG_E(context_->opName, "quantScale2 and quantOffset2 support bf16 when inputQ type is bf16"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            inputKvType_ == ge::DT_INT4 && context_->quantScale2.tensor != nullptr,
            OPS_LOG_E(context_->opName, "PostQuant is not supported if Input Kv Dtype is INT4 or INT32 currently."),
            return ge::GRAPH_FAILED);
        if (context_->quantScale2.desc->GetDataType() == ge::DT_BF16) {
            isOutQuantTypeBf16_ = true;
        }
    } else {
        OPS_ERR_IF(context_->quantScale2.tensor != nullptr,
                   OPS_LOG_E(context_->opName, "output data type is not int8, but quantScale2 exist"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(context_->quantOffset2.tensor != nullptr,
                   OPS_LOG_E(context_->opName, "output data type is not int8, but quantOffset2 exist"),
                   return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessQuant2()
{
    auto qtScale2 = context_->quantScale2.tensor;
    auto qtOffset2 = context_->quantOffset2.tensor;
    auto qtScale2Desc = context_->quantScale2.desc;
    auto qtOffset2Desc = context_->quantOffset2.desc;

    if (ProcessQuant2Dtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (outputType_ == ge::DT_INT8) {
        if (qtScale2->GetShapeSize() == 1) {
            OPS_LOG_D(context_->opName, "quant scale2 is a const value.");
        } else {
            OPS_LOG_D(context_->opName, "quant scale2 is a tensor.");
            if (CheckQuant2Shape(qtScale2->GetStorageShape()) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            isOutQuantPerChnOut_ = true;
        }

        // for offset optional
        if (qtOffset2 != nullptr && qtOffset2Desc != nullptr && qtScale2Desc != nullptr) {
            if (qtScale2Desc->GetDataType() != qtOffset2Desc->GetDataType()) {
                OPS_LOG_E(context_->opName, "quant_scale2 and quant_offset2 should have the same data type.");
                return ge::GRAPH_FAILED;
            }
            if (qtOffset2->GetShapeSize() == 1) {
                OPS_LOG_D(context_->opName, "quant offset2 is a const value.");
            } else {
                OPS_LOG_D(context_->opName, "quant offset2 is a tensor.");
                if (CheckQuant2Shape(qtOffset2->GetStorageShape()) != ge::GRAPH_SUCCESS) {
                    OPS_LOG_E(context_->opName, "check quant Offset2 shape failed");
                    return ge::GRAPH_FAILED;
                }
                isOutQuantPerChnOut_ = true;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessDequant1()
{
    if (context_->deqScale1.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessDequant2()
{
    if (context_->deqScale2.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantPerHead(const gert::Shape &inputParaShape)
{
    if (antiquantMode_ == PER_TOKEN_MODE) { // per-token head
        OPS_ERR_IF((inputParaShape.GetDimNum() != 3), // 3: Dim of BGS is 3
                   OPS_LOG_E(context_->opName, "The dim of antiquant should be 3 instead of the current %lu",
                             inputParaShape.GetDimNum()),
                             return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputParaShape.GetDim(0) != batchSize_),
                   OPS_LOG_E(context_->opName, "The 1st dim of antiquant should be %u instead of the current %ld",
                             batchSize_, inputParaShape.GetDim(0)),
                             return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputParaShape.GetDim(1) != numKvHeads_),
                   OPS_LOG_E(context_->opName, "The 2nd dim of antiquant should be %u instead of the current %ld",
                             numKvHeads_, inputParaShape.GetDim(1)),
                             return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputParaShape.GetDim(2) < seqSize_),
                   OPS_LOG_E(context_->opName, "The 3rd dim of antiquant should bigger than or equal to %u instead of the current %ld",
                             seqSize_, inputParaShape.GetDim(2)), // 2 : BGS S index is 2
                   return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    } else { // per-tensor head
        gert::Shape expectParamShape = gert::Shape({numKvHeads_});
        OPS_ERR_IF((inputParaShape != expectParamShape),
                   OPS_LOG_E(context_->opName,
                             "The shape of antiquant parameter[%ld] is not expected. Expect[%u] When per_tensor_head mode.",
                             inputParaShape.GetDim(0), numKvHeads_),
               return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus IFATiling::CheckKVAntiQuantParamsShapeInPagedAttention(const gert::Shape &inputParaShape) {
    OPS_ERR_IF((inputParaShape.GetDim(0) != totalBlockNum_),
            OPS_LOG_E(context_->opName,
                        "The 1st dim of antiquant parameter should be %u instead of the current %ld",
                        totalBlockNum_, inputParaShape.GetDim(0)),
            return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        (inputParaShape.GetDim(1) != blockSize_),
        OPS_LOG_E(context_->opName,
                "The 2nd dim of antiquant parameter should be %u instead of the current %ld",
                blockSize_, inputParaShape.GetDim(1)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantParamsInPagedAttention() {
    auto keyAntiquantScaleTensor = context_->keyAntiquantScale.tensor;
    auto KeyAntiquantScaleShape = keyAntiquantScaleTensor->GetStorageShape();
    if (CheckKVAntiQuantParamsShapeInPagedAttention(KeyAntiquantScaleShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto keyAntiquantOffsetTensor = context_->keyAntiquantOffset.tensor;
    if (keyAntiquantOffsetTensor != nullptr) {
        auto KeyAntiquantOffsetShape = keyAntiquantOffsetTensor->GetStorageShape();
        if (CheckKVAntiQuantParamsShapeInPagedAttention(KeyAntiquantOffsetShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantMode() {
    if ((antiquantMode_ != DEQUANT_PER_CHANNEL_MODE) &&
            (antiquantMode_ != DEQUANT_PER_TOKEN_MODE) &&
            (antiquantMode_ != DEQUANT_PER_TENSOR_HEAD_MODE) && 
            (antiquantMode_ != DEQUANT_PER_TOKEN_HEAD_MODE) && 
            (antiquantMode_ != DEQUANT_PER_TOKEN_PA_MODE)) {
        OPS_LOG_E(context_->opName,
            "antiquantMode value:%u is invalid, it should be 0、1、2、3 or 4", antiquantMode_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantPerToken(const gert::Shape &inputParaShape)
{
    if (inputParaShape.GetDimNum() == DIM_PER_TOKEN) {
        OPS_ERR_IF((inputParaShape.GetDim(PER_TOKEN_N) != antiquantNum_),
                   OPS_LOG_E(context_->opName, "The 1st dim of antiquant should be %u instead of the current %ld",
                             antiquantNum_, inputParaShape.GetDim(PER_TOKEN_N)),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputParaShape.GetDim(PER_TOKEN_B) != batchSize_),
                   OPS_LOG_E(context_->opName, "The 2nd dim of antiquant should be %u instead of the current %ld",
                             batchSize_, inputParaShape.GetDim(PER_TOKEN_B)),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            (inputParaShape.GetDim(PER_TOKEN_S) < seqSize_),
            OPS_LOG_E(context_->opName,
                      "The 3rd dim of antiquant should be greater than or equal to %u instead of the current %ld",
                      seqSize_, inputParaShape.GetDim(PER_TOKEN_S)),
            return ge::GRAPH_FAILED);
    } else if (inputParaShape.GetDimNum() == DIM_PER_TOKEN_KvSplit && kvAntiParamSplitFlag_) {
        if (!antiquantParamsInPagedAttentionFlag_) {
            // 使用pa模式管理scale/offset时，scale/offset形状有变化，屏蔽原有校验
            OPS_ERR_IF((inputParaShape.GetDim(PER_TOKEN_Split_B) != batchSize_),
                    OPS_LOG_E(context_->opName,
                                "The 1st dim of antiquant should be %u instead of the current %ld",
                                batchSize_, inputParaShape.GetDim(PER_TOKEN_B)),
                    return ge::GRAPH_FAILED);
            OPS_ERR_IF(
                (inputParaShape.GetDim(PER_TOKEN_Split_S) < seqSize_),
                OPS_LOG_E(context_->opName,
                        "The 2nd dim of antiquant should be greater than or equal to %u instead of the current %ld",
                        seqSize_, inputParaShape.GetDim(PER_TOKEN_S)),
                return ge::GRAPH_FAILED);
        }
    } else {
        OPS_LOG_E(context_->opName, "The dim of antiquant is illegal, When per_token mode.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantPerChannel(const gert::Shape& inputParaShape) {
  std::string layOutStr = context_->layOut;
  gert::Shape expectParamShapeBNSD = gert::Shape({antiquantNum_, numKvHeads_, 1, headDim_});
  gert::Shape expectParamShapeBSNDType1 = gert::Shape({antiquantNum_, 1, numKvHeads_, headDim_});
  gert::Shape expectParamShapeBSNDType2 = gert::Shape({antiquantNum_, numKvHeads_, headDim_});
  gert::Shape expectParamShapeBH = gert::Shape({antiquantNum_, numKvHeads_ * headDim_});
  bool validOffsetShape = (inputParaShape == expectParamShapeBNSD) || (inputParaShape == expectParamShapeBSNDType1) ||
                          (inputParaShape == expectParamShapeBSNDType2) || (inputParaShape == expectParamShapeBH);

  OPS_ERR_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BNSD),
    OPS_LOG_E(context_->opName, "The shape of antiquant parameter[%ld, %ld, %ld, %ld] is not expected. "
      "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BNSD_B_IDX),
      inputParaShape.GetDim(BNSD_N_IDX), inputParaShape.GetDim(BNSD_S_IDX), inputParaShape.GetDim(BNSD_D_IDX),
      antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_ * headDim_),
      return ge::GRAPH_FAILED);
  OPS_ERR_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BSND),
    OPS_LOG_E(context_->opName, "The shape of antiquant parameter[%ld, %ld, %ld] is not expected. "
      "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BND_B_IDX),
      inputParaShape.GetDim(BND_N_IDX), inputParaShape.GetDim(BND_D_IDX), antiquantNum_, numKvHeads_, headDim_,
      antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_ * headDim_),
      return ge::GRAPH_FAILED);
  OPS_ERR_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_BH),
    OPS_LOG_E(context_->opName, "The shape of antiquant parameter[%ld, %ld] is not expected. "
      "Expect [%u, %u, 1, %u] or [%u, %u, %u] or [%u, %u] when per_channel mode.", inputParaShape.GetDim(BH_B_IDX),
      inputParaShape.GetDim(BH_H_IDX), antiquantNum_, numKvHeads_, headDim_, antiquantNum_, numKvHeads_, headDim_,
      antiquantNum_, numKvHeads_ * headDim_),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantParaShapeLegal(const gert::Shape &inputParaShape)
{
    if (kvAntiParamSplitFlag_) {
        antiquantNum_ = 1;
    }
    gert::Shape expectParamShapePerTensor = gert::Shape({antiquantNum_});
    if (antiquantPerHeadFlag_) {
        return CheckKVAntiQuantPerHead(inputParaShape);
    }
    if (antiquantMode_ == PER_TOKEN_MODE) { // per-token
        return CheckKVAntiQuantPerToken(inputParaShape);
    } else if (inputParaShape.GetDimNum() == DIM_PER_TENSOR) { // per-tensor
        antiquantMode_ = PER_CHANNEL_MODE;
        antiquantPerTensorFlag_ = 1;
        OPS_ERR_IF((inputParaShape != expectParamShapePerTensor),
                   OPS_LOG_E(context_->opName,
                             "The shape of antiquant parameter[%ld] is not expected. Expect[%u] When per_tensor mode.",
                             inputParaShape.GetDim(BH_B_IDX), antiquantNum_),
                   return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    } else if (inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BNSD ||
               inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BSND ||
               inputParaShape.GetDimNum() == DIM_BH) { // per-channel
        return CheckKVAntiQuantPerChannel(inputParaShape);
    } else {
        OPS_LOG_E(context_->opName, "The layout[%lu] does not match the dim of antiquant, When per_channel mode.",
                  inputParaShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckAntiQuantParam(const gert::Tensor *antiquantScaleTensor,
                                               const gert::Tensor *antiquantOffsetTensor,
                                               const gert::CompileTimeTensorDesc *antiquantScaleDesc,
                                               const gert::CompileTimeTensorDesc *antiquantOffsetDesc)
{
    OPS_ERR_IF((antiquantMode_ != 0) && (antiquantMode_ != 1), // unseparated antiquant need this
               OPS_LOG_E(context_->opName, "antiquantMode value:%u is invalid, it should be 0 or 1", antiquantMode_),
               return ge::GRAPH_FAILED);
    if (antiquantScaleTensor == nullptr) {
        OPS_LOG_E(context_->opName, "KV antiquant is enabled, but the input antiquant scale is NULL");
        return ge::GRAPH_FAILED;
    }
    if (antiquantOffsetTensor != nullptr &&
        antiquantScaleTensor->GetStorageShape().GetDimNum() != antiquantOffsetTensor->GetStorageShape().GetDimNum()) {
        OPS_LOG_E(context_->opName,
                  "KV antiquant is enabled, but antiquant params have different layouts[scale: %lu, offset: %lu].",
                  antiquantScaleTensor->GetStorageShape().GetDimNum(),
                  antiquantOffsetTensor->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    auto tmpAntiquantScale = antiquantScaleTensor->GetStorageShape();
    if (CheckKVAntiQuantParaShapeLegal(tmpAntiquantScale) == ge::GRAPH_FAILED) {
        OPS_LOG_E(context_->opName, "illegal shape of antiquant scale.");
        return ge::GRAPH_FAILED;
    }
    if (antiquantOffsetTensor != nullptr) {
        auto tmpAntiquantOffset = antiquantOffsetTensor->GetStorageShape();
        if (CheckKVAntiQuantParaShapeLegal(tmpAntiquantOffset) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }

    ge::DataType antiquantScaleType = antiquantScaleDesc->GetDataType();
    if (antiquantMode_ == DEQUANT_PER_CHANNEL_MODE) { // per-tensor and per-channel
        if (antiquantScaleType != inputQType_) {
            OPS_LOG_E(context_->opName, "illegal datatype of antiquant scale, it should be same with input qtype");
            return ge::GRAPH_FAILED;
        }
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_MODE) {
        if (antiquantScaleType != ge::DT_FLOAT) {
            OPS_LOG_E(context_->opName, "per-token mode is enabled, datatype of antiquant scale should be float32 ");
            return ge::GRAPH_FAILED;
        }
    }

    if (antiquantOffsetTensor != nullptr && antiquantOffsetDesc != nullptr) {
        ge::DataType antiquantOffsetType = antiquantOffsetDesc->GetDataType();
        if (antiquantScaleType != antiquantOffsetType) {
            OPS_LOG_E(context_->opName, "datatype of antiquant scale and antiquant offset should be the same");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessAntiQuant()
{
    auto antiquantScaleTensor = context_->antiquantScale.tensor;
    auto antiquantScaleDesc = context_->antiquantScale.desc;
    auto antiquantOffsetTensor = context_->antiquantOffset.tensor;
    auto antiquantOffsetDesc = context_->antiquantOffset.desc;
    auto keyAntiquantScaleTensor = context_->keyAntiquantScale.tensor;
    auto keyAntiquantScaleDesc = context_->keyAntiquantScale.desc;
    auto keyAntiquantOffsetTensor = context_->keyAntiquantOffset.tensor;
    auto keyAntiquantOffsetDesc = context_->keyAntiquantOffset.desc;
    auto valueAntiquantScaleTensor = context_->valueAntiquantScale.tensor;
    auto valueAntiquantOffsetTensor = context_->valueAntiquantOffset.tensor;
    auto valueAntiquantOffsetDesc = context_->valueAntiquantOffset.desc;
    if (!antiQuantFlag_ && (antiquantScaleTensor != nullptr || antiquantOffsetTensor != nullptr ||
                            keyAntiquantScaleTensor != nullptr || keyAntiquantOffsetTensor != nullptr ||
                            valueAntiquantScaleTensor != nullptr || valueAntiquantOffsetTensor != nullptr)) {
        OPS_LOG_E(context_->opName, "KV antiquant is unenabled, but antiquant antiquantScale/antiquantOffset exist");
        return ge::GRAPH_FAILED;
    }

    if (!antiQuantFlag_) {
        return ge::GRAPH_SUCCESS;
    }
    kvAntiParamSplitFlag_ = false;
    if (keyAntiquantScaleTensor != nullptr && valueAntiquantScaleTensor == nullptr) {
        OPS_LOG_E(context_->opName, "valueAntiquantScaleTensor is null, but keyAntiquantScaleTensor exist");
        return ge::GRAPH_FAILED;
    }
    if (valueAntiquantScaleTensor != nullptr && keyAntiquantScaleTensor == nullptr) {
        OPS_LOG_E(context_->opName, "keyAntiquantScaleTensor is null, but valueAntiquantScaleTensor exist");
        return ge::GRAPH_FAILED;
    }
    if (keyAntiquantOffsetTensor != nullptr && valueAntiquantOffsetTensor == nullptr) {
        OPS_LOG_E(context_->opName, "valueAntiquantOffsetTensor is null, but keyAntiquantOffsetTensor exist");
        return ge::GRAPH_FAILED;
    }
    if (valueAntiquantOffsetTensor != nullptr && keyAntiquantOffsetTensor == nullptr) {
        OPS_LOG_E(context_->opName, "keyAntiquantOffsetTensor is null, but valueAntiquantOffsetTensor exist");
        return ge::GRAPH_FAILED;
    }
    if (keyAntiquantScaleTensor == nullptr && keyAntiquantOffsetTensor != nullptr) {
        OPS_LOG_E(context_->opName, "keyAntiquantScaleTensor is null, but keyAntiquantOffsetTensor exist");
        return ge::GRAPH_FAILED;
    }
    if (keyAntiquantOffsetTensor != nullptr && valueAntiquantOffsetTensor != nullptr) {
        OPS_ERR_IF((keyAntiquantOffsetDesc == nullptr),
                   OPS_LOG_E(context_->opName, "keyAntiquantScaleTensor isn't nullptr, keyAntiquantOffsetDesc is null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            (valueAntiquantOffsetDesc == nullptr),
            OPS_LOG_E(context_->opName, "valueAntiquantScaleTensor isn't nullptr, valueAntiquantOffsetDesc is null"),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF((keyAntiquantOffsetDesc->GetDataType() != valueAntiquantOffsetDesc->GetDataType()),
                   OPS_LOG_E(context_->opName,
                             "valueAntiquantScaleDesc and keyAntiquantScaleDesc should have the same data type"),
                   return ge::GRAPH_FAILED);
        if (!ShapeEqual(keyAntiquantOffsetTensor->GetStorageShape(), valueAntiquantOffsetTensor->GetStorageShape())) {
            OPS_LOG_E(context_->opName,
                      "keyAntiquantOffsetTensor and valueAntiquantOffsetTensor should have the same shape");
            return ge::GRAPH_FAILED;
        }
    }
    if (keyAntiquantScaleTensor != nullptr && valueAntiquantScaleTensor != nullptr) {
        if (!ShapeEqual(keyAntiquantScaleTensor->GetStorageShape(), valueAntiquantScaleTensor->GetStorageShape())) {
            OPS_LOG_E(context_->opName,
                      "keyAntiquantScaleTensor and valueAntiquantScaleTensor should have the same shape");
            return ge::GRAPH_FAILED;
        }
        kvAntiParamSplitFlag_ = true;
    }
    if (kvAntiParamSplitFlag_) {
        OPS_LOG_D(context_->opName, "kv antiquant is split mode");
        uint32_t keyAntiquantMode = context_->keyAntiquantMode != nullptr ? *context_->keyAntiquantMode : 0;
        uint32_t valueAntiquantMode = context_->valueAntiquantMode != nullptr ? *context_->valueAntiquantMode : 0;
        if (keyAntiquantMode != valueAntiquantMode) {
            OPS_LOG_E(context_->opName, "keyAntiquantMode and valueAntiquantMode should be the same");
            return ge::GRAPH_FAILED;
        }
        antiquantMode_ = keyAntiquantMode;
        OPS_LOG_D(context_->opName, "org antiquantMode value:%u", antiquantMode_);
        if(CheckKVAntiQuantMode() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (antiquantMode_ == DEQUANT_PER_TENSOR_HEAD_MODE) { // 2:per tensor head
          antiquantMode_ = PER_CHANNEL_MODE;
          antiquantPerHeadFlag_ = 1;
        }
        if (antiquantMode_ == DEQUANT_PER_TOKEN_HEAD_MODE) { // 3:per token head
          antiquantMode_ = PER_TOKEN_MODE;
          antiquantPerHeadFlag_ = 1;
        }
        if (antiquantMode_ == DEQUANT_PER_TOKEN_PA_MODE) { // 4:per token + pageAttention scale/offset
          antiquantMode_ = PER_TOKEN_MODE;
          antiquantParamsInPagedAttentionFlag_ = 1;
          OPS_ERR_IF(!pageAttentionFlag_,
                     OPS_LOG_E(context_->opName,
                        "keyAntiquantMode/valueAntiquantMode 4 use page attention to manage scale/offset, must be used in page attention scene"),
                     return ge::GRAPH_FAILED);
        }
        if (CheckAntiQuantParam(keyAntiquantScaleTensor, keyAntiquantOffsetTensor, keyAntiquantScaleDesc,
                                keyAntiquantOffsetDesc) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OPS_LOG_D(context_->opName, "kv antiquant is not split mode");
        if (context_->antiquantMode != nullptr) {
            antiquantMode_ = *context_->antiquantMode;
        }
        if (CheckAntiQuantParam(antiquantScaleTensor, antiquantOffsetTensor, antiquantScaleDesc, antiquantOffsetDesc) ==
            ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    antiqSeqSize_ = GetAntiquantSeqLength();
    OPS_LOG_D(context_->opName, "antiquant info, iter num:%u, antiquant mode:%u", msdIterNum_, antiquantMode_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessBlockTable()
{
    if (!pageAttentionFlag_) {
        return ge::GRAPH_SUCCESS;
    }

    // gm到l1，copynd2nz的srcDValue最大支持65535
    if ((inputLayout_ == IfaLayout::BSH_BSND) && (numKvHeads_ * headDim_ > COPYND2NZ_SRC_STRIDE_LIMITATION)) { // 0: BSH
        OPS_LOG_E(context_->opName,
                  "When input kvcache layout is BSH, the N * D of kvcache is %u, "
                  "exceeds the maximum limit (%u) of the datacopy instruction.",
                  numKvHeads_ * headDim_, COPYND2NZ_SRC_STRIDE_LIMITATION);
        return ge::GRAPH_FAILED;
    }

    if (CheckPABlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    totalBlockNum_ = context_->kCache[0]->GetStorageShape().GetDim(0);
    OPS_ERR_IF(maxActualseq_ > blockSize_ * maxBlockNumPerBatch_,
               OPS_LOG_E(context_->opName,
                         "Invalid actual seq length for PA, max actual seq length(%u) "
                         "is larger than blocksize(%u) * max block num per batch(%u)",
                         maxActualseq_, blockSize_, maxBlockNumPerBatch_),
               return ge::GRAPH_FAILED);

    if ((antiquantMode_ == PER_TOKEN_MODE) && antiquantParamsInPagedAttentionFlag_) {
        // 在处理pa相关信息时，才能获取到totalBlockNum_用于scale/offset校验
        if (CheckKVAntiQuantParamsInPagedAttention() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessKVPaddingSize()
{
    auto kvPaddingSize = context_->kvPaddingSize.tensor;
    if (kvPaddingSize == nullptr) {
        OPS_LOG_D(context_->opName, "KVLeftPadding illegal condition: kvPaddingSize.tensor is nullptr: %d",
                  context_->kvPaddingSize.tensor == nullptr);
        return ge::GRAPH_SUCCESS;
    }

    if (kvPaddingSize->GetStorageShape().GetShapeSize() == 0) {
        OPS_LOG_D(context_->opName, "KVLeftPadding illegal condition: kvPaddingSize.tensor shape is empty: %d",
                  kvPaddingSize->GetStorageShape().GetShapeSize() == 0);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ret = CheckSupportKVLeftPadding();

    return ret;
}

ge::graphStatus IFATiling::CheckSupportKVLeftPadding()
{
    if (inputKvType_ == ge::DT_INT4) {
        OPS_LOG_E(context_->opName, "When input Kv Dtypes is INT4 or INT32, KvLeftPadding is not supported currently.");
        return ge::GRAPH_FAILED;
    }
    if (!batchContinuousFlag_ || !actualSeqLenFlag_ || pageAttentionFlag_) {
        OPS_LOG_D(context_->opName, "KVLeftPadding illegal condition:  \
      pagedAttention scene: %d, not isBatchContinues: %d, actualSeqLen not exist: %d.",
                  pageAttentionFlag_, !batchContinuousFlag_, !actualSeqLenFlag_);
        return ge::GRAPH_SUCCESS;
    }
    kvPaddingSizeFlag_ = true;
    OPS_LOG_D(context_->opName, "KVLeftPadding starts to be used.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessSharedPrefix()
{
    if (context_->keySharedPrefix.tensor == nullptr && context_->valueSharedPrefix.tensor == nullptr) {
        sysPrefixFlag_ = false;
        return ge::GRAPH_SUCCESS;
    }

    if (SharedPrefixCheckBasic() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto keyShape = context_->keySharedPrefix.tensor->GetStorageShape();
    auto valueShape = context_->valueSharedPrefix.tensor->GetStorageShape();
    if (SharedPrefixCheckShapes(keyShape, valueShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        sMaxPrefix_ = keyShape.GetDim(1); // 1:BSH的S维
    } else {
        sMaxPrefix_ = keyShape.GetDim(2); // 2:BNSD的S维
    }

    if (keyShape.GetShapeSize() == 0) { // 兼容空tensor场景
        sMaxPrefix_ = 0;
    }

    sysPrefixFlag_ = true;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SharedPrefixCheckBasic()
{
    OPS_ERR_IF(context_->keySharedPrefix.tensor == nullptr,
               OPS_LOG_E(context_->opName, "tensor  of key_shared_prefix is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->keySharedPrefix.desc == nullptr,
               OPS_LOG_E(context_->opName, "desc  of key_shared_prefix is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->valueSharedPrefix.tensor == nullptr,
               OPS_LOG_E(context_->opName, "tensor of value_shared_prefix is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->valueSharedPrefix.desc == nullptr,
               OPS_LOG_E(context_->opName, "desc  of value_shared_prefix is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->keySharedPrefix.desc->GetDataType() != inputKvType_,
               OPS_LOG_E(context_->opName, "type of key_shared_prefix not equal to type of KV"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->valueSharedPrefix.desc->GetDataType() != inputKvType_,
               OPS_LOG_E(context_->opName, "type of value_shared_prefix not equal to type of KV"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(pageAttentionFlag_, OPS_LOG_E(context_->opName, "shared prefix with page attention is not supported"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(kvPaddingSizeFlag_, OPS_LOG_E(context_->opName, "shared prefix with kv padding is not supported"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(socVersion_ == IfaSocVersion::SOC_ASCEND_310P,
               OPS_LOG_E(context_->opName, "shared prefix is not supported on 310p"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SharedPrefixCheckShapes(const gert::Shape &keyShape, const gert::Shape &valueShape)
{
    OPS_ERR_IF(!ShapeEqual(keyShape, valueShape),
               OPS_LOG_E(context_->opName, "tensor shape of key_shared_prefix and value_shared_prefix not equal."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(keyShape.GetDimNum() != context_->query.shape->GetStorageShape().GetDimNum(),
               OPS_LOG_E(context_->opName, "tensor shape dim of key_shared_prefix[%lu] is not compatable with query",
                         keyShape.GetDimNum()),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(keyShape.GetDim(0) != 1,
               OPS_LOG_E(context_->opName, "batch of key_shared_prefix[%ld] must be 1", keyShape.GetDim(0)),
               return ge::GRAPH_FAILED);

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        OPS_ERR_IF(
            keyShape.GetDimNum() == 3 && keyShape.GetDim(2) != numKvHeads_ * headDim_,
            OPS_LOG_E(context_->opName, "H of key_shared_prefix[%lu] is not equal to H of key", keyShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            keyShape.GetDimNum() == 4 && keyShape.GetDim(2) != numKvHeads_,
            OPS_LOG_E(context_->opName, "N of key_shared_prefix[%lu] is not equal to N of key", keyShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            keyShape.GetDimNum() == 4 && keyShape.GetDim(3) != headDim_,
            OPS_LOG_E(context_->opName, "D of key_shared_prefix[%lu] is not equal to D of key", keyShape.GetDimNum()),
            return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(
            keyShape.GetDim(1) != numKvHeads_,
            OPS_LOG_E(context_->opName, "N of key_shared_prefix[%ld] is not equal to N of key", keyShape.GetDim(1)),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            keyShape.GetDim(3) != headDim_,
            OPS_LOG_E(context_->opName, "D of key_shared_prefix[%ld] is not equal to D of key", keyShape.GetDim(3)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

uint32_t IFATiling::GetAntiquantSeqLength()
{
    if (antiquantParamsInPagedAttentionFlag_) {
        return seqSize_;
    }
    const size_t antiquantSIdx = 2;
    return kvAntiParamSplitFlag_ ? context_->keyAntiquantScale.tensor->GetStorageShape().GetDim(antiquantSIdx) :
                                   context_->antiquantScale.tensor->GetStorageShape().GetDim(antiquantSIdx);
}

ge::graphStatus IFATiling::ProcessSharedPrefixLen()
{
    auto tensor = context_->actualSharedPrefixLen.tensor;
    if (tensor == nullptr || tensor->GetStorageShape().GetShapeSize() == 0 || !sysPrefixFlag_) {
        maxActualPrefixLen_ = sMaxPrefix_;
        return ge::GRAPH_SUCCESS;
    }

    maxActualPrefixLen_ = sMaxPrefix_;
    auto actulLenShape = context_->actualSharedPrefixLen.tensor->GetStorageShape();

    OPS_ERR_IF((actulLenShape.GetDimNum() != 1 || actulLenShape.GetDim(0) != 1),
               OPS_LOG_E(context_->opName, "actual shared prefix shape[%lu] must be 1", actulLenShape.GetDimNum()),
               return ge::GRAPH_FAILED);

    actualLenDimsPrefix_ = 1;
    const int64_t *actualLenData = context_->actualSharedPrefixLen.tensor->GetData<int64_t>();
    if (actualLenData != nullptr) {
        maxActualPrefixLen_ = actualLenData[0];
        OPS_ERR_IF(maxActualPrefixLen_ > sMaxPrefix_,
                   OPS_LOG_E(context_->opName, "actual prefix len[%u] should not be larger than S[%u] of prefix tensor",
                             maxActualPrefixLen_, sMaxPrefix_),
                   return ge::GRAPH_FAILED);
    }

    uint32_t totalS = maxActualPrefixLen_ + maxActualseq_;
    if (pseShiftFlag_) { // 存在pse时才校验
        OPS_ERR_IF((totalS > pseShiftS1_),
                   OPS_LOG_E(context_->opName, "total kv S Size (with shared prefix)[%u] bigger than pseShift size[%u]",
                             totalS, pseShiftS1_),
                   return ge::GRAPH_FAILED);
    }

    if (attenMaskFlag_) { // 存在attenMask时才校验
        OPS_ERR_IF((totalS > attenMaskSize_),
                   OPS_LOG_E(context_->opName,
                             "total kv S Size (with shared prefix)[%u] bigger than attenMask size[%u]", totalS,
                             attenMaskSize_),
                   return ge::GRAPH_FAILED);
    }

    if (antiquantMode_ == PER_TOKEN_MODE) {
        uint32_t perTokenSize = GetAntiquantSeqLength();
        OPS_ERR_IF((totalS > perTokenSize),
                   OPS_LOG_E(context_->opName,
                             "total kv S Size (with shared prefix)[%u] bigger than antiquant perToken size[%u]", totalS,
                             perTokenSize),
                   return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

bool IFATiling::IsFlashDecode() const
{
    if (pageAttentionFlag_ && socVersion_ == IfaSocVersion::SOC_ASCEND_910B) {
        return false;
    }

    float flashDecodeBNRatio = static_cast<float>(0.4); // 0.4, 经验值
    if (perfMode_ == IfaPerfMode::BMM_ALL_BY_VEC) {
        flashDecodeBNRatio = 0.5; // 0.5, 全V模板可以按0.5切分
    }

    if ((batchSize_ * numKvHeads_ < flashDecodeBNRatio * coreNum_) && (nNumOfQInOneGroup_ == 1)) {
        OPS_LOG_D(context_->opName, "Flash Decode Split kv."); // 非gqa时这里只判断bn是否满足
        return true;
    }

    if ((batchSize_ * numKvHeads_ < flashDecodeBNRatio * coreNum_) &&
        (maxActualseq_ >= 2048)) { // 2048, 在flash decode + gqa时的经验值
        OPS_LOG_D(context_->opName, "Flash Decode And GQA Split kv.");
        return true;
    }
    return false;
}

ge::graphStatus IFATiling::Split()
{
    if (IsFlashDecode()) {
        splitKVFlag_ = true;
        kvSplit_++;
        return SplitBNS();
    }

    CalcInnerSize(seqSize_);
    return SplitBN();
}

ge::graphStatus IFATiling::SplitBN()
{
    uint32_t bn = batchSize_ * numKvHeads_;

    for (uint32_t i = 0; i < MAX_CORE_NUM; i++) {
        startIdxEachCore_[i] = bn;
    }

    if (isSysPrefixTiling_) {
        return SplitBN_V0();
    }

    if (actualLenDims_ == 1 || bn <= coreNum_ || (actualLenDims_ == 0 && kvListSeqLens_.size() == 1)) {
        return SplitBN_V0();
    }

    std::vector<int64_t> validArray;
    if (actualLenDims_ > 0) {
        const int64_t *actualLenData = context_->actualSeqLengths.tensor->GetData<int64_t>();
        validArray = InitSparseValidArray(actualLenData);
    } else {
        validArray = InitSparseValidArray(&kvListSeqLens_[0]);
    }

    SetSparseStartIdx(validArray, bn, coreNum_, startIdxEachCore_, CeilDivision(bn, coreNum_));

    usedCoreNum_ = coreNum_;
    return ge::GRAPH_SUCCESS;
}

std::vector<int64_t> IFATiling::InitSparseValidArray(const int64_t *actualLens)
{
    std::vector<int64_t> res((batchSize_ * numKvHeads_));
    for (uint32_t b = 0; b < batchSize_; b++) {
        for (uint32_t n = 0; n < numKvHeads_; n++) {
            int64_t estimatedLoad = seqSize_;
            if (actualLens != nullptr) {
                estimatedLoad = actualLens[b];
                if (antiQuantFlag_ && estimatedLoad < MSD_VEC_LOAD) {
                    estimatedLoad = MSD_VEC_LOAD;
                } else if (actualLens[b] == 0) {
                    // 空tensor输出也计入负载，否则当最后一个batch为空tensor时，分核算法会将该batch优化掉
                    estimatedLoad = 1;
                }
            }
            res[b * numKvHeads_ + n] = estimatedLoad;
        }
    }
    return res;
}
// code copy from flash_attention_score_tiling
bool IFATiling::BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                            std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single
    // core
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return (tmpMaxVal >= maxVal) ? false : true;
}

void IFATiling::InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                              const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue)
{
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0);
        } else {
            break;
        }
    }
}

void IFATiling::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                                  uint32_t *sparseStartIdx, int64_t splitFactorSize)
{
    // initLoad: 使用均分策略, 保证后续不会比均分差
    std::vector<int64_t> localSparseStartIdx(MAX_CORE_NUM, totalSize);
    for (int64_t idx = 0; idx < MAX_CORE_NUM; ++idx) {
        localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
    }
    std::vector<int64_t> localValue(validAivNum, 0);
    InitLoadValue(sparseValidArray, totalSize, validAivNum, localSparseStartIdx, localValue);

    // 负载均衡粗调
    std::vector<int64_t> tmpLocalValue(validAivNum, 0);
    std::vector<int64_t> tmpsparseStartIdx(MAX_CORE_NUM, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
    int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

    tmpsparseStartIdx[0] = 0;
    for (int64_t idx = 1; idx < MAX_CORE_NUM; ++idx) {
        int64_t start = tmpsparseStartIdx[idx - 1];
        int64_t singleLoadValue = 0;
        tmpsparseStartIdx[idx] = start;
        while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
            singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
            tmpsparseStartIdx[idx] += 1;
        }

        if ((start + 1) < tmpsparseStartIdx[idx]) {
            int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
            if ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) {
                tmpsparseStartIdx[idx] = tmpsparseStartIdx[idx] - 1;
                singleLoadValue = redoSingleLoadValue;
            }
            sparseArraySum -= singleLoadValue;
            avgVal = CeilDivision(sparseArraySum, (validAivNum - idx));
        }
    }

    InitLoadValue(sparseValidArray, totalSize, validAivNum, tmpsparseStartIdx, tmpLocalValue);

    // 负载均衡精调
    while (BalanceLoad(sparseValidArray, totalSize, validAivNum, tmpLocalValue, tmpsparseStartIdx)) {
        // 根据负载均衡是否能得到更好预测结果决定是否结束循环
    }

    // exchange initLoad and 负载均衡
    if ((*std::max_element(localValue.begin(), localValue.end())) >
        (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
        localSparseStartIdx.swap(tmpsparseStartIdx);
        localValue.swap(tmpLocalValue);
    }
    for (int64_t idx = 0; idx < MAX_CORE_NUM; ++idx) {
        sparseStartIdx[idx] = localSparseStartIdx[idx];
    }
}

ge::graphStatus IFATiling::SplitBN_V0()
{
    uint32_t bn = batchSize_ * numKvHeads_;
    formerCoreNum_ = bn % coreNum_;
    if (formerCoreNum_ == 0) {
        blockSplitBn2Range_ = bn / coreNum_;
        tailSplitedBatchRange_ = blockSplitBn2Range_;
    } else {
        blockSplitBn2Range_ = bn / coreNum_ + 1;
        tailSplitedBatchRange_ = blockSplitBn2Range_ - 1;
    }

    usedCoreNum_ = bn > coreNum_ ? coreNum_ : bn;

    for (uint32_t i = 0; i < formerCoreNum_; i++) {
        startIdxEachCore_[i] = blockSplitBn2Range_ * i;
    }

    uint32_t formerBase = formerCoreNum_ * blockSplitBn2Range_;
    for (uint32_t i = formerCoreNum_; i < usedCoreNum_; i++) {
        startIdxEachCore_[i] = formerBase + tailSplitedBatchRange_ * (i - formerCoreNum_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SplitBNS()
{
    formerCoreNum_ = 0;
    blockSplitBn2Range_ = 1;
    tailSplitedBatchRange_ = 1;

    uint32_t bn = batchSize_ * numKvHeads_;
    kvSplitPart_ = coreNum_ / bn;
    while (((maxActualseq_ / kvSplitPart_) < 512) && (kvSplitPart_ > 1)) { // 512, 经验值
        kvSplitPart_--;
    }

    usedCoreNum_ = bn * kvSplitPart_;
    uint32_t computeSeqSize = (seqSize_ + (kvSplitPart_ - 1)) / kvSplitPart_;
    if (inputKvType_ == ge::DT_INT4) {
        computeSeqSize = Align(computeSeqSize, 2U);
    }
    CalcInnerSize(computeSeqSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CalcInnerSize(uint32_t seqSize)
{
    /**
     * sInnerSize：s2的切分大小，直接决定了MM的singleN/K和vector的切块大小，但当前切分也并非适用所有case。
     * 1、非GQA场景：按照vector的最大基本块8*1024进行切分，sInnerSize=8192
     * 2、GQA场景：(1) 非伪量化：vector计算比较少，cube MTE2bound，
     *                          因此，cube发射大块，减少通信次数。sInnerSize=8192
     *            (2) 伪量化：vector比较重，尽量较少vector的循环次数,
     *                          因此，cube发小块，期望vector尽量被cube的mte2掩盖。sInnerSize=1024
     */
    sInnerSize_ = MAX_SPLIT_SIZE; // 8192
    if (antiQuantFlag_ && nNumOfQInOneGroup_ > 1) {
        sInnerSize_ = 1024U;
    } else if (!antiQuantFlag_) {
        /** 当前版本限制workspace大小不超过32MB，否则会影响网络中前后算子性能，
         *  GQA场景下 nNumOfQInOneGroup_和sInnerSize_切分大小直接影响workspace大小,
         *  具体计算参考CalcWorkSpace函数，这里根据nNumOfQInOneGroup_将sInnerSize_
         *  分为8192，4096，2048三档，nNumOfQInOneGroup_增大时减小sInnerSize_，
         *   保证最终workspace大小不超过32MB。
         */
        uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
        uint32_t idx = std::min(nNumOfQInOneGroup_ / 5U, 2U);
        sInnerSize_ = sInnerSize[idx];
    }

    // PA特性泛化场景，blockSize_可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
    if (pageAttentionFlag_ && blockSize_ != 0) {
        if (sInnerSize_ % blockSize_ != 0) {
            sInnerSize_ = (sInnerSize_ / blockSize_) * blockSize_;
        }
    }

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > seqSize) {
        sInnerSize_ = seqSize;
    }
    if (inputKvType_ == ge::DT_INT4) {
        sInnerSize_ = Align(sInnerSize_, 2U);
        sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1) * sInnerSize_;
        sInnerSizeAlign_ = Align(sInnerSize_, 64U); // 元素个数按照基本块大小对齐
    } else {
        sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    }

    CheckUbSpace();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckUbSpace()
{
    if (!CalcUbBmm() || !CalcUbSoftMax() || !CalcUbAttenMask()) {
        return false;
    }
    return true;
}

bool IFATiling::CalcUbBmm()
{
    mmResUbSize_ = msdIterNum_ * nNumOfQInOneGroup_ * sInnerSizeAlign_;
    bmm2ResUbSize_ = msdIterNum_ * nNumOfQInOneGroup_ * headDimAlign_;

    if (isSysPrefixTiling_) {
        mmResUbSize_ *= batchSizeQ_;
        bmm2ResUbSize_ *= batchSizeQ_;
    }
    return true;
}

bool IFATiling::CalcUbSoftMax()
{
    auto tmpShape = Shape({nNumOfQInOneGroup_, Align(sInnerSize_, BYTE_BLOCK / blockTypeSize_)});
    softmaxFlashTmpSize_ = GetSoftMaxFlashV2MinTmpSize(tmpShape, blockTypeSize_, blockTypeSize_, true, false);
    return true;
}

bool IFATiling::CalcUbAttenMask()
{
    if (!attenMaskFlag_) {
        selectWithByteMaskTmpMinSize_ = 0;
        return true;
    }
    // bool/int8/uint8类型的mask，每个占1字节
    attenMaskTypeSize_ = 1; // 1:sizeof(bool)
    auto selectWithByteMaskTmpShape =
        Shape({msdIterNum_ * nNumOfQInOneGroup_, Align(sInnerSize_, BYTE_BLOCK / attenMaskTypeSize_)});
    selectWithByteMaskTmpMinSize_ = GetSelectWithBytesMaskMinTmpSize(
        selectWithByteMaskTmpShape, Shape({1, 1}), FP32_BYTES, selectWithByteMaskTmpShape, FP32_BYTES, false);

    return true;
}

ge::graphStatus IFATiling::CalcWorkSpace()
{
    uint32_t mmResElemSize = 4;         // 4:fp32
    uint32_t vec1ResElemSize = 2;       // 2:fp16/bf16
    uint32_t bmm2ResElemSize = 4;       // 4:fp32
    uint32_t vec2ResElemSize = 4;       // 4:fp32
    uint32_t qPreProcResElemSize = 0;   // 普通场景不涉及Q预处理
    uint32_t mmPACallBackDataSize = 64; // 64: matmul回调信息需要7个uint32值，dcci cacheline需要64B对齐
    float kvDtypeRatio = 1.0;
    if (antiQuantFlag_) {
        mmResElemSize = 4;       // 4:int32
        vec1ResElemSize = 1;     // int
        bmm2ResElemSize = 4;     // 4:int32
        vec2ResElemSize = 4;     // 4:float
        qPreProcResElemSize = 1; // int
        kvDtypeRatio = inputKvType_ == ge::DT_INT4 ? 0.5 : 1.0; // 0.5:int4 1.0:elseType
    }

    workspaceSize_ = libapiSize_;
    if (perfMode_ != IfaPerfMode::BMM_ALL_BY_VEC) {
        workspaceSize_ += mmResUbSize_ * coreNum_ * mmResElemSize;
        workspaceSize_ += (size_t)((float)(mmResUbSize_ * coreNum_ * vec1ResElemSize) * kvDtypeRatio);
        workspaceSize_ += bmm2ResUbSize_ * coreNum_ * bmm2ResElemSize;
        workspaceSize_ += bmm2ResUbSize_ * coreNum_ * vec2ResElemSize;
        workspaceSize_ += (size_t)((float)(bmm2ResUbSize_ * coreNum_ * qPreProcResElemSize) * kvDtypeRatio);
    }
    if (splitKVFlag_) {
        auto accumOutSize = batchSizeQ_ * numHeads_ * kvSplitPart_ * headDimAlign_;
        auto logSumExpSize = 2 * batchSizeQ_ * numHeads_ * kvSplitPart_ * (BYTE_BLOCK / blockTypeSize_);
        workspaceSize_ += (accumOutSize + logSumExpSize) * blockTypeSize_;
        if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
            workspaceSize_ += static_cast<size_t>(coreNum_) * 32; // 每个核SyncAll软同步需要32Byte记录状态
        }
    }
    if (pageAttentionFlag_) {
        workspaceSize_ += coreNum_ * mmPACallBackDataSize * 2; // bmm1 bmm2 2份
    }

    if (isSysPrefixTiling_) {
        if (antiQuantFlag_) {
            size_t blockSize = nNumOfQInOneGroup_ * BYTE_BLOCK * batchSizeQ_;
            workspaceSize_ += coreNum_ * blockSize * 4; // 4, rowMax1 rowMax2 rowSum1 rowSum2
        }

        size_t blockSize = nNumOfQInOneGroup_ * BYTE_BLOCK * batchSizeQ_;
        workspaceSize_ += coreNum_ * blockSize * 3; // 3, sum log exp

        if (!antiQuantFlag_) {
            workspaceSize_ += batchSizeQ_ * nNumOfQInOneGroup_ * headDimAlign_ * 2 * coreNum_; // 2:fp16/bf16
        }
    }

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::FillTiling()
{
    FillTilingBaseParams();
    FillTilingSplitKV();
    FillTilingCoreParams();
    FillTilingSingleCoreParams();
    FillTilingSingleCoreTensorSize();
    FillTilingSoftmax();
    FillTilingOutputParams();
    return FillTilingBmm() ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

void IFATiling::FillTilingBaseParams()
{
    tilingData_->baseParams.set_batchSize(batchSize_);
    tilingData_->baseParams.set_seqSize(sMax_);
    tilingData_->baseParams.set_headSize(headDim_);
    tilingData_->baseParams.set_blockSize(blockSize_);
    tilingData_->baseParams.set_maxBlockNumPerBatch(maxBlockNumPerBatch_);
    tilingData_->baseParams.set_scaleValue(scaleValue_);
    tilingData_->baseParams.set_kvHeadNum(numKvHeads_);
    tilingData_->baseParams.set_qHeadNum(numHeads_);
    tilingData_->baseParams.set_nNumOfQInOneGroup(numHeads_ / numKvHeads_);
    tilingData_->baseParams.set_batchContinuousFlag(batchContinuousFlag_);
    tilingData_->baseParams.set_pseShiftFlag((pseShiftFlag_) ? 1 : 0);
    tilingData_->baseParams.set_pseShiftB(pseShiftBatch_);
    tilingData_->baseParams.set_pseShiftS(pseShiftS1_);
    tilingData_->baseParams.set_selectWithByteMaskTmpMinSize(selectWithByteMaskTmpMinSize_); // mask

    tilingData_->baseParams.set_actualLenDims(isSysPrefixTiling_ ? actualLenDimsPrefix_ : actualLenDims_);
    tilingData_->baseParams.set_msdIterNum(msdIterNum_);
    tilingData_->baseParams.set_kvPaddingFlag(kvPaddingSizeFlag_ ? 1 : 0);
    tilingData_->baseParams.set_antiquantPerTensorFlag(antiquantPerTensorFlag_);
    tilingData_->baseParams.set_antiquantPerHeadFlag(antiquantPerHeadFlag_);
    tilingData_->baseParams.set_antiquantParamsInPagedAttentionFlag(antiquantParamsInPagedAttentionFlag_);
    tilingData_->baseParams.set_attenMaskFlag(attenMaskFlag_ ? 1 : 0);
    tilingData_->baseParams.set_attenMaskSize(attenMaskSize_);
    tilingData_->baseParams.set_l2CacheOffFlag(l2CacheOffFlag_);
    tilingData_->baseParams.set_softmaxLseFlag(softmaxLseFlag_); // whether return lse
    tilingData_->baseParams.set_totalBlockNum(totalBlockNum_);
    tilingData_->baseParams.set_antiqSeqSize(antiqSeqSize_);
}

// for flash decode
void IFATiling::FillTilingSplitKV()
{
    tilingData_->splitKVParams.set_s2(kvSplitPart_);
    uint32_t sInnerLoopSize_ = (maxActualseq_ + (kvSplitPart_ - 1)) / kvSplitPart_;
    if (inputKvType_ == ge::DT_INT4) {
        sInnerLoopSize_ = Align(sInnerLoopSize_, 2U);
    }
    tilingData_->splitKVParams.set_sInnerLoopSize(sInnerLoopSize_);
    tilingData_->splitKVParams.set_accumOutSize(batchSizeQ_ * numHeads_ * kvSplitPart_ * headDimAlign_);
    tilingData_->splitKVParams.set_logSumExpSize(2 * batchSizeQ_ * numHeads_ * kvSplitPart_ *
                                                 (BYTE_BLOCK / blockTypeSize_)); // 2: sum + max
    if (!splitKVFlag_) {
        tilingData_->splitKVParams.set_s2(0);
    }
}

void IFATiling::FillTilingCoreParams()
{
    uint32_t *coreStartIdx = tilingData_->increFlashAttentionCoreParams.get_coreSidxEnd();
    memcpy_s(coreStartIdx, MAX_CORE_NUM * sizeof(uint32_t), startIdxEachCore_, MAX_CORE_NUM * sizeof(uint32_t));
}

void IFATiling::FillTilingSingleCoreParams()
{
    tilingData_->increFlashAttentionSingleCoreParams.set_sInnerLoopTimes(sInnerLoopTimes_);
    tilingData_->increFlashAttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerSize_);
    tilingData_->increFlashAttentionSingleCoreParams.set_singleProcessSInnerSizeTail(sInnerSizeTail_);
    tilingData_->increFlashAttentionSingleCoreParams.set_usedCoreNum(usedCoreNum_);
    tilingData_->increFlashAttentionSingleCoreParams.set_formerCoreNum(formerCoreNum_);
    tilingData_->increFlashAttentionSingleCoreParams.set_blockSplitBn2Range(blockSplitBn2Range_);
    tilingData_->increFlashAttentionSingleCoreParams.set_tailSplitedBatchRange(tailSplitedBatchRange_);
}

void IFATiling::FillTilingSingleCoreTensorSize()
{
    tilingData_->increFlashAttentionSingleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_->increFlashAttentionSingleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void IFATiling::FillTilingSoftmax()
{
    auto softmaxShape = Shape({1, Align(sInnerSize_, BYTE_BLOCK / blockTypeSize_)});
    SoftMaxFlashV2TilingFunc(softmaxShape, blockTypeSize_, blockTypeSize_, softmaxFlashTmpSize_,
                             tilingData_->softmaxFlashTilingData, true, false);
}

// for zero output
void IFATiling::FillTilingOutputParams()
{
    tilingData_->outputParams.set_isOutQuantTypeBf16(isOutQuantTypeBf16_);
    tilingData_->outputParams.set_isPerChnOut(isOutQuantPerChnOut_);
}

void IFATiling::AdjustPABmm1Tiling(uint32_t &bmm1BaseN)
{
    if (bmm1BaseN < blockSize_) {
        while (blockSize_ % bmm1BaseN != 0) {
            bmm1BaseN /=
                2; // 2:不断减半，确保1个base块不会跨block拷贝。已校验过blockSize 16/32对齐，因此bmm1BaseN最小值为16/32
        }
    } else if (bmm1BaseN > blockSize_) {
        // nd2nz拷贝时ndnum>1场景性能较差，通过设置baseN <= blocksize避免
        uint32_t tmpBaseN = increGcd(bmm1BaseN, blockSize_);
        bmm1BaseN = tmpBaseN;
    }
    OPS_LOG_D(context_->opName, "PA is enabled, blockSize is %d, bmm1 baseN is adjusted to %d", blockSize_, bmm1BaseN);
}

void IFATiling::AdjustPABmm2Tiling() const
{
    uint32_t targetBaseK = 128;
    if (targetBaseK < blockSize_) {
        while ((blockSize_ % targetBaseK != 0) ||
               (targetBaseK * tilingData_->bmm2TilingData.get_baseN() * sizeof(float) > L0B_SIZE)) {
            targetBaseK /=
                2; // 2:不断减半，确保1个base块不会跨block拷贝，已校验过blockSize_16/32对齐，因此targetBaseK最小值为16/32
        }
    } else {
        uint32_t tmpBaseK = increGcd(targetBaseK, blockSize_);
        while (tmpBaseK * tilingData_->bmm2TilingData.get_baseN() * sizeof(float) > L0B_SIZE) {
            tmpBaseK /= 2; // 2: 不断减半，确保base块大小在LOB有效范围内
        }
        targetBaseK = tmpBaseK;
    }
    // mm api不支持通过 SetFixSplit 设置baseK，需要直接配置tiling结构体
    tilingData_->bmm2TilingData.set_baseK(targetBaseK);
    OPS_LOG_D(context_->opName, "PA is enabled, blockSize is %d, bmm2 baseK is adjusted to %d", blockSize_,
              targetBaseK);
}

bool IFATiling::GetBmm1Tiling(const matmul_tiling::DataType &qType, const matmul_tiling::DataType &kvType,
                              const uint32_t M)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
    uint32_t baseN;
    uint32_t bmm1OrgKa;
    bmm1.SetShape(M, sInnerSize_, headDim_);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, true);
    if (antiQuantFlag_) {
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                      matmul_tiling::DataType::DT_INT32);
        bmm1OrgKa = headDimAlign_;
        baseN = MAX_MATMUL_BASE; // antiquant to split K
    } else {
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        bmm1OrgKa = headDim_;
        baseN = MATMUL_BASE_N;
    }
    // 存在输入query是BNSD格式，但使能PA，需要按BSH SetOrgShape
    if (inputLayout_ == IfaLayout::BSH_BSND) {
        bmm1.SetOrgShape(M, seqSize_, bmm1OrgKa, headDim_ * numKvHeads_);
    } else {
        bmm1.SetOrgShape(M, seqSize_, bmm1OrgKa, headDim_);
    }
    bmm1.SetBias(false);

    uint32_t bmm1BaseN = std::min(Align(sInnerSize_, 16U), baseN);
    if (pageAttentionFlag_) {
        AdjustPABmm1Tiling(bmm1BaseN);
    }

    if (!isSysPrefixTiling_) {
        // 向下对齐保证M*N不超过L0C，且由于bmm1BaseN有最大限制，L0C_SIZE / sizeof(float) / bmm1BaseN不会小于16
        uint32_t bmm1MaxBaseM = Align(static_cast<uint32_t>(L0C_SIZE / sizeof(float) / bmm1BaseN) - 16U, 16U);
        OPS_ERR_IF((bmm1.SetFixSplit(std::min(Align(M, 16U), bmm1MaxBaseM), bmm1BaseN) == -1),
                   OPS_LOG_E(context_->opName, "bmm1 SetFixSplit fail"), return false);
    } else {
        // prefix 模式下A矩阵较大，可能超过L0A，使用默认值-1，由matmul计算baseM
        OPS_ERR_IF((bmm1.SetFixSplit(-1, bmm1BaseN) == -1), OPS_LOG_E(context_->opName, "bmm1 SetFixSplit fail"),
                   return false);
    }

    OPS_ERR_IF((bmm1.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN) == -1),
               OPS_LOG_E(context_->opName, "bmm1 SetTraverse fail"), return false);

    if (bmm1.GetTiling(tilingData_->bmm1TilingData) == -1) {
        OPS_LOG_E(context_->opName, "bmm1 get tiling fail");
        return false;
    }
    return true;
}

bool IFATiling::GetBmm2Tiling(const matmul_tiling::DataType &qType, const matmul_tiling::DataType &kvType,
                              const uint32_t M)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    if (antiQuantFlag_) {
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                      matmul_tiling::DataType::DT_INT32);
    } else {
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
                      matmul_tiling::DataType::DT_FLOAT);
    }
    // (m, n, k) (so, d, si)
    bmm2.SetShape(M, headDim_, sInnerSize_);
    // 存在输入query是BNSD格式，但使能PA，需要按BSH SetOrgShape
    if (inputLayout_ == IfaLayout::BSH_BSND) {
        bmm2.SetOrgShape(M, headDim_ * numKvHeads_, sInnerSizeAlign_, seqSize_);
    } else {
        bmm2.SetOrgShape(M, headDim_, sInnerSizeAlign_, seqSize_);
    }
    bmm2.SetBias(false);
    OPS_ERR_IF((bmm2.SetFixSplit(std::min(Align(M, 16U), MAX_MATMUL_BASE_M)) == -1),
               OPS_LOG_E(context_->opName, "bmm2 SetFixSplit fail"), return false);

    if (bmm2.GetTiling(tilingData_->bmm2TilingData) == -1) {
        OPS_LOG_E(context_->opName, "bmm2 get tiling fail");
        return false;
    }
    if (pageAttentionFlag_) {
        AdjustPABmm2Tiling();
    }
    return true;
}

bool IFATiling::FillTilingBmm()
{
    matmul_tiling::DataType qType;
    matmul_tiling::DataType kvType;

    if (!GetMatmulType(inputQType_, &qType) || !GetMatmulType(inputKvType_, &kvType)) {
        OPS_LOG_E(context_->opName, "get matmul type error");
        return false;
    }
    uint32_t M = msdIterNum_ * nNumOfQInOneGroup_;
    if (isSysPrefixTiling_) {
        M *= batchSizeQ_;
    }
    return GetBmm1Tiling(qType, kvType, M) && GetBmm2Tiling(qType, kvType, M);
}

bool IFATiling::GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType)
{
    static struct {
        ge::DataType a;
        matmul_tiling::DataType b;
    } typeTrans[] = {{ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
                     {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
                     {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
                     {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT}};

    for (uint32_t i = 0; i < sizeof(typeTrans) / sizeof(typeTrans[0]); i++) {
        if (typeTrans[i].a == getype) {
            *mmType = typeTrans[i].b;
            return true;
        }
    }
    return false;
}

ge::graphStatus IFATiling::GenTilingKey()
{
    uint8_t layoutVal{0}, inputQVal{0}, inputKvVal{0}, outputVal{0}, originVal{0};
    uint8_t splitKvVal = kvSplit_ > 0 ? 1 : 0;
    uint8_t paVal = pageAttentionFlag_ == true ? 1 * 2 : 0;
    uint8_t antiquantModeVal = antiquantMode_ == PER_TOKEN_MODE ? 1 * 4 : 0;
    uint64_t modeVal = sysPrefixFlag_ ? 2 : 1;

    // page attention 新模板上线后删除这里的特殊处理
    if (pageAttentionFlag_ && sMax_ == 0) {
        paVal = 0;
    }

    switch (inputLayout_) {
        case IfaLayout::BSH_BSND:
            layoutVal = 1;
            break;
        case IfaLayout::BNSD:
            layoutVal = 0;
            break;
        default:
            OPS_LOG_E(context_->opName, "not support inputLayout%u", inputLayout_);
            return ge::GRAPH_FAILED;
    }
    switch (inputQType_) {
        case ge::DT_FLOAT16:
            inputQVal = 0;
            break;
        case ge::DT_BF16:
            inputQVal = 2U;
            break;
        default:
            OPS_LOG_E(context_->opName, "not support inputQType%d", inputQType_);
            return ge::GRAPH_FAILED;
    }
    switch (inputKvType_) {
        case ge::DT_FLOAT16:
            inputKvVal = 0;
            break;
        case ge::DT_BF16:
            inputKvVal = 2U;
            break;
        case ge::DT_INT8:
            inputKvVal = 3U;
            break;
        case ge::DT_INT4:
            inputKvVal = 4U;
            break;
        default:
            OPS_LOG_E(context_->opName, "not support inputKvType%d", inputKvType_);
            return ge::GRAPH_FAILED;
    }
    switch (outputType_) {
        case ge::DT_FLOAT16:
            outputVal = 0;
            break;
        case ge::DT_BF16:
            outputVal = 2U;
            break;
        case ge::DT_INT8:
            outputVal = 3U;
            break;
        default:
            OPS_LOG_E(context_->opName, "not support outputType%d", outputType_);
            return ge::GRAPH_FAILED;
    }

    originVal = inputQVal;

    uint64_t baseOffset =
        modeVal * IFA_TILINGKEYOFFSET + (static_cast<uint64_t>(perfMode_)) * IFA_PERF_MODE_TILINGKEYOFFSET;
    context_->tilingKey = baseOffset + IFA_GET_TILINGKEY(layoutVal, inputQVal, inputKvVal, outputVal, originVal,
                                                         (paVal + splitKvVal + antiquantModeVal));
    OPS_LOG_D(context_->opName, "IFA tilingKey:%llu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    auto aicNum = aicNum_;
    auto aivNum = aivNum_;
    UpdatePerfMode();
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        aivNum = aicNum;
    } else {
        if (!splitKVFlag_) {
            if (perfMode_ == IfaPerfMode::C1_V1) { // 2:bn数不超过vector core一半时，CV开启CV 1:1
                aivNum = usedCoreNum_;             // CV 1:1时,GetTaskRation()的结果为1,所以aivNum与aicNum相等
                aicNum = aivNum;
            } else {
                aivNum = Align(usedCoreNum_, 2U); // aivNum必须为偶数达成CV 1:2
                aicNum = (aivNum + 1) / 2;        // cube核的数量为vector核的数量按2向上对齐
            }
        }
    }
    context_->blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum); // 暂时与当前代码一致
    OPS_LOG_D(context_->opName, "IFA block dim:%u aivNum:%u aicNum:%u", context_->blockDim, aivNum, aicNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SharedPrefixTiling()
{
    // 重新配置长度
    isSysPrefixTiling_ = true;
    splitKVFlag_ = false;
    batchSizeQ_ = batchSize_;
    batchSize_ = 1;
    maxActualseq_ = maxActualPrefixLen_;
    sMax_ = sMaxPrefix_;
    seqSize_ = sMax_;
    batchContinuousFlag_ = true;

    (void)ZeroTensorProcess();
    (void)Split();
    (void)SplitForLseCombine();
    (void)CalcSysPrefixWorkSpace();
    (void)FillSysPrefixTiling();
    (void)CalcSysPrefixBlockDim();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::FillSysPrefixTiling()
{
    tilingDataPrefix_->set_prefixAttenOutOffset(prefixAttenOutOffset_);
    tilingDataPrefix_->set_userPromptAttenOutOffset(userPromptAttenOutOffset_);
    tilingDataPrefix_->set_tmpLseOffset(tmpLseOffset_);
    tilingDataPrefix_->set_prefixLen(maxActualPrefixLen_);
    tilingDataPrefix_->set_formerCoreNum(formerCoreNumSp_);
    tilingDataPrefix_->set_blockSplitBn2Range(blockSplitBn2RangeSp_);
    tilingDataPrefix_->set_tailSplitedBatchRange(tailSplitedBatchRangeSp_);
    tilingDataPrefix_->set_usedCoreNum(combinUsedCore_);
    tilingDataPrefix_->set_batchSizeQ(batchSizeQ_);
    tilingData_ = &tilingDataPrefix_->base;
    return FillTiling();
}

ge::graphStatus IFATiling::CalcSysPrefixWorkSpace()
{
    size_t size0 = workspaceSize_;
    size_t outSize = batchSizeQ_ * numHeads_ * headDimAlign_ * blockTypeSize_;
    size_t lseSize = 4 * batchSizeQ_ * numHeads_ * BYTE_BLOCK;

    CalcWorkSpace();

    workspaceSize_ = std::max(workspaceSize_, size0);
    workspaceSize_ = Align(workspaceSize_, 512UL);
    prefixAttenOutOffset_ = workspaceSize_ - libapiSize_;
    workspaceSize_ += outSize;
    userPromptAttenOutOffset_ = workspaceSize_ - libapiSize_;
    workspaceSize_ += outSize;

    tmpLseOffset_ = workspaceSize_ - libapiSize_;
    workspaceSize_ += lseSize;

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CalcSysPrefixBlockDim()
{
    uint32_t blockDim0 = context_->blockDim;
    CalcBlockDim();

    context_->blockDim = std::max(blockDim0, context_->blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SplitForLseCombine()
{
    uint32_t coreNum = usedCoreNum_;

    uint32_t bn = batchSizeQ_ * numKvHeads_;
    formerCoreNumSp_ = bn % coreNum;
    if (formerCoreNumSp_ == 0) {
        blockSplitBn2RangeSp_ = bn / coreNum;
        tailSplitedBatchRangeSp_ = blockSplitBn2RangeSp_;
    } else {
        blockSplitBn2RangeSp_ = bn / coreNum + 1;
        tailSplitedBatchRangeSp_ = blockSplitBn2RangeSp_ - 1;
    }
    combinUsedCore_ = bn > coreNum ? coreNum : bn;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ConvertContext(gert::TilingContext &context, IncreFlashAttentionContext &ifaContext)
{
    if (context.GetNodeName() == nullptr) {
        OPS_LOG_E("IncreFlashAttention", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    ifaContext.opName = context.GetNodeName();
    ifaContext.platformInfo = context.GetPlatformInfo();
    ifaContext.query.desc = context.GetInputDesc(QUERY_INPUT_INDEX);
    ifaContext.query.shape = context.GetInputShape(QUERY_INPUT_INDEX);
    ifaContext.key.desc = context.GetInputDesc(KEY_INPUT_INDEX);
    ifaContext.key.shape = context.GetInputShape(KEY_INPUT_INDEX);
    OPS_ERR_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
               OPS_LOG_E(context.GetNodeName(), "shape of query or shape of key is null."), return ge::GRAPH_FAILED);
    auto batchOfQuery = ifaContext.query.shape->GetStorageShape().GetDim(0);
    auto batchOfKey = ifaContext.key.shape->GetStorageShape().GetDim(0);
    if (batchOfQuery != batchOfKey) {
        ifaContext.kCache.resize(batchOfQuery);
        ifaContext.vCache.resize(batchOfQuery);
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            ifaContext.kCache[size] =
                const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, size));
            ifaContext.vCache[size] =
                const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, size));
        }
    } else {
        ifaContext.kCache.resize(1);
        ifaContext.vCache.resize(1);
        ifaContext.kCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(KEY_INPUT_INDEX, 0));
        ifaContext.vCache[0] = const_cast<gert::StorageShape *>(context.GetDynamicInputShape(VALUE_INPUT_INDEX, 0));
    }

    ifaContext.value.desc = context.GetInputDesc(VALUE_INPUT_INDEX);
    ifaContext.value.shape = context.GetInputShape(VALUE_INPUT_INDEX);
    ifaContext.pseShift.desc = context.GetOptionalInputDesc(PSE_SHIFT_INPUT_INDEX);
    ifaContext.pseShift.tensor = context.GetOptionalInputTensor(PSE_SHIFT_INPUT_INDEX);
    ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX);
    ifaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);
    ifaContext.attenOut.desc = context.GetOutputDesc(OUTPUT_INDEX);
    ifaContext.attenOut.shape = context.GetOutputShape(OUTPUT_INDEX);

    ifaContext.actualSeqLengths.tensor = context.GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
    ifaContext.deqScale1.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_1_INPUT_INDEX);
    ifaContext.quantScale1.tensor = context.GetOptionalInputTensor(QUANT_SCALE_1_INPUT_INDEX);
    ifaContext.deqScale2.tensor = context.GetOptionalInputTensor(DEQUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantScale2.tensor = context.GetOptionalInputTensor(QUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantOffset2.tensor = context.GetOptionalInputTensor(QUANT_OFFSET_2_INPUT_INDEX);
    ifaContext.deqScale1.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_1_INPUT_INDEX);
    ifaContext.quantScale1.desc = context.GetOptionalInputDesc(QUANT_SCALE_1_INPUT_INDEX);
    ifaContext.deqScale2.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantScale2.desc = context.GetOptionalInputDesc(QUANT_SCALE_2_INPUT_INDEX);
    ifaContext.quantOffset2.desc = context.GetOptionalInputDesc(QUANT_OFFSET_2_INPUT_INDEX);
    ifaContext.antiquantScale.tensor = context.GetOptionalInputTensor(ANTIQUANT_SCALE_INPUT_INDEX);
    ifaContext.antiquantOffset.tensor = context.GetOptionalInputTensor(ANTIQUANT_OFFSET_INPUT_INDEX);
    ifaContext.antiquantScale.desc = context.GetOptionalInputDesc(ANTIQUANT_SCALE_INPUT_INDEX);
    ifaContext.antiquantOffset.desc = context.GetOptionalInputDesc(ANTIQUANT_OFFSET_INPUT_INDEX);
    ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    ifaContext.kvPaddingSize.tensor = context.GetOptionalInputTensor(KV_PADDING_SIZE_INPUT_INDEX);
  ifaContext.kvPaddingSize.desc = context.GetOptionalInputDesc(KV_PADDING_SIZE_INPUT_INDEX);

    auto attrs = context.GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context.GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    ifaContext.numHeads = attrs->GetAttrPointer<uint32_t>(NUM_HEADS_ATTR_INDEX);
    ifaContext.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    ifaContext.layOut = attrs->GetStr(LAYOUT_ATTR_INDEX);
    ifaContext.kvHeadNums = attrs->GetAttrPointer<uint32_t>(KV_NUM_HEADS_ATTR_INDEX);
    ifaContext.blockSize = attrs->GetAttrPointer<uint32_t>(BLOCK_SIZE_ATTR_INDEX);
    ifaContext.innerPrecise = attrs->GetAttrPointer<uint32_t>(INNER_PRECISE_ATTR_INDEX);

    OPS_ERR_IF(context.GetWorkspaceSizes(1) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    ifaContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::RunBigKernelTiling(IncreFlashAttentionContext &context,
                                              IncreFlashAttentionTilingDataV2 &tilingData, bool isWorkspace)
{
    this->context_ = &context;
    this->tilingData_ = &tilingData.tilingBase;
    this->tilingDataPrefix_ = &tilingData.tilingPrefix;
    this->isWorkspace_ = isWorkspace;

    if ((GetNpuInfo() != ge::GRAPH_SUCCESS) || (PreProcess() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    // user prompt tiling
    if ((ZeroTensorProcess() != ge::GRAPH_SUCCESS) || (Split() != ge::GRAPH_SUCCESS) ||
        (FillTiling() != ge::GRAPH_SUCCESS) || (CalcWorkSpace() != ge::GRAPH_SUCCESS) ||
        (CalcBlockDim() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    if (sysPrefixFlag_ && SharedPrefixTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return GenTilingKey();
}

ge::graphStatus IFATiling::IncreFlashAttentionSetTilingData(gert::TilingContext &context,
                                                            IncreFlashAttentionTilingDataV2 &tilingData)
{
    OPS_ERR_IF(context.GetRawTilingData() == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "RawTilingData got from ge context is nullptr."),
               return GRAPH_FAILED);
    tilingData.SaveToBuffer(context.GetRawTilingData()->GetData(), context.GetRawTilingData()->GetCapacity());
    context.GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OPS_LOG_E("IncreFlashAttention", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

ge::graphStatus TilingIncreFlashAttentionAdapter(gert::TilingContext *context, IncreFlashAttentionContext &ifaContext,
                                                 IncreFlashAttentionTilingDataV2 &ifaTilingData)
{
    IFATiling ifaTilingNew;
    if (ifaTilingNew.RunBigKernelTiling(ifaContext, ifaTilingData) == ge::SUCCESS) {
        context->SetTilingKey(ifaContext.tilingKey);
        context->SetBlockDim(ifaContext.blockDim);
        ifaTilingNew.IncreFlashAttentionSetTilingData(*context, ifaTilingData);
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_FAILED;
}

ge::graphStatus TilingIncreFlashAttention(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("IncreFlashAttention", "Context is nullptr."),
               return ge::GRAPH_FAILED);

    IncreFlashAttentionTilingDataV2 tilingData;
    IncreFlashAttentionContext ifaContext{.opName = nullptr,
                                          .platformInfo = nullptr,
                                          .query = {nullptr, nullptr},
                                          .key = {nullptr, nullptr},
                                          .value = {nullptr, nullptr},
                                          .pseShift = {nullptr, nullptr},
                                          .attenMask = {nullptr, nullptr},
                                          .actualSeqLengths = {nullptr, nullptr},
                                          .deqScale1 = {nullptr, nullptr},
                                          .quantScale1 = {nullptr, nullptr},
                                          .deqScale2 = {nullptr, nullptr},
                                          .quantScale2 = {nullptr, nullptr},
                                          .quantOffset2 = {nullptr, nullptr},
                                          .antiquantScale = {nullptr, nullptr},
                                          .antiquantOffset = {nullptr, nullptr},
                                          .blockTable = {nullptr, nullptr},
                                          .kvPaddingSize = {nullptr, nullptr},
                                          .keyAntiquantScale = {nullptr, nullptr},
                                          .keyAntiquantOffset = {nullptr, nullptr},
                                          .valueAntiquantScale = {nullptr, nullptr},
                                          .valueAntiquantOffset = {nullptr, nullptr},
                                          .keySharedPrefix = {nullptr, nullptr},
                                          .valueSharedPrefix = {nullptr, nullptr},
                                          .actualSharedPrefixLen = {nullptr, nullptr},
                                          .attenOut = {nullptr, nullptr},
                                          .numHeads = nullptr,
                                          .scaleValue = nullptr,
                                          .kvHeadNums = nullptr,
                                          .layOut = nullptr,
                                          .blockSize = nullptr,
                                          .innerPrecise = nullptr,
                                          .antiquantMode = nullptr,
                                          .softmaxLseFlag = nullptr,
                                          .keyAntiquantMode = nullptr,
                                          .valueAntiquantMode = nullptr,
                                          .workSpaces = nullptr,
                                          .kCache = {nullptr},
                                          .vCache = {nullptr},
                                          .tilingKey = 0,
                                          .blockDim = 0};

    if (IFATiling::ConvertContext(*context, ifaContext) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Error occurred while converting tilingContext to ifa context");
        return ge::GRAPH_FAILED;
    }
    return TilingIncreFlashAttentionAdapter(context, ifaContext, tilingData);
}
} // namespace optiling
