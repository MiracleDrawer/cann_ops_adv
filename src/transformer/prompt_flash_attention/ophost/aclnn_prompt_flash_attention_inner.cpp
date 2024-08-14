/**
 * Copyright (c) 2023 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

#include "aclnn_prompt_flash_attention_inner.h"
#include "prompt_flash_attention.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const uint64_t PAD_BASIC_BLOCK = 32;
static const uint64_t HALF_PAD_BASIC_BLOCK = 16;
static const uint64_t MAX_STRIDE_S1 = 65535;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t INDEX_2 = 2;
static const uint64_t INDEX_3 = 3;
static const uint64_t CHAR_0 = 0;
static const uint64_t CHAR_1 = 1;
static const uint64_t CHAR_2 = 2;
static const uint64_t CHAR_3 = 3;
static const uint64_t CHAR_4 = 4;
static const uint64_t CHAR_9 = 9;


struct AxesInfo {
    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t d;
};

enum class InputLayout { SH, BSH, NSD, BNSD, BSND, BNSD_BSND, NONE, };

static std::unordered_map<DataType, string> StrDataTypePfa = {
    {DataType::DT_FLOAT, "DT_FLOAT"},
    {DataType::DT_FLOAT16, "DT_FLOAT16"},
    {DataType::DT_INT8, "DT_INT8"},
    {DataType::DT_INT16, "DT_INT16"},
    {DataType::DT_UINT16, "DT_UINT16"},
    {DataType::DT_UINT8, "DT_UINT8"},
    {DataType::DT_INT32, "DT_INT32"},
    {DataType::DT_INT64, "DT_INT64"},
    {DataType::DT_UINT32, "DT_UINT32"},
    {DataType::DT_UINT64, "DT_UINT64"},
    {DataType::DT_BOOL, "DT_BOOL"},
    {DataType::DT_DOUBLE, "DT_DOUBLE"},
    {DataType::DT_STRING, "DT_STRING"},
    {DataType::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {DataType::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8V"},
    {DataType::DT_COMPLEX64, "DT_COMPLEX64"},
    {DataType::DT_COMPLEX128, "DT_COMPLEX128"},
    {DataType::DT_QINT8, "DT_QINT8"},
    {DataType::DT_QINT16, "DT_QINT16"},
    {DataType::DT_QINT32, "DT_QINT32"},
    {DataType::DT_QUINT8, "DT_QUINT8"},
    {DataType::DT_QUINT16, "DT_QUINT16"},
    {DataType::DT_RESOURCE, "DT_RESOURCE"},
    {DataType::DT_STRING_REF, "DT_STRING_REF"},
    {DataType::DT_DUAL, "DT_DUAL"},
    {DataType::DT_VARIANT, "DT_VARIANT"},
    {DataType::DT_BF16, "DT_BF16"},
    {DataType::DT_UNDEFINED, "DT_UNDEFINED"},
};

struct FaShapeInfo {
    AxesInfo axes;

    InputLayout inputLayout;
    string l0InputLayoutStr;

    uint64_t dimNum = 0;
    uint64_t padNum = 0;
    uint64_t basicBlock = HALF_PAD_BASIC_BLOCK;

    FVector<int64_t, DIM_NUM_4> perm_in;
    FVector<int64_t, DIM_NUM_4> perm_out;
    FVector<int64_t, DIM_NUM_4> reshapedQueryShape;
    FVector<int64_t, DIM_NUM_4> reshapedKeyValueShape;

    bool needPad = false;
    bool needTranspose = false;
    bool needReshape = false;
};

static aclnnStatus CheckDimsAndLayout(const aclTensor *query, const aclTensor *key, const aclTensor *value, const char *inputLayout) {
    auto qDimNum = query->GetViewShape().GetDimNum();
    auto kDimNum = key->GetViewShape().GetDimNum();
    auto vDimNum = value->GetViewShape().GetDimNum();
    if (qDimNum != kDimNum || qDimNum != vDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "the layout of q and k v must be same, but got q dim:%lu k dim:%lu v dim:%lu", qDimNum, kDimNum, vDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_4) && strnlen(inputLayout, CHAR_4) >= CHAR_4 && (inputLayout[CHAR_0] == 'B' && inputLayout[CHAR_1] == 'N' &&
        inputLayout[CHAR_2] == 'S' && inputLayout[CHAR_3] == 'D')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is BNSD, input shape dim should be 4, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_4) && strnlen(inputLayout, CHAR_4) >= CHAR_4 && (inputLayout[CHAR_0] == 'B' && inputLayout[CHAR_1] == 'S' &&
        inputLayout[CHAR_2] == 'N' && inputLayout[CHAR_3] == 'D')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is BSND, input shape dim should be 4, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_3) && strnlen(inputLayout, CHAR_4) >= CHAR_3 && (inputLayout[CHAR_0] == 'B' && inputLayout[CHAR_1] == 'S' &&
               inputLayout[CHAR_2] == 'H')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is BSH, input shape dim should be 3, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_3) && strnlen(inputLayout, CHAR_4) >= CHAR_3 && (inputLayout[CHAR_0] == 'N' && inputLayout[CHAR_1] == 'S' &&
               inputLayout[CHAR_2] == 'D')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is NSD, input shape dim should be 3, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    } else if ((qDimNum != DIM_NUM_2) && strnlen(inputLayout, CHAR_4) >= CHAR_2 && (inputLayout[CHAR_0] == 'S' && inputLayout[CHAR_1] == 'H')) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout is SH, input shape dim should be 2, but got %lu", qDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
static aclnnStatus analysisAxis(const aclTensor *query, const aclTensor *key, const char *inputLayout, int64_t headNum,
                                FaShapeInfo &shapeInfo)
{
    Shape qShape = query->GetViewShape();
    Shape kShape = key->GetViewShape();
    shapeInfo.dimNum = qShape.GetDimNum();

    // 记录轴的长度 b, n2, g, s1, s2, d
    // H1等于N1*D, H2等于N2*D
    // N1等于g*N2
    shapeInfo.axes.n1 = headNum;

    if (strlen(inputLayout) != 0) {
        shapeInfo.inputLayout = InputLayout::NONE;
        shapeInfo.l0InputLayoutStr = "NONE";
    }

    // query: (B*S1, N1*D)
    // key/value: (B*S2, N2*D)
    if (shapeInfo.dimNum == DIM_NUM_2 && strnlen(inputLayout, CHAR_4) >= CHAR_2 && inputLayout[0] == 'S' && inputLayout[1] == 'H') {
        uint64_t dSize = qShape[1] / headNum;
        if (dSize == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input query shape is (S[%ld], H[%ld]), input num_head N is %ld, "
                    "corresponding headsize D = H/N = %ld, is invalid.",
                    qShape[0], qShape[1], headNum, dSize); // 0:S, 1:H
            return ACLNN_ERR_PARAM_INVALID;
        }
        shapeInfo.axes.b = 1;
        shapeInfo.axes.n2 = kShape[1] / dSize;
        shapeInfo.axes.s1 = qShape[0];
        shapeInfo.axes.s2 = kShape[0];
        shapeInfo.axes.d = dSize;
        shapeInfo.inputLayout = InputLayout::SH;
        shapeInfo.l0InputLayoutStr = "SH";
    }

    // query: (B,S1,N1*D)
    // key/value: (B,S2,N2*D)
    if (shapeInfo.dimNum == DIM_NUM_3 && strnlen(inputLayout, CHAR_4) >= CHAR_3 && inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'H') {
        uint64_t dSize = qShape[2] / headNum;
        if (dSize == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input query shape is (B[%ld], S[%ld], H[%ld]), input num_head N is %ld, "
                    "corresponding headsize D = H/N = %ld, is invalid.",
                    qShape[0], qShape[1], qShape[2], headNum, dSize); // 0:B, 1:S, 2:H
            return ACLNN_ERR_PARAM_INVALID;
        }
        shapeInfo.axes.b = qShape[0];
        shapeInfo.axes.n2 = kShape[2] / dSize;
        shapeInfo.axes.s1 = qShape[1];
        shapeInfo.axes.s2 = kShape[1];
        shapeInfo.axes.d = dSize;
        shapeInfo.inputLayout = InputLayout::BSH;
        shapeInfo.l0InputLayoutStr = "BSH";
    }

    // query: (B,S1,N1,D)
    // key/value: (B,S2,N2,D)
    if (shapeInfo.dimNum == DIM_NUM_4 && strnlen(inputLayout, CHAR_4) >= CHAR_4 && inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' &&
        inputLayout[3] == 'D') {
        shapeInfo.axes.b = qShape[0];
        shapeInfo.axes.n2 = kShape[INDEX_2];
        shapeInfo.axes.s1 = qShape[1];
        shapeInfo.axes.s2 = kShape[1];
        shapeInfo.axes.d = qShape[INDEX_3];
        shapeInfo.inputLayout = InputLayout::BSND;
        shapeInfo.l0InputLayoutStr = "BSND";
    }

    // query: (B*N1,S1,D)
    // key/value: (B*N2,S2,D)
    if (shapeInfo.dimNum == DIM_NUM_3 && strnlen(inputLayout, CHAR_4) >= CHAR_3 && inputLayout[0] == 'N' && inputLayout[1] == 'S' && inputLayout[2] == 'D') {
        shapeInfo.axes.b = 1;
        shapeInfo.axes.n2 = kShape[0];
        shapeInfo.axes.s1 = qShape[1];
        shapeInfo.axes.s2 = kShape[1];
        shapeInfo.axes.d = qShape[2];
        shapeInfo.inputLayout = InputLayout::NSD;
        shapeInfo.l0InputLayoutStr = "NSD";
    }

    // query: (B,N1,S1,D)
    // key/value: (B,N2,S2,D)
    if (shapeInfo.dimNum == DIM_NUM_4 && strnlen(inputLayout, CHAR_4) >= CHAR_4 && inputLayout[0] == 'B' && inputLayout[1] == 'N' && inputLayout[2] == 'S' &&
        inputLayout[3] == 'D') {
        shapeInfo.axes.b = qShape[0];
        shapeInfo.axes.n2 = kShape[1];
        shapeInfo.axes.s1 = qShape[2];
        shapeInfo.axes.s2 = kShape[2];
        shapeInfo.axes.d = qShape[3];
        if (strnlen(inputLayout, CHAR_9) == CHAR_9 && inputLayout[4] == '_' && inputLayout[5] == 'B' && inputLayout[6] == 'S' && inputLayout[7] == 'N' &&
            inputLayout[8] == 'D') {
            shapeInfo.inputLayout = InputLayout::BNSD_BSND;
            shapeInfo.l0InputLayoutStr = "BNSD_BSND";
        } else {
            shapeInfo.inputLayout = InputLayout::BNSD;
            shapeInfo.l0InputLayoutStr = "BNSD";
        }
    }

    OP_LOGD("Analysis axis success. "
            "The axis result: [B]: %ld, [n1]: %ld, [n2]: %ld, [s1]: %ld, [s2]: %ld, [d]: %ld",
            shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.n2, shapeInfo.axes.s1, shapeInfo.axes.s2,
            shapeInfo.axes.d);
    return ACLNN_SUCCESS;
}

static void SetShapeInfoForSH(FaShapeInfo &shapeInfo) {
    if (shapeInfo.needPad) {
        shapeInfo.needReshape = true;
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.b, shapeInfo.axes.s1, shapeInfo.axes.n1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyValueShape.assign(
            {shapeInfo.axes.b, shapeInfo.axes.s2, shapeInfo.axes.n2, shapeInfo.axes.d});
    }
}

static void SetShapeInfoForNSD(FaShapeInfo &shapeInfo) {
    if (shapeInfo.needPad) {
        shapeInfo.needReshape = true;
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.s1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyValueShape.assign(
            {shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.s2, shapeInfo.axes.d});
    }
}

static aclnnStatus analysisInputShapeInfo(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                          char *inputLayout, int64_t headNum, FaShapeInfo &shapeInfo,
                                          const aclTensor *attentionOut)
{
    if (headNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "head_num must > 0, but got %ld", headNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckDimsAndLayout(query, key, value, inputLayout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(analysisAxis(query, key, inputLayout, headNum, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (shapeInfo.axes.n2 == 0 || shapeInfo.axes.s2 == 0 || shapeInfo.axes.d == 0) {
        return ACLNN_SUCCESS;
    }

    // 根据dtype计算D补齐时，考虑输入q, k, v和输出，只要有int8类型，均按32元素对齐
    DataType queryDataType = query->GetDataType();
    DataType keyDataType = key->GetDataType();
    DataType valueDataType = value->GetDataType();
    DataType outputDataType = attentionOut->GetDataType();
    if ((queryDataType == DataType::DT_INT8) || (keyDataType == DataType::DT_INT8) ||
        (valueDataType == DataType::DT_INT8) || (outputDataType == DataType::DT_INT8)) {
        shapeInfo.basicBlock = PAD_BASIC_BLOCK;
    }

    if (shapeInfo.axes.d % shapeInfo.basicBlock != 0) {
        shapeInfo.needPad = true;
        shapeInfo.padNum =
            (shapeInfo.axes.d + shapeInfo.basicBlock - 1) / shapeInfo.basicBlock * shapeInfo.basicBlock -
            shapeInfo.axes.d;
    }

    if ((shapeInfo.inputLayout == InputLayout::BSH) ||
        (shapeInfo.inputLayout == InputLayout::SH)) {
        SetShapeInfoForSH(shapeInfo);
    } else if (shapeInfo.inputLayout == InputLayout::NSD) {
        SetShapeInfoForNSD(shapeInfo);
    }

    OP_LOGD("Analysis input success. The analysis result: [needReshape]: %d, [needPad]: %d, [padNum]: %lu,"
        "[needTranspose]: %d, [basicBlock]: %lu ",
        shapeInfo.needReshape, shapeInfo.needPad, shapeInfo.padNum, shapeInfo.needTranspose, shapeInfo.basicBlock);
    return ACLNN_SUCCESS;
}

static inline const aclTensor *GeneratePaddings(int32_t dimNum, int32_t padNum, aclOpExecutor *executor)
{
    // 2代表每根轴的前后都可以补0
    FVector<int64_t> padVec(dimNum * 2, 0);
    padVec[padVec.size() - 1] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), padVec.size());
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

static aclnnStatus contiguousInput(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                   const aclTensor *&pseShift, const aclTensor *&attenMask,
                                   aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (pseShift) {
        pseShift = l0op::Contiguous(pseShift, executor);
        CHECK_RET(pseShift != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (attenMask) {
        attenMask = l0op::Contiguous(attenMask, executor);
        CHECK_RET(attenMask != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus reShapeMiddle(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                 const int64_t *queryValue, uint64_t querySize,
                                 const int64_t *keyValueValue, uint64_t keyValueSize,
                                 aclOpExecutor *executor)
{
    query = l0op::Reshape(query, executor->AllocIntArray(queryValue, querySize), executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
    key = l0op::Reshape(key, executor->AllocIntArray(keyValueValue, keyValueSize), executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
    value = l0op::Reshape(value, executor->AllocIntArray(keyValueValue, keyValueSize), executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus preprocessQKVInput(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                      const aclTensor *&quantScale2, const aclTensor *&quantOffset2,
                                      const struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.needReshape) {
        query = l0op::Reshape(
            query, executor->AllocIntArray(shapeInfo.reshapedQueryShape.data(), shapeInfo.reshapedQueryShape.size()),
            executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
        key = l0op::Reshape(
            key,
            executor->AllocIntArray(shapeInfo.reshapedKeyValueShape.data(), shapeInfo.reshapedKeyValueShape.size()),
            executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
        value = l0op::Reshape(
            value,
            executor->AllocIntArray(shapeInfo.reshapedKeyValueShape.data(), shapeInfo.reshapedKeyValueShape.size()),
            executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.needPad) {
        auto paddings = GeneratePaddings(DIM_NUM_4, shapeInfo.padNum, executor);

        query = l0op::Pad(query, paddings, executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);
        key = l0op::Pad(key, paddings, executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_INNER_NULLPTR);
        value = l0op::Pad(value, paddings, executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto quant_paddings = paddings;
        if (quantScale2 != nullptr) {
            auto scale2DimNum = quantScale2->GetViewShape().GetDimNum();
            if (scale2DimNum == DIM_NUM_3) {
                quant_paddings = GeneratePaddings(DIM_NUM_3, shapeInfo.padNum, executor);
            }
            if (scale2DimNum == DIM_NUM_3 || scale2DimNum == DIM_NUM_4) {
                quantScale2 = l0op::Pad(quantScale2, quant_paddings, executor);
                CHECK_RET(quantScale2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
                quantOffset2 = l0op::Pad(quantOffset2, quant_paddings, executor);
                CHECK_RET(quantOffset2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }
    }

    if (shapeInfo.inputLayout == InputLayout::BSH && shapeInfo.needPad) {
        // (B,S,N,D) -> (B,S,N*D)
        FVector<int64_t, DIM_NUM_3> queryShape{shapeInfo.axes.b, shapeInfo.axes.s1,
                                               shapeInfo.axes.n1 * (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};
        FVector<int64_t, DIM_NUM_3> keyValueShape{shapeInfo.axes.b, shapeInfo.axes.s2,
                                                  shapeInfo.axes.n2 * (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};

        CHECK_RET(reShapeMiddle(query, key, value, queryShape.data(), queryShape.size(),
                                keyValueShape.data(), keyValueShape.size(), executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_TILING_ERROR);
    }

    if (shapeInfo.inputLayout == InputLayout::SH && shapeInfo.needPad) {
        // (B,S,N,D) -> (B*S,N*D)
        FVector<int64_t, DIM_NUM_2> queryShape{shapeInfo.axes.b * shapeInfo.axes.s1,
                                               shapeInfo.axes.n1 * (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};
        FVector<int64_t, DIM_NUM_2> keyValueShape{shapeInfo.axes.b * shapeInfo.axes.s2,
                                                  shapeInfo.axes.n2 * (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};

        CHECK_RET(reShapeMiddle(query, key, value, queryShape.data(), queryShape.size(),
                                keyValueShape.data(), keyValueShape.size(), executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_TILING_ERROR);
    }

    if (shapeInfo.inputLayout == InputLayout::NSD && shapeInfo.needPad) {
        // (B,N,S,Dp) -> (B*N,S,Dp)
        FVector<int64_t, DIM_NUM_3> queryShape{shapeInfo.axes.b * shapeInfo.axes.n1, shapeInfo.axes.s1,
                                               (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};
        FVector<int64_t, DIM_NUM_3> keyValueShape{shapeInfo.axes.b * shapeInfo.axes.n2, shapeInfo.axes.s2, 
                                                  (shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum))};

        CHECK_RET(reShapeMiddle(query, key, value, queryShape.data(), queryShape.size(),
                                keyValueShape.data(), keyValueShape.size(), executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_TILING_ERROR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus postProcessOutput(const aclTensor *&l0AttentionOutOut, const aclTensor *attentionOutOut,
                                     struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.needPad) {
        if ((shapeInfo.inputLayout == InputLayout::BSH) ||
            (shapeInfo.inputLayout == InputLayout::SH)) {
            // (B,S,Hp) -> (B,S,N,Dp)
            FVector<int64_t, DIM_NUM_4> paddedSBNDShape{shapeInfo.axes.b, shapeInfo.axes.s1, shapeInfo.axes.n1,
                                                shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum)};
            l0AttentionOutOut =
                l0op::Reshape(l0AttentionOutOut,
                            executor->AllocIntArray(paddedSBNDShape.data(), paddedSBNDShape.size()), executor);
            CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else if (shapeInfo.inputLayout == InputLayout::NSD) {
            // (N,S,Dp) -> (B,N,S,Dp)
            FVector<int64_t, DIM_NUM_4> paddedSBNDShape{shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.s1,
                                                shapeInfo.axes.d + static_cast<int64_t>(shapeInfo.padNum)};
            l0AttentionOutOut =
                l0op::Reshape(l0AttentionOutOut,
                            executor->AllocIntArray(paddedSBNDShape.data(), paddedSBNDShape.size()), executor);
            CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    if (shapeInfo.needPad) {
        // (B,S,N,D)
        // (S,B,N,D)
        // (B,N,S,D)
        FVector<int64_t, DIM_NUM_4> offsetVec(DIM_NUM_4, 0);
        FVector<int64_t, MAX_DIM_NUM> sizeVec = ToShapeVector(l0AttentionOutOut->GetViewShape());
        sizeVec.back() -= shapeInfo.padNum;

        l0AttentionOutOut = l0op::Slice(l0AttentionOutOut, executor->AllocIntArray(offsetVec.data(), offsetVec.size()),
                                        executor->AllocIntArray(sizeVec.data(), sizeVec.size()), executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (shapeInfo.needReshape) {
        auto attentionOutOutShape = ToShapeVector(attentionOutOut->GetViewShape());
        l0AttentionOutOut =
            l0op::Reshape(l0AttentionOutOut,
                          executor->AllocIntArray(attentionOutOutShape.data(), attentionOutOutShape.size()), executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static bool CheckNotNull(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                         char *inputLayout, const aclTensor *attentionOut) {
    OP_CHECK_NULL(query, return false);
    OP_CHECK_NULL(key, return false);
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(attentionOut, return false);
    if (inputLayout == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "expected a value of type char but got null for argument inputLayout.");
        return false;
    }
    return true;
}

static bool CheckTensorDataType(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                const aclTensor* attenMask, const aclTensor* attentionOut) {
    DataType queryDataType = query->GetDataType();
    DataType keyDataType = key->GetDataType();
    DataType valueDataType = value->GetDataType();
    DataType outputDataType = attentionOut->GetDataType();

    // 当前 PFA支持场景，q k v datatype需要一致
    if ((queryDataType != keyDataType) || (queryDataType != valueDataType)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Please check input Tensor datatype. "
            "The combination of [queryDataType]: %s, [keyDataType]: %s, [valueDataType]: %s is not supported by PFA.",
            StrDataTypePfa[queryDataType].c_str(), StrDataTypePfa[keyDataType].c_str(), StrDataTypePfa[valueDataType].c_str());
        return false;
    }

    // 当前仅在量化相关场景（int8进/fp16出 或 fp16进/int8出）时输入输出dtype不同
    if (queryDataType != outputDataType) {
        bool isQuant = ((queryDataType == DataType::DT_INT8 && outputDataType == DataType::DT_FLOAT16) || 
        (queryDataType == DataType::DT_FLOAT16 && outputDataType == DataType::DT_INT8) ||
        (queryDataType == DataType::DT_BF16 && outputDataType == DataType::DT_INT8));
        if (!isQuant) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Please check input/output Tensor datatype. "
                "The combination of [queryDataType]: %s, [outputDataType]: %s is not supported by PFA.",
                StrDataTypePfa[queryDataType].c_str(), StrDataTypePfa[outputDataType].c_str());
            return false;
        }
    }

    // int8量化场景，不支持 fp16类型 atten mask 
    if (attenMask != nullptr) {
        DataType attenMaskDataType = attenMask->GetDataType();
        if ((queryDataType == DataType::DT_INT8) && (attenMaskDataType == DataType::DT_FLOAT16)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Please check Tensor datatype. "
                "When input tensor datatype is %s, attenMaskDataType can not be %s.",
                StrDataTypePfa[queryDataType].c_str(), StrDataTypePfa[attenMaskDataType].c_str());
            return false;
        }
    }

    return true;
}

static inline bool CheckResultOutShapePfa(const aclTensor *inferOut, const aclTensor *out) {
    auto const &xShape = inferOut->GetViewShape();
    auto const &yShape = out->GetViewShape();
    if(xShape != yShape) {
        if(!(xShape.GetShapeSize() == 1 && yShape.GetShapeSize() == 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out tensor's shape[%s] is not equal with inferOut shape[%s].",
                op::ToString(out->GetViewShape()).GetString(), op::ToString(inferOut->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

aclnnStatus aclnnInnerPromptFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens,
    char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnInnerPromptFlashAttention,
                DFX_IN(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv,
                        deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
                        numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads,
                        sparseMode, innerPrecise),
                DFX_OUT(attentionOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 检查必选输入指针是否为空
    CHECK_RET(CheckNotNull(query, key, value, inputLayout, attentionOut), ACLNN_ERR_PARAM_NULLPTR);

    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (attentionOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(CheckTensorDataType(query, key, value, attenMask, attentionOut), ACLNN_ERR_PARAM_INVALID);

    FaShapeInfo shapeInfo;
    CHECK_RET(analysisInputShapeInfo(query, key, value, inputLayout, numHeads, shapeInfo, attentionOut) ==
              ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(contiguousInput(query, key, value, pseShift, attenMask, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(preprocessQKVInput(query, key, value, quantScale2, quantOffset2, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto l0AttentionOutOut = l0op::PromptFlashAttention(query, key, value, pseShift, attenMask,
                                                        actualSeqLengths, actualSeqLengthsKv,
                                                        deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
                                                        numHeads, scaleValue, preTokens, nextTokens,
                                                        inputLayout,
                                                        numKeyValueHeads, sparseMode, innerPrecise,
                                                        attentionOut, l0Executor);
    CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(postProcessOutput(l0AttentionOutOut, attentionOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckResultOutShapePfa(l0AttentionOutOut, attentionOut), ACLNN_ERR_PARAM_INVALID);
    auto viewCopyResult = l0op::ViewCopy(l0AttentionOutOut, attentionOut, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInnerPromptFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInnerPromptFlashAttention);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // namespace

#ifdef __cplusplus
}
#endif