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
 * \file aclnn_fa_param.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 参数信息.
 */

#pragma once

#include "fa_case.h"
#include "tests/utils/aclnn_tensor.h"

namespace ops::adv::tests::fa {

class AclnnFaParam : public ops::adv::tests::fa::FaParam {
public:
    using AclnnTensor = ops::adv::tests::utils::AclnnTensor;

public:
    /* 输入输出 */
    AclnnTensor aclnnQuery, aclnnKey, aclnnValue, aclnnDy, aclnnPse, aclnnDropMask, aclnnPaddingMask, aclnnAttenMask,
        aclnnPrefix, aclnnSoftmaxMax, aclnnSoftmaxSum, aclnnSoftmaxRes, aclnnAttenRes, aclnnDq, aclnnDk, aclnnDv,
        aclnnDpse, aclnnActualSeqQLen, aclnnActualSeqKvLen;
    aclIntArray *aclnnPrefixIntAry = nullptr;
    aclIntArray *aclnnActualSeqQLenIntAry = nullptr;
    aclIntArray *aclnnActualSeqKvLenIntAry = nullptr;

public:
    AclnnFaParam() = default;
    AclnnFaParam(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, ge::DataType dtype,
                 LayoutType layoutType, float scale, float keepProb, int64_t preTokens, int64_t nxtTokens,
                 int64_t innerPrecise, int64_t sparseMode, PseShapeType pseShapeType,
                 DropMaskShapeType dropMaskShapeType, PaddingMaskShapeType paddingMaskShapeType,
                 AttenMaskShapeType attenMaskShapeType, ge::DataType attenMaskDtype, PrefixShapeType prefixShapeType);
    AclnnFaParam(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, ge::DataType dtype,
                 LayoutType layoutType, float scale, float keepProb, int64_t preTokens, int64_t nxtTokens,
                 int64_t innerPrecise, int64_t sparseMode, PseShapeType pseShapeType,
                 DropMaskShapeType dropMaskShapeType, PaddingMaskShapeType paddingMaskShapeType,
                 AttenMaskShapeType attenMaskShapeType, ge::DataType attenMaskDtype, PrefixShapeType prefixShapeType,
                 std::vector<int64_t> prefixTensorData, std::vector<int64_t> actualSeqQLenTensorData,
                 std::vector<int64_t> actualSeqKvLenTensorData);

    ~AclnnFaParam();

    bool Init() override;

    bool IsUnPaddingAttention() override;
};

} // namespace ops::adv::tests::fa
