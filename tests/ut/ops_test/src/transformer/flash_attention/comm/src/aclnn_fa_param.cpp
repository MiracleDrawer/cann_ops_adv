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
 * \file aclnn_fa_param.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 参数信息.
 */

#include "aclnn_fa_param.h"
#include <utility>
#include "tests/utils/case.h"
#include "tests/utils/io.h"
#include "tests/utils/log.h"

namespace {
template <class T> bool InitAclIntArray(aclIntArray **intArray, std::vector<T> &hostData)
{
    if (intArray == nullptr) {
        LOG_ERR("intArray nil.");
        return false;
    }
    if (*intArray != nullptr) {
        auto ret = aclDestroyIntArray(*intArray);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclDestroyIntArray failed, ERROR: %d", ret), *intArray = nullptr);
    }
    if (hostData.empty()) {
        return true;
    }
    *intArray = aclCreateIntArray(hostData.data(), hostData.size());
    if (*intArray == nullptr) {
        LOG_ERR("aclCreateIntArray failed.");
        return false;
    }
    return true;
}
} // namespace

using namespace ops::adv::tests::fa;

AclnnFaParam::AclnnFaParam(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, ge::DataType dtype,
                           LayoutType layoutType, float scale, float keepProb, int64_t preTokens, int64_t nxtTokens,
                           int64_t innerPrecise, int64_t sparseMode, PseShapeType pseShapeType,
                           DropMaskShapeType dropMaskShapeType, PaddingMaskShapeType paddingMaskShapeType,
                           AttenMaskShapeType attenMaskShapeType, ge::DataType attenMaskDtype,
                           PrefixShapeType prefixShapeType)
    : AclnnFaParam(b, n2, g, s1, s2, d, dtype, layoutType, scale, keepProb, preTokens, nxtTokens, innerPrecise,
                   sparseMode, pseShapeType, dropMaskShapeType, paddingMaskShapeType, attenMaskShapeType,
                   attenMaskDtype, prefixShapeType, {}, {}, {})
{
}

AclnnFaParam::AclnnFaParam(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, ge::DataType dtype,
                           LayoutType layoutType, float scale, float keepProb, int64_t preTokens, int64_t nxtTokens,
                           int64_t innerPrecise, int64_t sparseMode, PseShapeType pseShapeType,
                           DropMaskShapeType dropMaskShapeType, PaddingMaskShapeType paddingMaskShapeType,
                           AttenMaskShapeType attenMaskShapeType, ge::DataType attenMaskDtype,
                           PrefixShapeType prefixShapeType, std::vector<int64_t> prefixTensorData,
                           std::vector<int64_t> actualSeqQLenTensorData, std::vector<int64_t> actualSeqKvLenTensorData)
    : FaParam(b, n2, g, s1, s2, d, dtype, layoutType, scale, keepProb, preTokens, nxtTokens, innerPrecise, sparseMode,
              pseShapeType, dropMaskShapeType, paddingMaskShapeType, attenMaskShapeType, attenMaskDtype,
              prefixShapeType, std::move(prefixTensorData), std::move(actualSeqQLenTensorData),
              std::move(actualSeqKvLenTensorData)),
      aclnnPrefixIntAry(nullptr), aclnnActualSeqQLenIntAry(nullptr), aclnnActualSeqKvLenIntAry(nullptr)
{
}

AclnnFaParam::AclnnFaParam(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, ge::DataType dtype,
                           LayoutType layoutType, float scale, float keepProb, int64_t preTokens, int64_t nxtTokens,
                           int64_t innerPrecise, int64_t sparseMode, int64_t pseType, PseShapeType pseShapeType,
                           DropMaskShapeType dropMaskShapeType, PaddingMaskShapeType paddingMaskShapeType,
                           AttenMaskShapeType attenMaskShapeType, ge::DataType attenMaskDtype,
                           PrefixShapeType prefixShapeType)
    : AclnnFaParam(b, n2, g, s1, s2, d, dtype, layoutType, scale, keepProb, preTokens, nxtTokens, innerPrecise,
                   sparseMode, pseType, pseShapeType, dropMaskShapeType, paddingMaskShapeType, attenMaskShapeType,
                   attenMaskDtype, prefixShapeType, {}, {}, {})
{
}

AclnnFaParam::AclnnFaParam(int64_t b, int64_t n2, int64_t g, int64_t s1, int64_t s2, int64_t d, ge::DataType dtype,
                           FaParam::LayoutType layoutType, float scale, float keepProb, int64_t preTokens,
                           int64_t nxtTokens, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
                           FaParam::PseShapeType pseShapeType, FaParam::DropMaskShapeType dropMaskShapeType,
                           FaParam::PaddingMaskShapeType paddingMaskShapeType,
                           FaParam::AttenMaskShapeType attenMaskShapeType, ge::DataType attenMaskDtype,
                           FaParam::PrefixShapeType prefixShapeType, std::vector<int64_t> prefixTensorData,
                           std::vector<int64_t> actualSeqQLenTensorData, std::vector<int64_t> actualSeqKvLenTensorData)
    : FaParam(b, n2, g, s1, s2, d, dtype, layoutType, scale, keepProb, preTokens, nxtTokens, innerPrecise, sparseMode,
              pseType, pseShapeType, dropMaskShapeType, paddingMaskShapeType, attenMaskShapeType, attenMaskDtype,
              prefixShapeType, std::move(prefixTensorData), std::move(actualSeqQLenTensorData),
              std::move(actualSeqKvLenTensorData), {}, {}),
      aclnnPrefixIntAry(nullptr), aclnnActualSeqQLenIntAry(nullptr), aclnnActualSeqKvLenIntAry(nullptr),
      qStartIdxOptionalIntAry(nullptr), kvStartIdxOptionalIntAry(nullptr)
{
}

AclnnFaParam::~AclnnFaParam()
{
    if (aclnnPrefixIntAry != nullptr) {
        auto ret = aclDestroyIntArray(aclnnPrefixIntAry);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclDestroyIntArray failed, ERROR: %d", ret),
                    aclnnPrefixIntAry = nullptr);
    }
    if (aclnnActualSeqQLenIntAry != nullptr) {
        auto ret = aclDestroyIntArray(aclnnActualSeqQLenIntAry);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclDestroyIntArray failed, ERROR: %d", ret),
                    aclnnActualSeqQLenIntAry = nullptr);
    }
    if (aclnnActualSeqKvLenIntAry != nullptr) {
        auto ret = aclDestroyIntArray(aclnnActualSeqKvLenIntAry);
        LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclDestroyIntArray failed, ERROR: %d", ret),
                    aclnnActualSeqKvLenIntAry = nullptr);
    }
}

bool AclnnFaParam::Init()
{
    if (!FaParam::Init()) {
        return false;
    }
    aclnnQuery = ops::adv::tests::utils::AclnnTensor(query);
    aclnnKey = ops::adv::tests::utils::AclnnTensor(key);
    aclnnValue = ops::adv::tests::utils::AclnnTensor(value);
    aclnnDy = ops::adv::tests::utils::AclnnTensor(dy);
    aclnnPse = ops::adv::tests::utils::AclnnTensor(pse);
    aclnnDropMask = ops::adv::tests::utils::AclnnTensor(dropMask);
    aclnnPaddingMask = ops::adv::tests::utils::AclnnTensor(paddingMask);
    aclnnAttenMask = ops::adv::tests::utils::AclnnTensor(attenMask);
    aclnnPrefix = ops::adv::tests::utils::AclnnTensor(prefix);
    aclnnSoftmaxMax = ops::adv::tests::utils::AclnnTensor(softmaxMax);
    aclnnSoftmaxSum = ops::adv::tests::utils::AclnnTensor(softmaxSum);
    aclnnSoftmaxRes = ops::adv::tests::utils::AclnnTensor(softmaxRes);
    aclnnAttenRes = ops::adv::tests::utils::AclnnTensor(attenRes);
    aclnnDq = ops::adv::tests::utils::AclnnTensor(dq);
    aclnnDk = ops::adv::tests::utils::AclnnTensor(dk);
    aclnnDv = ops::adv::tests::utils::AclnnTensor(dv);
    aclnnDpse = ops::adv::tests::utils::AclnnTensor(dPse);
    aclnnActualSeqQLen = ops::adv::tests::utils::AclnnTensor(actualSeqQLen);
    aclnnActualSeqKvLen = ops::adv::tests::utils::AclnnTensor(actualSeqKvLen);

    if (!InitAclIntArray(&aclnnPrefixIntAry, prefixTensorData)) {
        return false;
    }
    if (!InitAclIntArray(&aclnnActualSeqQLenIntAry, actualSeqQLenTensorData)) {
        return false;
    }
    if (!InitAclIntArray(&aclnnActualSeqKvLenIntAry, actualSeqKVLenTensorData)) {
        return false;
    }

    auto *cs = static_cast<ops::adv::tests::utils::Case *>(ops::adv::tests::utils::Case::GetCurrentCase());
    LOG_IF_EXPR(cs == nullptr, LOG_ERR("Can't get current case"), return false);

    for (auto *t : {&aclnnQuery, &aclnnKey, &aclnnValue, &aclnnDy, &aclnnPse, &aclnnDropMask, &aclnnPaddingMask,
                    &aclnnAttenMask, &aclnnSoftmaxMax, &aclnnSoftmaxSum, &aclnnSoftmaxRes, &aclnnAttenRes, &aclnnDq,
                    &aclnnDk, &aclnnDv, &aclnnDpse}) {
        t->FreeDevData();
        if (t->GetExpDataSize() <= 0) {
            continue;
        }
        auto *devData = t->AllocDevData(0, 0);
        if (devData == nullptr) {
            return false;
        }
        if (t->IsOutput()) {
            continue;
        }
        std::string filePath = cs->rootPath + t->Name() + ".bin";
        if (ops::adv::tests::utils::FileExist(filePath)) {
            if (!t->LoadFileToDevData(filePath)) {
                return false;
            }
        }
    }
    return true;
}

bool AclnnFaParam::IsUnPaddingAttention()
{
    if (!FaParam::IsUnPaddingAttention()) {
        return false;
    }
    aclnnStatus ret;
    uint64_t aclnnActualSeqQLenIntArySize = 0;
    uint64_t aclnnActualSeqKVLenIntArySize = 0;
    ret = aclGetIntArraySize(aclnnActualSeqQLenIntAry, &aclnnActualSeqQLenIntArySize);
    LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclGetIntArraySize failed, ERROR: %d", ret), return false);
    ret = aclGetIntArraySize(aclnnActualSeqKvLenIntAry, &aclnnActualSeqKVLenIntArySize);
    LOG_IF_EXPR(ret != ACL_SUCCESS, LOG_ERR("aclGetIntArraySize failed, ERROR: %d", ret), return false);
    return aclnnActualSeqQLenIntArySize != 0 && aclnnActualSeqKVLenIntArySize != 0;
}
