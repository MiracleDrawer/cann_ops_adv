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
 * \file aclnn_fa_case.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 测试用例.
 */

#include <utility>
#include "aclnn_fa_case.h"
#include "tests/utils/log.h"
#include "aclnnop/aclnn_flash_attention_score.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

using namespace ops::adv::tests::fa;

bool FasTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnFaCase *>(curCase);
    auto *aclnnParam = &cs->aclnnParam;
    auto *exp = &cs->forward.exp;

    aclnnStatus ret;
    if (!aclnnParam->IsUnPaddingAttention()) {
        ret = aclnnFlashAttentionScoreGetWorkspaceSize(
            aclnnParam->aclnnQuery.GetAclTensor(), aclnnParam->aclnnKey.GetAclTensor(),
            aclnnParam->aclnnValue.GetAclTensor(), aclnnParam->aclnnPse.GetAclTensor(),
            aclnnParam->aclnnDropMask.GetAclTensor(), aclnnParam->aclnnPaddingMask.GetAclTensor(),
            aclnnParam->aclnnAttenMask.GetAclTensor(), aclnnParam->aclnnPrefixIntAry, aclnnParam->scale,
            aclnnParam->keepProb, aclnnParam->preTokens, aclnnParam->nxtTokens, aclnnParam->n1,
            (char *)aclnnParam->layout.c_str(), aclnnParam->innerPrecise, aclnnParam->sparseMode,
            aclnnParam->aclnnSoftmaxMax.GetAclTensor(), aclnnParam->aclnnSoftmaxSum.GetAclTensor(),
            aclnnParam->aclnnSoftmaxRes.GetAclTensor(), aclnnParam->aclnnAttenRes.GetAclTensor(), workSpaceSize,
            opExecutor);
        LOG_IF(ret != ACL_SUCCESS && exp->success,
               LOG_ERR("aclnnFlashAttentionScoreGetWorkspaceSize failed, ERROR: %d", ret));
    } else {
        ret = aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
            aclnnParam->aclnnQuery.GetAclTensor(), aclnnParam->aclnnKey.GetAclTensor(),
            aclnnParam->aclnnValue.GetAclTensor(), aclnnParam->aclnnPse.GetAclTensor(),
            aclnnParam->aclnnDropMask.GetAclTensor(), aclnnParam->aclnnPaddingMask.GetAclTensor(),
            aclnnParam->aclnnAttenMask.GetAclTensor(), aclnnParam->aclnnPrefixIntAry,
            aclnnParam->aclnnActualSeqQLenIntAry, aclnnParam->aclnnActualSeqKvLenIntAry, aclnnParam->scale,
            aclnnParam->keepProb, aclnnParam->preTokens, aclnnParam->nxtTokens, aclnnParam->n1,
            (char *)aclnnParam->layout.c_str(), aclnnParam->innerPrecise, aclnnParam->sparseMode,
            aclnnParam->aclnnSoftmaxMax.GetAclTensor(), aclnnParam->aclnnSoftmaxSum.GetAclTensor(),
            aclnnParam->aclnnSoftmaxRes.GetAclTensor(), aclnnParam->aclnnAttenRes.GetAclTensor(), workSpaceSize,
            opExecutor);
        LOG_IF(ret != ACL_SUCCESS && exp->success,
               LOG_ERR("aclnnFlashAttentionVarLenScoreGetWorkspaceSize failed, ERROR: %d", ret));
    }
    return ret == ACL_SUCCESS;
}

bool FasKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnFaCase *>(curCase);
    auto *aclnnParam = &cs->aclnnParam;
    auto *ctx = &cs->aclnnForwardCtx;

    aclnnStatus ret;
    if (!aclnnParam->IsUnPaddingAttention()) {
        ret = aclnnFlashAttentionScore(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(), ctx->GetAclOpExecutor(),
                                       ctx->GetAclRtStream());
        LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionScore failed, ERROR: %d", ret));
    } else {
        ret = aclnnFlashAttentionVarLenScore(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(), ctx->GetAclOpExecutor(),
                                             ctx->GetAclRtStream());
        LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionVarLenScore failed, ERROR: %d", ret));
    }
    return ret == ACL_SUCCESS;
}

bool FagTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnFaCase *>(curCase);
    auto *aclnnParam = &cs->aclnnParam;
    auto *exp = &cs->reverse.exp;

    aclnnStatus ret;
    if (aclnnParam->pseType == 1) {
        if (!aclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreGradGetWorkspaceSize(
                aclnnParam->aclnnQuery.GetAclTensor(), aclnnParam->aclnnKey.GetAclTensor(),
                aclnnParam->aclnnValue.GetAclTensor(), aclnnParam->aclnnDy.GetAclTensor(),
                aclnnParam->aclnnPse.GetAclTensor(), aclnnParam->aclnnDropMask.GetAclTensor(),
                aclnnParam->aclnnPaddingMask.GetAclTensor(), aclnnParam->aclnnAttenMask.GetAclTensor(),
                aclnnParam->aclnnSoftmaxMax.GetAclTensor(), aclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                aclnnParam->aclnnSoftmaxRes.GetAclTensor(), aclnnParam->aclnnAttenRes.GetAclTensor(),
                aclnnParam->aclnnPrefixIntAry, aclnnParam->scale, aclnnParam->keepProb, aclnnParam->preTokens,
                aclnnParam->nxtTokens, aclnnParam->n1, (char *)aclnnParam->layout.c_str(), aclnnParam->innerPrecise,
                aclnnParam->sparseMode, aclnnParam->aclnnDq.GetAclTensor(), aclnnParam->aclnnDk.GetAclTensor(),
                aclnnParam->aclnnDv.GetAclTensor(), aclnnParam->aclnnDpse.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->success,
                   LOG_ERR("aclnnFlashAttentionScoreGradGetWorkspaceSize failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize(
                aclnnParam->aclnnQuery.GetAclTensor(), aclnnParam->aclnnKey.GetAclTensor(),
                aclnnParam->aclnnValue.GetAclTensor(), aclnnParam->aclnnDy.GetAclTensor(),
                aclnnParam->aclnnPse.GetAclTensor(), aclnnParam->aclnnDropMask.GetAclTensor(),
                aclnnParam->aclnnPaddingMask.GetAclTensor(), aclnnParam->aclnnAttenMask.GetAclTensor(),
                aclnnParam->aclnnSoftmaxMax.GetAclTensor(), aclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                aclnnParam->aclnnSoftmaxRes.GetAclTensor(), aclnnParam->aclnnAttenRes.GetAclTensor(),
                aclnnParam->aclnnPrefixIntAry, aclnnParam->aclnnActualSeqQLenIntAry,
                aclnnParam->aclnnActualSeqKvLenIntAry, aclnnParam->scale, aclnnParam->keepProb, aclnnParam->preTokens,
                aclnnParam->nxtTokens, aclnnParam->n1, (char *)aclnnParam->layout.c_str(), aclnnParam->innerPrecise,
                aclnnParam->sparseMode, aclnnParam->aclnnDq.GetAclTensor(), aclnnParam->aclnnDk.GetAclTensor(),
                aclnnParam->aclnnDv.GetAclTensor(), aclnnParam->aclnnDpse.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->success,
                   LOG_ERR("aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize failed, ERROR: %d", ret));
        }
    } else {
        if (!aclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreGradV2GetWorkspaceSize(
                aclnnParam->aclnnQuery.GetAclTensor(), aclnnParam->aclnnKey.GetAclTensor(),
                aclnnParam->aclnnValue.GetAclTensor(), aclnnParam->aclnnDy.GetAclTensor(),
                aclnnParam->aclnnPse.GetAclTensor(), aclnnParam->aclnnDropMask.GetAclTensor(),
                aclnnParam->aclnnPaddingMask.GetAclTensor(), aclnnParam->aclnnAttenMask.GetAclTensor(),
                aclnnParam->aclnnSoftmaxMax.GetAclTensor(), aclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                aclnnParam->aclnnSoftmaxRes.GetAclTensor(), aclnnParam->aclnnAttenRes.GetAclTensor(),
                aclnnParam->aclnnPrefixIntAry, aclnnParam->qStartIdxOptionalIntAry,
                aclnnParam->kvStartIdxOptionalIntAry, aclnnParam->scale, aclnnParam->keepProb, aclnnParam->preTokens,
                aclnnParam->nxtTokens, aclnnParam->n1, (char *)aclnnParam->layout.c_str(), aclnnParam->innerPrecise,
                aclnnParam->sparseMode, aclnnParam->pseType, aclnnParam->aclnnDq.GetAclTensor(),
                aclnnParam->aclnnDk.GetAclTensor(), aclnnParam->aclnnDv.GetAclTensor(),
                aclnnParam->aclnnDpse.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->success,
                   LOG_ERR("aclnnFlashAttentionScoreGradV2GetWorkspaceSize failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(
                aclnnParam->aclnnQuery.GetAclTensor(), aclnnParam->aclnnKey.GetAclTensor(),
                aclnnParam->aclnnValue.GetAclTensor(), aclnnParam->aclnnDy.GetAclTensor(),
                aclnnParam->aclnnPse.GetAclTensor(), aclnnParam->aclnnDropMask.GetAclTensor(),
                aclnnParam->aclnnPaddingMask.GetAclTensor(), aclnnParam->aclnnAttenMask.GetAclTensor(),
                aclnnParam->aclnnSoftmaxMax.GetAclTensor(), aclnnParam->aclnnSoftmaxSum.GetAclTensor(),
                aclnnParam->aclnnSoftmaxRes.GetAclTensor(), aclnnParam->aclnnAttenRes.GetAclTensor(),
                aclnnParam->aclnnPrefixIntAry, aclnnParam->aclnnActualSeqQLenIntAry,
                aclnnParam->aclnnActualSeqKvLenIntAry, aclnnParam->qStartIdxOptionalIntAry,
                aclnnParam->kvStartIdxOptionalIntAry, aclnnParam->scale, aclnnParam->keepProb, aclnnParam->preTokens,
                aclnnParam->nxtTokens, aclnnParam->n1, (char *)aclnnParam->layout.c_str(), aclnnParam->innerPrecise,
                aclnnParam->sparseMode, aclnnParam->pseType, aclnnParam->aclnnDq.GetAclTensor(),
                aclnnParam->aclnnDk.GetAclTensor(), aclnnParam->aclnnDv.GetAclTensor(),
                aclnnParam->aclnnDpse.GetAclTensor(), workSpaceSize, opExecutor);
            LOG_IF(ret != ACL_SUCCESS && exp->success,
                   LOG_ERR("aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize failed, ERROR: %d", ret));
        }
    }
    return ret == ACL_SUCCESS;
}

bool FagKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnFaCase *>(curCase);
    auto *aclnnParam = &cs->aclnnParam;
    auto *ctx = &cs->aclnnReverseCtx;

    aclnnStatus ret;
    if (aclnnParam->pseType == 1) {
        if (!aclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreGrad(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(), ctx->GetAclOpExecutor(),
                                               ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionScoreGrad failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionUnpaddingScoreGrad(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(),
                                                        ctx->GetAclOpExecutor(), ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionUnpaddingScoreGrad failed, ERROR: %d", ret));
        }
    } else {
        if (!aclnnParam->IsUnPaddingAttention()) {
            ret = aclnnFlashAttentionScoreGradV2(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(),
                                                 ctx->GetAclOpExecutor(), ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionScoreGradV2 failed, ERROR: %d", ret));
        } else {
            ret = aclnnFlashAttentionUnpaddingScoreGradV2(ctx->GetWorkspacePtr(), ctx->GetWorkspaceSize(),
                                                          ctx->GetAclOpExecutor(), ctx->GetAclRtStream());
            LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFlashAttentionUnpaddingScoreGradV2 failed, ERROR: %d", ret));
        }
    }
    return ret == ACL_SUCCESS;
}

AclnnFaCase::AclnnFaCase()
    : FaCase(), aclnnForwardCtx(AclnnContext()), aclnnReverseCtx(AclnnContext()), aclnnParam(AclnnFaParam())
{
}

AclnnFaCase::AclnnFaCase(const char *name, bool enable, const char *dbgInfo, OpInfo forward, OpInfo reverse,
                         const AclnnFaParam &aclnnParam, int32_t tilingTemplatePriority)
    : FaCase(name, enable, dbgInfo, std::move(forward), std::move(reverse), FaParam(), tilingTemplatePriority),
      aclnnParam(aclnnParam)
{
}

bool AclnnFaCase::InitParam()
{
    return aclnnParam.Init();
}

bool AclnnFaCase::InitOpInfo()
{
    if (!FaCase::InitOpInfo()) {
        return false;
    }
    auto rst = aclnnForwardCtx.SetOpName(this->forward.name.c_str());
    rst = rst && aclnnForwardCtx.SetTilingRunCbf(FasTilingRunCbf);
    rst = rst && aclnnForwardCtx.SetKernelRunCbf(FasKernelRunCbf);
    rst = rst && aclnnForwardCtx.SetOutputs({&aclnnParam.aclnnSoftmaxMax, &aclnnParam.aclnnSoftmaxSum,
                                             &aclnnParam.aclnnSoftmaxRes, &aclnnParam.aclnnAttenRes});
    rst = rst && forward.SetContext(&aclnnForwardCtx);
    rst = rst && aclnnReverseCtx.SetOpName(this->reverse.name.c_str());
    rst = rst && aclnnReverseCtx.SetTilingRunCbf(FagTilingRunCbf);
    rst = rst && aclnnReverseCtx.SetKernelRunCbf(FagKernelRunCbf);
    rst = rst && aclnnReverseCtx.SetOutputs(
                     {&aclnnParam.aclnnDq, &aclnnParam.aclnnDk, &aclnnParam.aclnnDv, &aclnnParam.aclnnDpse});
    rst = rst && reverse.SetContext(&aclnnReverseCtx);
    return rst;
}

bool AclnnFaCase::InitCurrentCasePtr()
{
    Case::currentCasePtr = this;
    return true;
}
