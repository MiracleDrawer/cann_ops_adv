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
 * \file aclnn_ffn_case.cpp
 * \brief FFN Aclnn 测试用例.
 */

#include <utility>
#include "tests/utils/log.h"
#include "aclnn_ffn.h"
#include "aclnn_ffn_v2.h"
#include "aclnn_ffn_v3.h"
#include "aclnn_ffn_case.h"

using namespace ops::adv::tests::ffn;

bool FFNTilingRunCbf(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor)
{
    auto *cs = static_cast<AclnnFFNCase *>(curCase);
    auto *aclnnParam = &cs->aclnnParam;

    aclnnStatus ret;
    if (aclnnParam->aclnnFFNVersion == AclnnFFNVersion::V1) {
        ret = aclnnFFNGetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensor(), aclnnParam->aclnnWeight1.GetAclTensor(),
            aclnnParam->aclnnWeight2.GetAclTensor(), aclnnParam->aclnnExpertTokensIntAry,
            aclnnParam->aclnnBias1.GetAclTensor(), aclnnParam->aclnnBias2.GetAclTensor(),
            aclnnParam->aclnnScale.GetAclTensor(), aclnnParam->aclnnOffset.GetAclTensor(),
            aclnnParam->aclnnDeqScale1.GetAclTensor(), aclnnParam->aclnnDeqScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantScale1.GetAclTensor(), aclnnParam->aclnnAntiquantScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantOffset1.GetAclTensor(), aclnnParam->aclnnAntiquantOffset2.GetAclTensor(),
            aclnnParam->activation.c_str(), aclnnParam->innerPrecise, aclnnParam->aclnnY.GetAclTensor(), workSpaceSize,
            opExecutor);
    } else if (aclnnParam->aclnnFFNVersion == AclnnFFNVersion::V2) {
        ret = aclnnFFNV2GetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensor(), aclnnParam->aclnnWeight1.GetAclTensor(),
            aclnnParam->aclnnWeight2.GetAclTensor(), aclnnParam->aclnnExpertTokensIntAry,
            aclnnParam->aclnnBias1.GetAclTensor(), aclnnParam->aclnnBias2.GetAclTensor(),
            aclnnParam->aclnnScale.GetAclTensor(), aclnnParam->aclnnOffset.GetAclTensor(),
            aclnnParam->aclnnDeqScale1.GetAclTensor(), aclnnParam->aclnnDeqScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantScale1.GetAclTensor(), aclnnParam->aclnnAntiquantScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantOffset1.GetAclTensor(), aclnnParam->aclnnAntiquantOffset2.GetAclTensor(),
            aclnnParam->activation.c_str(), aclnnParam->innerPrecise, aclnnParam->tokensIndexFlag,
            aclnnParam->aclnnY.GetAclTensor(), workSpaceSize, opExecutor);
    } else if (aclnnParam->aclnnFFNVersion == AclnnFFNVersion::V3) {
        ret = aclnnFFNV3GetWorkspaceSize(
            aclnnParam->aclnnX.GetAclTensor(), aclnnParam->aclnnWeight1.GetAclTensor(),
            aclnnParam->aclnnWeight2.GetAclTensor(), aclnnParam->aclnnExpertTokens.GetAclTensor(),
            aclnnParam->aclnnBias1.GetAclTensor(), aclnnParam->aclnnBias2.GetAclTensor(),
            aclnnParam->aclnnScale.GetAclTensor(), aclnnParam->aclnnOffset.GetAclTensor(),
            aclnnParam->aclnnDeqScale1.GetAclTensor(), aclnnParam->aclnnDeqScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantScale1.GetAclTensor(), aclnnParam->aclnnAntiquantScale2.GetAclTensor(),
            aclnnParam->aclnnAntiquantOffset1.GetAclTensor(), aclnnParam->aclnnAntiquantOffset2.GetAclTensor(),
            aclnnParam->activation.c_str(), aclnnParam->innerPrecise, aclnnParam->tokensIndexFlag,
            aclnnParam->aclnnY.GetAclTensor(), workSpaceSize, opExecutor);
    }

    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFFNGetWorkspaceSize failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

bool FFNKernelRunCbf(void *curCase)
{
    auto *cs = static_cast<AclnnFFNCase *>(curCase);
    auto *aclnnParam = &cs->aclnnParam;
    auto *aclnnCtx = &cs->aclnnCtx;

    aclnnStatus ret;
    if (aclnnParam->aclnnFFNVersion == AclnnFFNVersion::V1) {
        ret = aclnnFFN(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                       aclnnCtx->GetAclRtStream());
    } else if (aclnnParam->aclnnFFNVersion == AclnnFFNVersion::V2) {
        ret = aclnnFFNV2(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                         aclnnCtx->GetAclRtStream());
    } else if (aclnnParam->aclnnFFNVersion == AclnnFFNVersion::V3) {
        ret = aclnnFFNV3(aclnnCtx->GetWorkspacePtr(), aclnnCtx->GetWorkspaceSize(), aclnnCtx->GetAclOpExecutor(),
                         aclnnCtx->GetAclRtStream());
    }
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclnnFFN failed, ERROR: %d", ret));

    return ret == ACL_SUCCESS;
}

AclnnFFNCase::AclnnFFNCase() : FFNCase(), aclnnCtx(AclnnContext()), aclnnParam(AclnnFFNParam())
{
}

AclnnFFNCase::AclnnFFNCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, AclnnFFNParam aclnnParam,
                           int32_t tilingTemplatePriority)
    : FFNCase(name, enable, dbgInfo, std::move(opInfo), Param(), tilingTemplatePriority),
      aclnnParam(std::move(aclnnParam))
{
}

bool AclnnFFNCase::InitParam()
{
    return aclnnParam.Init();
}

bool AclnnFFNCase::InitOpInfo()
{
    if (!FFNCase::InitOpInfo()) {
        return false;
    }

    auto rst = aclnnCtx.SetOpName(this->opInfo.name.c_str());
    rst = rst && aclnnCtx.SetTilingRunCbf(FFNTilingRunCbf);
    rst = rst && aclnnCtx.SetKernelRunCbf(FFNKernelRunCbf);
    rst = rst && aclnnCtx.SetOutputs({&aclnnParam.aclnnY});
    rst = rst && opInfo.SetContext(&aclnnCtx);
    return rst;
}

bool AclnnFFNCase::InitCurrentCasePtr()
{
    Case::currentCasePtr = this;
    return true;
}

bool AclnnFFNCase::Run()
{
    if (!enable) {
        return true;
    }
    if (!opInfo.ProcessTiling(name)) {
        return false;
    }
    if (!opInfo.ProcessKernel(name)) {
        return false;
    }
    return true;
}
