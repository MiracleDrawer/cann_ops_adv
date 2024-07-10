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
 * \file op_info.cpp
 * \brief 测试用例算子信息.
 */

#include "tests/utils/op_info.h"
#include "tests/utils/platform.h"
#include "tests/utils/log.h"

using namespace ops::adv::tests::utils;

OpInfo::OpInfo() : OpInfo(ControlInfo(true, false))
{
}

OpInfo::OpInfo(const ControlInfo &ctr) : OpInfo(ctr, ExpectInfo())
{
}

OpInfo::OpInfo(const ControlInfo &ctr, const ExpectInfo &exp) : OpInfo("Undefined", ctr, exp)
{
}

OpInfo::OpInfo(const char *name, const ControlInfo &ctr, const ExpectInfo &exp)
    : name(name), ctr(ctr), exp(exp), ctx(nullptr)
{
}

bool OpInfo::SetContext(ContextIntf *ctxParam)
{
    this->ctx = static_cast<ContextIntf *>(ctxParam);
    return true;
}

bool OpInfo::ProcessTiling(std::string &caseName)
{
    if (ctr.runTiling) {
        if (!this->RunTiling(caseName)) {
            return false;
        }
        if (!this->ChkTiling(caseName)) {
            return false;
        }
    }
    return true;
}

bool OpInfo::ProcessKernel(std::string &caseName)
{
    if (ctr.runKernel) {
        if (!this->RunKernel(caseName)) {
            return false;
        }
        if (!this->ChkKernel(caseName)) {
            return false;
        }
    }
    return true;
}

bool OpInfo::RunTiling(std::string &caseName)
{
    if (!ctx->RunTiling()) {
        LOG_IF(exp.success, LOG_ERR("Case[%s:%s] Run Tiling Failed.", caseName.c_str(), name.c_str()));
        return false;
    }
    return true;
}

bool OpInfo::ChkTiling(std::string &caseName)
{
    if (exp.tilingKey != ExpectInfo::kInvalidTilingKey) {
        auto actTilingKey = ctx->GetTilingKey();
        if (exp.tilingKey != actTilingKey) {
            LOG_IF(exp.success, LOG_ERR("Case[%s:%s] Check Tiling result failed(TilingKey), Exp=%lu, Act=%lu",
                                        caseName.c_str(), name.c_str(), exp.tilingKey, actTilingKey));
            return false;
        }
    }
    if (exp.tilingBlockDim != ExpectInfo::kInvalidTilingBlockDim) {
        int64_t expTilingBlockDim = exp.tilingBlockDim;
        if (expTilingBlockDim == ExpectInfo::kFullTilingBlockDim) {
            auto *platform = Platform::GetGlobalPlatform();
            expTilingBlockDim = platform != nullptr ? platform->GetBlockDim() : expTilingBlockDim;
        }
        auto actTilingBlockDim = ctx->GetTilingBlockDim();
        if (expTilingBlockDim != actTilingBlockDim) {
            LOG_IF(exp.success, LOG_ERR("Case[%s:%s] Check Tiling result failed(TilingBlockDim), Exp=%ld, Act=%ld",
                                        caseName.c_str(), name.c_str(), expTilingBlockDim, actTilingBlockDim));
            return false;
        }
    }
    return true;
}

bool OpInfo::RunKernel(std::string &caseName)
{
    if (!ctx->RunKernel()) {
        LOG_IF(exp.success, LOG_ERR("Case[%s:%s] Run Kernel Failed.", caseName.c_str(), name.c_str()));
        return false;
    }
    return true;
}

bool OpInfo::ChkKernel(std::string &caseName)
{
    return true;
}
