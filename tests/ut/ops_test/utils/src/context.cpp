/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file context.cpp
 * \brief 提供 Tiling / Kernel 阶段上下文功能, 辅助 Tiling / Kernel 运行.
 */

#include "tests/utils/context.h"
#include <utility>
#include <tikicpulib.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"

extern "C" int OpTilingForCompile(const char *opType, const char *compileInfo, const char *compileInfoHash,
                                  const char *inputs, const char *outputs, const char *attrs, char *runInfoJson,
                                  size_t runInfoLen, uint64_t *elapse, const char *extraInfo);

using namespace ops::adv::tests::utils;

Context::~Context()
{
    this->Destroy();
}

bool Context::SetDeterministic(bool enable)
{
    deterministic_ = enable ? 1 : 0;
    return true;
}

bool Context::SetAttrs(std::vector<std::pair<std::string, std::any>> attrs)
{
    attrs_ = std::move(attrs);
    return true;
}

bool Context::AddAttrs(std::vector<std::pair<std::string, std::any>> attrs)
{
    attrs_.insert(attrs_.end(), attrs.begin(), attrs.end());
    return true;
}

bool Context::MdfAttrs(const std::pair<std::string, std::any> &attr)
{
    for (auto &a : attrs_) {
        if (a.first == attr.first) {
            a.second = attr.second;
            return true;
        }
    }
    return false;
}

bool Context::SetTilingMaxDataSize(uint32_t size)
{
    tilingDataMaxLen_ = size;
    return true;
}

bool Context::SetKernelSoRelPath(const char *relPath)
{
    if (relPath == nullptr) {
        LOG_ERR("Kernel so relative path is null,");
        return false;
    }
    Platform *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Can't get global platform.");
        return false;
    }
    kernelSoAbsPath_ = std::string(platform->GetExeAbsPath()) + std::string(relPath);
    return true;
}

bool Context::SetKernelFuncName(const char *funcName)
{
    if (funcName == nullptr) {
        return false;
    }
    kernelFuncName_ = std::string(funcName);
    return true;
}

bool Context::SetKernelRunCbf(KernelRunCbf cbf)
{
    if (cbf == nullptr) {
        return false;
    }
    kernelRunCbf_ = cbf;
    return true;
}

int32_t Context::GetTilingDataNum() const
{
    return tilingDataNum_;
}

void *Context::GetTilingData() const
{
    return (void *)tilingData_.data();
}

const std::string &Context::GetTilingDataStr() const
{
    return tilingDataStr_;
}

const std::string &Context::GetTilingResult() const
{
    return tilingResult_;
}

bool Context::RunTiling()
{
    if (!this->InitTilingJsonStr()) {
        return false;
    }
    /* 调用 Tiling 接口, TbeOpTilingPyInterface 内包含 context 构造 */
    tilingResult_.resize(kDefaultTilingResultSize_, 0);
    tilingDataNum_ = OpTilingForCompile(opName_.c_str(), "{\"device_id\": null}", "", inputsJson_.c_str(),
                                        outputsJson_.c_str(), attrsJson_.c_str(), tilingResult_.data(),
                                        tilingResult_.size(), nullptr, extraInfoJson_.c_str());
    if (tilingDataNum_ != 1) {
        LOG_DBG("%s TilingDataNum = %d != 1", opName_.c_str(), tilingDataNum_);
        return false;
    }
    /* Tiling 结果解析 */
    return this->ParseTilingResult();
}

bool Context::RunKernelProcess()
{
    if (kernelRunCbf_ == nullptr) {
        return false;
    }
    kernelSoHdl_ = Platform::LoadSo(kernelSoAbsPath_.c_str());
    void *func = Platform::LoadSoSym(kernelSoHdl_, kernelFuncName_.c_str());
    if (func == nullptr) {
        LOG_ERR("Can't get KernelFunc[%s] from KernelSo[%p: %s]", kernelFuncName_.c_str(), kernelSoHdl_,
                kernelSoAbsPath_.c_str());
        return false;
    }
    ICPU_SET_TILING_KEY(tilingKey_);
    LOG_DBG("[BGN] Run %s Kernel(%s) async, TilingKey=%lu, BlockDim=%ld", opName_.c_str(), kernelFuncName_.c_str(),
            tilingKey_, tilingBlockDim_);
    return kernelRunCbf_(func, tilingKey_, tilingBlockDim_, inputs_, outputs_, workspacePtr_, tilingData_.data());
}

uint8_t *Context::AllocWorkspaceImpl(uint64_t size)
{
    auto *ptr = (uint8_t *)AscendC::GmAlloc(size);
    LOG_IF(ptr == nullptr, LOG_ERR("AscendC::GmAlloc failed, Size(%ld)", size));
    return ptr;
}

void Context::FreeWorkspaceImpl(uint8_t *ptr)
{
    AscendC::GmFree(ptr);
}

bool Context::InitTilingJsonStr()
{
    /* 构造 Input 所需 json */
    inputsJson_ = "[";
    for (auto &i : inputs_) {
        inputsJson_ += "\n " + i->GetTilingStr() + ",";
    }
    if (!inputs_.empty()) {
        inputsJson_.resize(inputsJson_.size() - 1);
    }
    inputsJson_ += "\n]";

    /* 构造 Output 所需 json */
    outputsJson_ = "[";
    for (auto &o : outputs_) {
        outputsJson_ += "\n " + o->GetTilingStr() + ",";
    }
    if (!outputs_.empty()) {
        outputsJson_.resize(outputsJson_.size() - 1);
    }
    outputsJson_ += "\n]";

    /* 构造 Attrs 所需 json */
    attrsJson_ = "[";
    for (auto &a : attrs_) {
        attrsJson_ += "\n { \"name\": \"" + a.first + "\", ";
        attrsJson_ += R"("dtype": ")";
        auto value = a.second;
        if (value.type() == typeid(float)) {
            auto fv = std::any_cast<float>(value);
            attrsJson_ += R"(float", "value": )" + std::to_string(fv) + " },";
        } else if (value.type() == typeid(int64_t)) {
            auto iv = std::any_cast<int64_t>(value);
            attrsJson_ += R"(int", "value": )" + std::to_string(iv) + " },";
        } else if (value.type() == typeid(int32_t)) {
            auto iv = std::any_cast<int32_t>(value);
            attrsJson_ += R"(int", "value": )" + std::to_string(iv) + " },";
        } else if (value.type() == typeid(std::string)) {
            auto sv = std::any_cast<std::string>(value);
            attrsJson_ += R"(str", "value": ")" + std::string(sv) + R"(" },)";
        } else {
            LOG_ERR("Unknown Attrs(%s)'s dtype(%s).", a.first.c_str(), value.type().name());
            return false;
        }
    }
    if (!attrs_.empty()) {
        attrsJson_.resize(attrsJson_.size() - 1);
    }
    attrsJson_ += "\n]";

    extraInfoJson_ = "{\"deterministic\": " + std::to_string(deterministic_) + "}";

    LOG_DBG("%s Inputs: %s", opName_.c_str(), inputsJson_.c_str());
    LOG_DBG("%s Outputs: %s", opName_.c_str(), outputsJson_.c_str());
    LOG_DBG("%s Attrs: %s", opName_.c_str(), attrsJson_.c_str());
    LOG_DBG("%s ExtraInfo: %s", opName_.c_str(), extraInfoJson_.c_str());
    return true;
}

bool Context::ParseTilingResult()
{
    tilingResult_.resize(tilingResult_.length());

    if (!this->DetectField(tilingBlockDim_, R"("block_dim":)", ",")) {
        LOG_ERR("%s tiling parse failed, can't detect [block_dim] from %s", opName_.c_str(), tilingResult_.data());
        return false;
    }
    if (!this->DetectField(clearAtomic_, R"("clear_atomic":)", ",")) {
        LOG_ERR("%s tiling parse failed, can't detect [clear_atomic] from %s", opName_.c_str(), tilingResult_.data());
        return false;
    }
    if (!this->DetectField(tilingKey_, R"("tiling_key":)", ",")) {
        LOG_ERR("%s tiling parse failed, can't detect [tiling_key] from %s", opName_.c_str(), tilingResult_.data());
        return false;
    }
    if (!this->DetectField(tilingDataStr_, R"("tiling_data":")", R"(",)")) {
        LOG_ERR("%s tiling parse failed, can't detect [tiling_data] from %s", opName_.c_str(), tilingResult_.data());
        return false;
    }
    if (!this->DetectField(workspaceSize_, R"("workspaces":[)", "]")) {
        LOG_ERR("%s tiling parse failed, can't detect [workspaces] from %s", opName_.c_str(), tilingResult_.data());
        return false;
    }

    uint32_t tilingDataStrLen = tilingDataStr_.length();
    uint32_t tilingDataLen = tilingDataStrLen / 2;
    if (tilingDataStrLen % 8 != 0 || tilingDataLen > tilingDataMaxLen_) {
        LOG_ERR("%s(TilingKey=%lu) TilingDataStrLen(%u) %% 8 = %u != 0 || TilingDataLen(%u) > %u", opName_.c_str(),
                tilingKey_, tilingDataStrLen, tilingDataStrLen % 8, tilingDataLen, tilingDataMaxLen_);
        return false;
    }
    tilingData_.resize(tilingDataLen, 0);
    for (uint32_t i = 0; i < tilingDataStrLen; i += 2) {
        if (sscanf_s(tilingDataStr_.c_str() + i, "%2hhx", &tilingData_[i / 2]) != 1) {
            LOG_ERR("%s tiling data parse failed, idx = %u", opName_.c_str(), i);
            return false;
        }
    }
    LOG_DBG("%s tiling success, TilingKey=%lu, TilingBlockDim=%ld, TilingWorkspaceSize=%zu, TilingDataSize=%u",
            opName_.c_str(), tilingKey_, tilingBlockDim_, workspaceSize_, tilingDataLen);
    return true;
}

bool Context::DetectPosit(char **bgn, char **end, const char *fPrefix, const char *fSuffix)
{
    *bgn = nullptr;
    *end = nullptr;
    *bgn = strstr(tilingResult_.data(), fPrefix);
    if (*bgn == nullptr) {
        return false;
    }
    *end = strstr(*bgn, fSuffix);
    if (*end == nullptr) {
        return false;
    }
    return true;
}

void Context::DetectValue(std::string &sub, int64_t &value)
{
    value = std::stoll(sub);
}

void Context::DetectValue(std::string &sub, uint64_t &value)
{
    value = std::stoul(sub);
}

void Context::DetectValue(std::string &sub, bool &value)
{
    value = sub == "true";
}

void Context::DetectValue(std::string &sub, std::string &value)
{
    value = sub;
}

void Context::Destroy()
{
    if (workspacePtr_ != nullptr) {
        this->FreeWorkspaceImpl(workspacePtr_);
        workspacePtr_ = nullptr;
    }
    workspaceSize_ = 0;

    if (kernelSoHdl_ != nullptr) {
        (void)Platform::UnLoadSo(kernelSoHdl_);
    }
}
