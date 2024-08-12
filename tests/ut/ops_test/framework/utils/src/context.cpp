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
#include <iostream>
#include <fstream>
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

bool Context::SetTilingDataMaxSize(uint32_t size)
{
    tilingDataMaxLen_ = size;
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

bool Context::SetKernelMainFunc(void *funcName)
{
    kernelMainFunc_ = funcName;
    return true;
}

int32_t Context::GetTilingDataNum() const
{
    return tilingDataNum_;
}

const void *Context::GetTilingData() const
{
    return (const void *)tilingData_.data();
}

const std::string &Context::GetTilingDataStr() const
{
    return tilingDataStr_;
}

const std::string &Context::GetTilingResult() const
{
    return tilingResult_;
}

bool Context::RunTiling(std::string &caseName)
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
        LOG_DBG("[%s:%s] TilingDataNum = %d != 1", opName_.c_str(), caseName.c_str(), tilingDataNum_);
        return false;
    }
    /* Tiling 结果解析 */
    return this->ParseTilingResult();
}

bool Context::RunKernelProcess(std::string &caseName)
{
    if (kernelRunCbf_ == nullptr) {
        LOG_ERR("[%s:%s] Can't get kernelRunCbf_", opName_.c_str(), caseName.c_str());
        return false;
    }
    if (kernelMainFunc_ == nullptr) {
        LOG_ERR("[%s:%s] Can't get KernelMainFunc", opName_.c_str(), caseName.c_str());
        return false;
    }
    LOG_DBG("[BGN] Run %s:%s Kernel async, TilingKey=%lu, BlockDim=%ld", opName_.c_str(), caseName.c_str(), tilingKey_,
            tilingBlockDim_);

    /* 重定向 std err/out/log 到同一个文件 */
    std::string filePath = std::string(platform_->GetExeAbsPath()) + "/" + opName_ + "_" + caseName + "_kernel.log";
    std::ofstream oFileHdl(filePath);
    std::streambuf *stdErr = std::cerr.rdbuf(oFileHdl.rdbuf());
    std::streambuf *stdLog = std::clog.rdbuf(oFileHdl.rdbuf());
    std::streambuf *stdOut = std::cout.rdbuf(oFileHdl.rdbuf());

    /* 调用回调函数, 触发具体算子 Kernel 执行 */
    ICPU_SET_TILING_KEY(tilingKey_);
    auto ret = kernelRunCbf_(kernelMainFunc_, tilingKey_, tilingBlockDim_, inputs_, outputs_, workspacePtr_,
                             tilingData_.data());

    /* 恢复 std err/out/log */
    std::cout.rdbuf(stdOut);
    std::clog.rdbuf(stdLog);
    std::cerr.rdbuf(stdErr);
    oFileHdl.close();

    /* Kernel 执行日志结果获取 */
    std::ifstream iFile;
    iFile.open(filePath, std::ios::in);
    if (!iFile.is_open()) {
        LOG_ERR("[%s:%s] Can't open KernelRstLogFile(%s).", opName_.c_str(), caseName.c_str(), filePath.c_str());
        return false;
    }
    std::stringstream iFileStrStream;
    iFileStrStream << iFile.rdbuf();
    std::string iFileStr(iFileStrStream.str());
    iFile.close();

    /* Kernel 执行日志结果校验 */
    ret = ret && CheckKernelResultStr(iFileStr);

    if (std::remove(filePath.c_str()) != 0) {
        LOG_ERR("[%s:%s] Can't remove KernelRstLogFile(%s)", opName_.c_str(), caseName.c_str(), filePath.c_str());
    }
    if (!ret) {
        LOG_ERR("[%s:%s] Run kernel failed, details:\n%s\n", opName_.c_str(), caseName.c_str(), iFileStr.c_str());
    } else {
        fprintf(stdout, "Run kernel finish, details:\n%s", iFileStr.c_str());
    }
    return ret;
}

uint8_t *Context::AllocWorkspaceImpl(uint64_t size)
{
    auto *ptr = (uint8_t *)AscendC::GmAlloc(size);
    LOG_IF(ptr == nullptr, LOG_ERR("AscendC::GmAlloc failed, Size(%lu)", size));
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
        } else if (value.type() == typeid(bool)) {
            auto bv = std::any_cast<bool>(value);
            attrsJson_ += R"(bool", "value": )" + std::string(bv ? "true" : "false") + " },";
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

    uint32_t hexByteSize = 2; // 16进制时使用2个字符表时一个Byte
    uint32_t tilingDataStrLen = tilingDataStr_.length();
    uint32_t tilingDataLen = tilingDataStrLen / hexByteSize;
    uint32_t tilingDataAlignSize = 8; // 要求 TilingData 8 字节对齐
    uint32_t tilingDataRemainder = tilingDataStrLen % tilingDataAlignSize;
    if (tilingDataRemainder != 0 || tilingDataLen > tilingDataMaxLen_) {
        LOG_ERR("%s(TilingKey=%lu) TilingDataStrLen(%u) %% %u = %u != 0 || TilingDataLen(%u) > %u", opName_.c_str(),
                tilingKey_, tilingDataStrLen, tilingDataAlignSize, tilingDataRemainder, tilingDataLen,
                tilingDataMaxLen_);
        return false;
    }
    tilingData_.resize(tilingDataLen, 0);
    for (uint32_t i = 0; i < tilingDataStrLen; i += hexByteSize) {
        if (sscanf_s(tilingDataStr_.c_str() + i, "%2hhx", &tilingData_[i / hexByteSize]) != 1) {
            LOG_ERR("%s tiling data parse failed, idx = %u", opName_.c_str(), i);
            return false;
        }
    }
    LOG_DBG("%s tiling success, TilingKey=%lu, TilingBlockDim=%ld, TilingWorkspaceSize=%zu, TilingDataSize=%u",
            opName_.c_str(), tilingKey_, tilingBlockDim_, workspaceSize_, tilingDataLen);
    return true;
}

bool Context::CheckKernelResultStr(std::string &kernelLog)
{
    /* 不存在 Ascend C 框架感知的错误 */
    auto rst = kernelLog.find("error happened! =========") == std::string::npos;
    /* 不存在 AddressSanitizer 感知的错误 */
    rst = rst && kernelLog.find("AddressSanitizer") == std::string::npos;
    /* 必需存在 Ascend C 框架感知的正常退出消息 */
    rst = rst && kernelLog.find("exit success!") != std::string::npos;
    return rst;
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
}
