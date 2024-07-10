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
 * \file context.h
 * \brief 提供 CPU模式 Tiling / Kernel 阶段上下文功能, 辅助 Tiling / Kernel 运行.
 */

#pragma once

#include "tests/utils/context_intf.h"
#include <any>

namespace ops::adv::tests::utils {

class Context : public ops::adv::tests::utils::ContextIntf {
public:
    /**
     * Ascend C 框架推荐的 TilingData 最大长度, 超过此值可能会导致算子性能下降.
     * 注意: 此值不可修改.
     */
    static constexpr uint32_t kMaxTilingDataSize = 2048;

    /**
     * Kernel 运行回调函数
     *
     * \param func 算子 kernel 入口函数
     * \param tilingKey TilingKey
     * \param blockDim Tiling 切分 BlockDim
     * \param inputs 算子输入
     * \param outputs 算子输出
     * \param workspace 运行所需的 workspace 空间
     * \param tilingData TilingData 结果
     */
    typedef bool (*KernelRunCbf)(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                                 std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData);

    Context() = default;
    ~Context() override;

    /* 属性设置 */
    [[maybe_unused]] [[nodiscard]] bool SetDeterministic(bool enable);
    [[maybe_unused]] [[nodiscard]] bool SetAttrs(std::vector<std::pair<std::string, std::any>> attrs);
    [[maybe_unused]] [[nodiscard]] bool AddAttrs(std::vector<std::pair<std::string, std::any>> attrs);
    [[maybe_unused]] [[nodiscard]] bool MdfAttrs(const std::pair<std::string, std::any> &attr);
    [[maybe_unused]] [[nodiscard]] bool SetTilingMaxDataSize(uint32_t size = kMaxTilingDataSize);
    [[maybe_unused]] [[nodiscard]] bool SetKernelSoRelPath(const char *relPath);
    [[maybe_unused]] [[nodiscard]] bool SetKernelFuncName(const char *funcName);
    [[maybe_unused]] [[nodiscard]] bool SetKernelRunCbf(KernelRunCbf cbf);

    /* 属性获取 */
    [[maybe_unused]] [[nodiscard]] int32_t GetTilingDataNum() const;
    [[maybe_unused]] [[nodiscard]] void *GetTilingData() const;
    [[maybe_unused]] [[nodiscard]] const std::string &GetTilingDataStr() const;
    [[maybe_unused]] [[nodiscard]] const std::string &GetTilingResult() const;

    /* Tiling */
    [[maybe_unused]] [[nodiscard]] bool RunTiling() override;

protected:
    static constexpr size_t kDefaultTilingResultSize_ = 10 * 1024 * 1024;

    /* 属性 */
    std::vector<std::pair<std::string, std::any>> attrs_;
    uint32_t tilingDataMaxLen_ = kMaxTilingDataSize;
    std::string kernelSoAbsPath_;
    std::string kernelFuncName_;
    void *kernelSoHdl_ = nullptr;
    KernelRunCbf kernelRunCbf_ = nullptr;
    int64_t deterministic_ = 0; /**< 默认不开启确定性计算 */

    /* Tiling */
    std::string inputsJson_;
    std::string outputsJson_;
    std::string attrsJson_;
    std::string extraInfoJson_;
    int32_t tilingDataNum_ = 0;
    bool clearAtomic_ = false;
    std::vector<uint8_t> tilingData_; /**< 向 Kernel 下发的 struct 表示的 TilingData  */
    std::string tilingResult_;        /**< Context 直接返回的 json 表示的 Tiling 结果 */
    std::string tilingDataStr_;

protected:
    bool RunKernelProcess() override;
    uint8_t *AllocWorkspaceImpl(uint64_t size) override;
    void FreeWorkspaceImpl(uint8_t *ptr) override;

    bool InitTilingJsonStr();
    bool ParseTilingResult();

    template <class T> bool DetectField(T &field, const char *fPrefix, const char *fSuffix)
    {
        char *bgn = nullptr;
        char *end = nullptr;
        if (!this->DetectPosit(&bgn, &end, fPrefix, fSuffix)) {
            return false;
        }
        std::string pre(fPrefix);
        std::string sub(bgn + pre.length(), end - bgn - pre.length());
        Context::DetectValue(sub, field);
        return true;
    }

    bool DetectPosit(char **bgn, char **end, const char *fPrefix, const char *fSuffix);
    [[maybe_unused]] static void DetectValue(std::string &sub, int64_t &value);
    [[maybe_unused]] static void DetectValue(std::string &sub, uint64_t &value);
    [[maybe_unused]] static void DetectValue(std::string &sub, bool &value);
    [[maybe_unused]] static void DetectValue(std::string &sub, std::string &value);

private:
    void Destroy();
};

} // namespace ops::adv::tests::utils
