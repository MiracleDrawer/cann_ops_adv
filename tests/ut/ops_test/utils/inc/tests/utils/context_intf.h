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
 * \file context_intf.h
 * \brief 提供 Tiling / Kernel 阶段上下文功能基类, 辅助 Tiling / Kernel 运行.
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "tests/utils/tensor_intf.h"

namespace ops::adv::tests::utils {

class ContextIntf {
public:
    static constexpr uint64_t kMaxWorkspaceSize = 1024 * 1024 * 1024;

    /**
     * Kernel 运行前设置 TensorData 回调函数
     */
    typedef bool (*KernelRunPrepareTensorDataCbf)(void *curCase);

public:
    ContextIntf() = default;
    virtual ~ContextIntf() = default;

    /* 属性设置 */
    [[maybe_unused]] [[nodiscard]] bool SetOpName(const char *name);
    [[maybe_unused]] [[nodiscard]] bool SetInputs(const std::vector<TensorIntf *> &inputs);
    [[maybe_unused]] [[nodiscard]] bool SetOutputs(const std::vector<TensorIntf *> &outputs);
    [[maybe_unused]] [[nodiscard]] bool SetKernelRunPrepareTensorDataCbf(KernelRunPrepareTensorDataCbf cbf);

    /* 属性获取 */
    [[maybe_unused]] [[nodiscard]] int64_t GetTilingBlockDim() const;
    [[maybe_unused]] [[nodiscard]] uint64_t GetTilingKey() const;
    [[maybe_unused]] [[nodiscard]] uint8_t *GetWorkspacePtr() const;
    [[maybe_unused]] [[nodiscard]] uint64_t GetWorkspaceSize() const;

    /* Tiling */
    virtual bool RunTiling() = 0;

    /* Kernel */
    virtual bool RunKernel();

    /* Others */
    bool SaveOutputsToDir(const std::string &dir, const std::string &filePrefix);

protected:
    /* 属性 */
    std::string opName_;
    std::vector<TensorIntf *> inputs_;
    std::vector<TensorIntf *> outputs_;
    KernelRunPrepareTensorDataCbf kernelRunPrepareTensorDataCbf_ = nullptr;

    /* Tiling & Kernel */
    int64_t tilingBlockDim_ = 0;
    uint64_t tilingKey_ = std::numeric_limits<uint64_t>::max();
    uint64_t workspaceSize_ = 0;
    uint8_t *workspacePtr_ = nullptr;

protected:
    virtual bool RunKernelPrepare();
    virtual bool RunKernelProcess() = 0;

    virtual uint8_t *AllocWorkspaceImpl(uint64_t size) = 0;
    virtual void FreeWorkspaceImpl(uint8_t *ptr) = 0;
};

} // namespace ops::adv::tests::utils
