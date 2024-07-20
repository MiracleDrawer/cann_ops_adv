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
 * \file prompt_flash_attention_empty_tensor.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_EMPTY_TENSOR_H
#define PROMPT_FLASH_ATTENTION_EMPTY_TENSOR_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_data_copy_transpose.h"

template<typename T>
class PromptFlashAttentionEmptyTensor {
public:
    __aicore__ inline PromptFlashAttentionEmptyTensor() {};
    __aicore__ inline void Init(__gm__ uint8_t*  attentionOut,
                                const PromptFlashAttentionTilingData* __restrict tiling);
    __aicore__ inline void Process();

protected:
    const PromptFlashAttentionTilingData* __restrict tilingData;
    GlobalTensor<T> attentionOutGm;
};

template<typename T>
__aicore__ inline void PromptFlashAttentionEmptyTensor<T>::Init(__gm__ uint8_t*  attentionOut,
                                                               const PromptFlashAttentionTilingData* __restrict tiling) {
    attentionOutGm.SetGlobalBuffer((__gm__ T*)attentionOut);
    tilingData = tiling;
}

template<typename T>
__aicore__ inline void PromptFlashAttentionEmptyTensor<T>::Process() {
        uint32_t tmp_block_idx = GetBlockIdx();
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - tmp_block_idx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<T>(attentionOutGm[tmp_block_idx * initParams.singleCoreSize], singleInitOutputSize, 0);
}
#endif  // PROMPT_FLASH_ATTENTION_EMPTY_TENSOR_H