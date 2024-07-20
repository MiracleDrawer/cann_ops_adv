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
 * \file fused_infer_attention_score_tiling_register.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
ge::graphStatus TilingPrepareForFusedInferAttentionScore(gert::TilingParseContext *context) {
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(FusedInferAttentionScore)
    .TilingInputsDataDependency({ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_KV_INDEX, BLOCK_TABLE_INDEX, QUERY_PADDING_SIZE_INDEX,
                                 KV_PADDING_SIZE_INDEX, ACTUAL_SHARED_PREFIX_LEN_INDEX},
                                {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .Tiling(DoOpTilingFusedInferAttentionScore)
    .TilingParse<FusedInferAttentionScoreCompileInfo>(TilingPrepareForFusedInferAttentionScore); // 向框架注册入口函数

} // namespace optiling