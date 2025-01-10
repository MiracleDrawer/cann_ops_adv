/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_
#include "prompt_flash_attention_tiling.h"
#include "incre_flash_attention_tiling.h"
#include "register/tilingdata_base.h"

#include "fused_infer_attention_score_tiling_attr_index.h"
#include "fused_infer_attention_score_tiling_compile_info.h"
#include "fused_infer_attention_score_tiling_const.h"
#include "fused_infer_attention_score_tiling_input_index.h"
#include "fused_infer_attention_score_tiling_output_index.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreTilingData)
TILING_DATA_FIELD_DEF(uint32_t, placeHolder);
END_TILING_DATA_DEF;
// Test purposes - using old key
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore, IncreFlashAttentionTilingDataV2)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_13, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_14, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_27, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_30, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000001001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000001001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000200, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000100, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000020, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020200, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020201, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020210, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020211, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020215, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000020216, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021217, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000201, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000210, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000211, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000215, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000216, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000001212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000001217, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000000, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000001, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000010, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000011, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000015, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000016, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000300, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000400, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800000021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000101, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000110, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000111, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000115, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000116, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000002011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000021112, PromptFlashAttentionTilingData)
// PA tilingkey
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010021217, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010001212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000010001217, PromptFlashAttentionTilingData)
// prefix tilingkey
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000101001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100101012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000101001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100001012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100121012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100021012, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000800100021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100021217, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100001212, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000100001217, PromptFlashAttentionTilingData)

// msd tilingkey
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200111112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200011112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300021112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200121112, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200021112, PromptFlashAttentionTilingData)

// msd tilingkey fp16
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200101612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200001612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400300021612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200121612, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000400200021612, PromptFlashAttentionTilingData)
extern "C" {
ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext *context);
ge::graphStatus DeviceDoOpTilingFusedInferAttentionScore(gert::TilingContext *context);
}
ge::graphStatus DoOpTilingFusedInferAttentionScore(gert::TilingContext *context);
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_H_