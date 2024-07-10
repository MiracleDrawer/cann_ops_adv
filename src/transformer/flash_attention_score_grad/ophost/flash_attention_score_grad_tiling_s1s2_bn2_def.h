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
 * \file flash_attention_score_grad_tiling_s1s2_bn2_def.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradBaseParamsS1s2Bn2)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreProcessNNum);
TILING_DATA_FIELD_DEF(uint32_t, remainCoreProcessNNum);
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, N2);
TILING_DATA_FIELD_DEF(int64_t, S1);
TILING_DATA_FIELD_DEF(int64_t, S2);
TILING_DATA_FIELD_DEF(int64_t, G);
TILING_DATA_FIELD_DEF(int64_t, D);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(int64_t, preTokens);
TILING_DATA_FIELD_DEF(int64_t, nextTokens);
TILING_DATA_FIELD_DEF(uint32_t, isSparse);
TILING_DATA_FIELD_DEF(uint32_t, maskDataType);
TILING_DATA_FIELD_DEF(uint32_t, maskShapeType);
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, pseType);
TILING_DATA_FIELD_DEF(uint32_t, resv1);
TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);
TILING_DATA_FIELD_DEF(uint32_t, bandIdx);
TILING_DATA_FIELD_DEF(uint32_t, existAttenMask);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, inputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, helpBufferLen);
// singleM * singleN * typeSize
TILING_DATA_FIELD_DEF(uint32_t, mm1WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm2WorkspaceLen);
// singleM * singleN * typeSize * 2
TILING_DATA_FIELD_DEF(uint32_t, mm4InputWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm3InputWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(uint32_t, castUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dropoutWorkspaceLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradBaseParamsS1s2Bn2Op, FlashAttentionScoreGradBaseParamsS1s2Bn2)

BEGIN_TILING_DATA_DEF(SplitCoreParamsS1s2Bn2)
TILING_DATA_FIELD_DEF(uint32_t, baseM);
TILING_DATA_FIELD_DEF(uint32_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, singleN);
TILING_DATA_FIELD_DEF(uint32_t, singleM);
TILING_DATA_FIELD_DEF(uint32_t, s1OuterOuter);
TILING_DATA_FIELD_DEF(uint32_t, s2OuterOuter);
TILING_DATA_FIELD_DEF(uint32_t, dInner);
TILING_DATA_FIELD_DEF(uint32_t, SFTBaseM);
TILING_DATA_FIELD_DEF(uint32_t, SFTSingleM);
TILING_DATA_FIELD_DEF(uint32_t, placeholderX2);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitCoreParamsS1s2Bn2Op, SplitCoreParamsS1s2Bn2)

BEGIN_TILING_DATA_DEF(CastParams)
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, inputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, outputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, singleUBProcessNum);
TILING_DATA_FIELD_DEF(uint32_t, dqSingleCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, dkvSingleCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, dqTailCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, kvTailCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, dqSingleCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, kvSingleCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, dqTailCoreLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, kvTailCoreLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, dqLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, kvLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, placeholder);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CastParamsOp, CastParams)

BEGIN_TILING_DATA_DEF(DeterministicTndSplitCoreS1s2Bn2)
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, bN2idxStarts);
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, bN2idxEnds);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DeterministicTndSplitCoreS1s2Bn2Op, DeterministicTndSplitCoreS1s2Bn2)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingDataS1s2Bn2)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradBaseParamsS1s2Bn2, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(SplitCoreParamsS1s2Bn2, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(PreParams, preTilingData);
TILING_DATA_FIELD_DEF_STRUCT(PostParams, postTilingData);
TILING_DATA_FIELD_DEF_STRUCT(DeterministicTndSplitCoreS1s2Bn2, tndSplitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm31TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
// Total Tiling Key: 256
// FLOAT16
// BSND
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011000134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111000134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011000134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111000134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011000134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111000134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011000134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111000134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End BSND


// SBND
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011010134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111010134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011010134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111010134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011010134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111010134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011010134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111010134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End SBND


// BNSD
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011020134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111020134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011020134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111020134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011020134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111020134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011020134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111020134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End BNSD


// TND
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011030134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111030134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011030134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111030134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011030134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111030134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011030134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111030134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End TND
// End FLOAT16

// BFLOAT16
// BSND
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011002134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111002134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011002134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111002134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011002134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111002134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011002134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111002134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End BSND


// SBND
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011012134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111012134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011012134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111012134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011012134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111012134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011012134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111012134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End SBND


// BNSD
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011022134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111022134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011022134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111022134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011022134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111022134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011022134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111022134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End BNSD
// TND
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011032134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111032134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011032134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111032134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011032134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111032134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011032134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111032134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End TND
// End BFLOAT16

// FP32
// BSND
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011001134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111001134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011001134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111001134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011001134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111001134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011001134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111001134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End BSND


// SBND
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011011134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111011134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011011134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111011134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011011134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111011134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011011134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111011134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End SBND


// BNSD
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011021134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111021134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011021134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111021134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011021134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111021134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011021134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111021134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End BNSD


// TND
// drop_out_cfg(0, 1), atten_mask_cfg(0, 1), pse_cfg(0, 1)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000001031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000010031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000011031134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000100031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000101031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000110031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000111031134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001000031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001001031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001010031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001011031134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001100031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001101031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001110031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000001111031134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010000031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010001031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010010031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010011031134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010100031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010101031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010110031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000010111031134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011000031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011001031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011010031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011011031134, FlashAttentionScoreGradTilingDataS1s2Bn2)

// mmOutFormat(ND: 0, NZ: 1), mm_config(NULL: 0, NORMAL: 1, MDL: 2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011100031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011101031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011110031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000011111031134, FlashAttentionScoreGradTilingDataS1s2Bn2)
// End TND
// End FP32
} // namespace optiling