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
 * \file flash_attention_score_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "flash_attention_score_grad_empty_tensor.h"
#include "flash_attention_score_grad_post.h"
#include "flash_attention_score_grad_s1s2_bn2gs1s2.h"
#include "flash_attention_score_grad_pre.h"
#include "flash_attention_score_grad_s1s2_bn2.h"
#include "flash_attention_score_grad_ngs1s2_bn.h"
#include "flash_attention_score_grad_bngs1s2_b.h"

constexpr MatmulConfig MM_CFG_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig MM_CFG_NORMAL = GetNormalConfig(false);
constexpr CubeFormat MM_NZ_OUT_FORMAT = CubeFormat::NZ;
constexpr CubeFormat MM_ND_OUT_FORMAT = CubeFormat::ND_ALIGN;
constexpr CubeFormat MM_ND_OUT_NOALIGN = CubeFormat::ND;
constexpr uint64_t INPUT_NONE = 0;
constexpr uint64_t INPUT_EXIST = 1;
constexpr uint32_t INPUT_DISABLE = 0;
constexpr uint32_t INPUT_ENABLE = 1;

constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;

constexpr static const uint32_t BNGSD = 0;
constexpr static const uint32_t SBNGD = 1;
constexpr static const uint32_t BSNGD = 2;
constexpr static const uint32_t TND = 3;

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, INPUT_LAYOUT, \
                                              MM2_OUT_FORMAT)                                                          \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, tiling_data_in, tiling_data);       \
        const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2 *__restrict tilingData = &tiling_data_in;                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm2tiling = &(tilingData->mm2TilingData);                                       \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm3TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true> opPre;      \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
                                                                                                                       \
        TPipe pipeBase;                                                                                                \
        FlashAttentionScoreGradS1s2Bn2gs1s2<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT,          \
                                            INPUT_LAYOUT, MM2_OUT_FORMAT>                                              \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling, op.mm4,             \
                          bmm3tiling);                                                                                 \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,       \
                prefix, actual_seq_qlen, actual_seq_kvlen, dq, dk, dv, dpse, user, tilingData, &pipeBase);             \
        op.Process();                                                                                                  \
        op.SyncALLCores();                                                                                             \
        pipeBase.Destroy();                                                                                            \
        TPipe pipePost;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2, true, INPUT_LAYOUT,     \
                                    input_format>                                                                      \
            opPost;                                                                                                    \
        opPost.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipePost);                       \
        opPost.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,     \
                                         INPUT_LAYOUT)                                                                 \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataS1s2Bn2, false> opPre;          \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT>                                                      \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm4, bmm4tiling, op.mm3_1,             \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeOp);                                                                                  \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT, ND>      \
            opCast;                                                                                                    \
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeCast);                       \
        opCast.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,       \
                                                    DROPOUT_CFG, INPUT_LAYOUT)                                         \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT>                                                      \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm4, bmm4tiling, op.mm3_1,             \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeIn);                                                                                  \
        op.Process();                                                                                                  \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT, ND>      \
            opCast;                                                                                                    \
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeCast);                       \
        opCast.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(INPUT_TYPE, layout, MM_CONFIG)                                          \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradUbngs1s2BbTilingData, tiling_data_in, tiling_data);         \
        const FlashAttentionScoreGradUbngs1s2BbTilingData *__restrict tilingData = &tiling_data_in;                    \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradUbngs1s2BbTilingData, false> opPre;       \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                 \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm31TilingData);                                      \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm32AndMm4TilingData);                                \
        FlashAttentionScoreGradUngs1s2Bb<INPUT_TYPE, float, MM_CONFIG, layout> op;                                     \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm31, bmm3tiling,                      \
                          op.mm32, bmm4tiling, op.mm4, bmm4tiling);                                                    \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq,   \
                dk, dv, user, tilingData, &pipeOp);                                                                    \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeMuls;                                                                                                \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradUbngs1s2BbTilingData, false, BSNGD, ND> opMuls; \
        opMuls.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeMuls);                       \
        opMuls.Process();                                                                                              \
        pipeMuls.Destroy();                                                                                            \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(INPUT_TYPE, layout, MM_CONFIG)                                            \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataUngs1s2Bbn, tiling_data_in, tiling_data);         \
        const FlashAttentionScoreGradTilingDataUngs1s2Bbn *__restrict tilingData = &tiling_data_in;                    \
        FlashAttentionScoreGradPre<INPUT_TYPE, float, FlashAttentionScoreGradTilingDataUngs1s2Bbn, false> opPre;       \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                                  \
        opPre.Process();                                                                                               \
        opPre.SyncALLCores();                                                                                          \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeOp;                                                                                                  \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                 \
        const TCubeTiling *__restrict bmm3tiling = &(tilingData->mm31TilingData);                                      \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm32AndMm4TilingData);                                \
        FlashAttentionScoreGradUngs1s2Bbn<INPUT_TYPE, float, MM_CONFIG, true, layout> op;                              \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm31, bmm3tiling,                      \
                          op.mm32, bmm4tiling, op.mm4, bmm4tiling);                                                    \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq,   \
                dk, dv, user, tilingData, &pipeOp);                                                                    \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeMuls;                                                                                                \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataUngs1s2Bbn, false, BSNGD, ND> opMuls; \
        opMuls.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeMuls);                       \
        opMuls.Process();                                                                                              \
        pipeMuls.Destroy();                                                                                            \
    } while (0)

// implementation of kernel function
extern "C" __global__ __aicore__ void flash_attention_score_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dy, __gm__ uint8_t *pse_shift,
    __gm__ uint8_t *drop_mask, __gm__ uint8_t *padding_mask, __gm__ uint8_t *atten_mask, __gm__ uint8_t *softmax_max,
    __gm__ uint8_t *softmax_sum, __gm__ uint8_t *softmax_in, __gm__ uint8_t *attention_in, __gm__ uint8_t *prefix,
    __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, __gm__ uint8_t *q_start_idx,
    __gm__ uint8_t *kv_start_idx, __gm__ uint8_t *dq, __gm__ uint8_t *dk,
    __gm__ uint8_t *dv, __gm__ uint8_t *dpse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data) {
    TPipe pipeIn;
    set_mask_norm();
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

// --------------------------------------------float16 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    // -----------------------1.1 start---------------------------------
    if (TILING_KEY_IS(10000000000111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // 1
    } else if (TILING_KEY_IS(10000000000111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // 2
    } else if (TILING_KEY_IS(10000000000111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // 3
    } else if (TILING_KEY_IS(10000000000111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_ND_OUT_NOALIGN);
        return;
        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000000010111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111033434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011033434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101033434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001033434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110033434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010033434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100033434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000033434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, TND,
                                              MM_NZ_OUT_FORMAT);
        return;
        // -----------------------1.1 end---------------------------------

        // -----------------------1.2 start---------------------------------
        // For BSNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000001000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000011000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000101000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000111000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001001000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001011000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010001000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010011000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001101000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001111000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010101000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010111000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011001000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011011000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011101000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011111000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD);
        return;
        // For SBNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000001020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000011020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000101020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000111020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001001020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001011020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010001020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010011020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001101020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001111020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010101020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010111020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011001020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011011020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011101020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011111020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000000000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000010030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000001030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000011030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000110030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000101030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000111030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001010030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001001030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001011030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, TND);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000010010030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000010001030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000010011030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         TND);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001110030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001101030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001111030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, TND);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000010110030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000010101030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000010111030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         TND);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000011010030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000011001030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000011011030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         TND);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000011110030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000011101030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         TND);
        return;
    } else if (TILING_KEY_IS(10000000011111030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         TND);
        return;
        // -----------------------1.2 end---------------------------------

    } else if (TILING_KEY_IS(10000000000000003199UL)) { // BSH BSNGD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(half, LayoutMode::BSNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000013199UL)) { // SBNGD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(half, LayoutMode::SBNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000023199UL)) { // BNGSD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(half, LayoutMode::BNGS1S2, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000001013199UL)) { // SBNGD & FLOAT16_PRECISION & speical MM tilingkey
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(half, LayoutMode::SBNGD, MM_CFG_EXCEED);
        return;
    } else if (TILING_KEY_IS(10000000000000103099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000001113099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::SBNGD, MM_CFG_EXCEED);
        return;
    } else if (TILING_KEY_IS(10000000000000123099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BNGS1S2, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000133099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(half, LayoutMode::BSNGD, MM_CFG_NORMAL);
        return;
    }
#endif
// --------------------------------------------------------------------------------------------------------------------

// --------------------------------------------bfloat16 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_BF16)
    // -----------------------1.1 start---------------------------------
    if (TILING_KEY_IS(10000000000111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // 1
    } else if (TILING_KEY_IS(10000000000111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // 2
    } else if (TILING_KEY_IS(10000000000111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // 3
    } else if (TILING_KEY_IS(10000000000111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;

        // mm2 out nz
    } else if (TILING_KEY_IS(10000000010111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111032434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011032434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101032434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001032434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110032434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010032434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100032434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000032434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
        // -----------------------1.1 end---------------------------------

        // -----------------------1.2 start-------------------------------
        // For BSNGD
    } else if (TILING_KEY_IS(10000000000000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000001002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000011002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000101002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000111002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001001002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001011002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010001002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010011002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001101002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001111002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010101002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010111002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011001002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011011002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011101002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011111002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
        // For SBNGD
    } else if (TILING_KEY_IS(10000000000000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000001022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000011022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000101022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000111022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001001022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001011022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010001022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010011022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001101022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001111022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010101022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010111022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011001022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011011022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011101022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011111022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000000000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000010032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000001032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000011032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000110032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000101032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000111032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001010032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001001032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001011032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010010032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010001032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010011032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001110032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001101032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001111032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010110032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010101032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010111032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011010032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011001032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011011032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011110032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011101032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011111032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
        // -----------------------1.2 end---------------------------------

    } else if (TILING_KEY_IS(10000000000000002199UL)) { // BSH BSNGD & BFLOAT16
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(bfloat16_t, LayoutMode::BSNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000012199UL)) { // SBNGD & BFLOAT16
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(bfloat16_t, LayoutMode::SBNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000022199UL)) { // BNGSD & BFLOAT16
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(bfloat16_t, LayoutMode::BNGS1S2, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000001012199UL)) { // SBNGD & BFLOAT16  & speical MM tilingkey
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(bfloat16_t, LayoutMode::SBNGD, MM_CFG_EXCEED);
        return;
    } else if (TILING_KEY_IS(10000000000000102099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000001112099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::SBNGD, MM_CFG_EXCEED);
        return;
    } else if (TILING_KEY_IS(10000000000000122099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BNGS1S2, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000132099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(bfloat16_t, LayoutMode::BSNGD, MM_CFG_NORMAL);
        return;
    }
#endif
    // --------------------------------------------------------------------------------------------------------------------

// --------------------------------------------float32 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_FLOAT)
    // -----------------------1.1 start---------------------------------
    if (TILING_KEY_IS(10000000000111001434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011001434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101001434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001001434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;

    } else if (TILING_KEY_IS(10000000000110001434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010001434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100001434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000001434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111001434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011001434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101001434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001001434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110001434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010001434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100001434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000001434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // 1
    } else if (TILING_KEY_IS(10000000000111011434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011011434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101011434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001011434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110011434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010011434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100011434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000011434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111011434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011011434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101011434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001011434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110011434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010011434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100011434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000011434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // 2
    } else if (TILING_KEY_IS(10000000000111021434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011021434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101021434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001021434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110021434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010021434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100021434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000021434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111021434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011021434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101021434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001021434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110021434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010021434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100021434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000021434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // 3
    } else if (TILING_KEY_IS(10000000000111031434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011031434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101031434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001031434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110031434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010031434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000100031434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000000031434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111031434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011031434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101031434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001031434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110031434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010031434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001100031434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001000031434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_ND_OUT_NOALIGN);
        return;

        // mm2 out nz
    } else if (TILING_KEY_IS(10000000010111031434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011031434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101031434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001031434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110031434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010031434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100031434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000031434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111031434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011031434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101031434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001031434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110031434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010031434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100031434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000031434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(float, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              TND, MM_NZ_OUT_FORMAT);
        return;
        // -----------------------1.1 end---------------------------------

        // -----------------------1.2 start-------------------------------
        // For BSNGD
    } else if (TILING_KEY_IS(10000000000000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000010001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000001001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000011001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000110001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000101001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000111001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001010001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001001001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001011001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010010001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010001001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010011001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001110001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001101001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000001111001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010110001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010101001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000010111001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011010001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011001001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011011001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011110001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011101001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000011111001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD);
        return;
        // For SBNGD
    } else if (TILING_KEY_IS(10000000000000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000010011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000011011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000110011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000111011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001010011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001011011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010010011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010011011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001110011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000001111011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010110011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000010111011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011010011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011011011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011110011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000011111011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000010021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000001021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000011021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000110021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000101021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000000111021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001010021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001001021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001011021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010010021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010001021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010011021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001110021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001101021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000001111021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010110021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010101021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000010111021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011010021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011001021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011011021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011110021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011101021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
    } else if (TILING_KEY_IS(10000000011111021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000000000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000010031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000001031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000011031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000110031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000101031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000000111031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001010031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001001031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001011031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010010031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010001031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010011031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001110031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001101031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
    } else if (TILING_KEY_IS(10000000001111031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010110031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010101031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000010111031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011010031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011001031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011011031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011110031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011101031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
    } else if (TILING_KEY_IS(10000000011111031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND);
        return;
        // -----------------------1.2 end---------------------------------
    } else if (TILING_KEY_IS(10000000000000001199UL)) { // BSH BSNGD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(float, LayoutMode::BSNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000011199UL)) { // SBNGD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(float, LayoutMode::SBNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000021199UL)) { // BNGSD & FLOAT16_PRECISION
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(float, LayoutMode::BNGS1S2, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000001011199UL)) { // SBNGD & FLOAT16_PRECISION & speical MM tilingkey
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(float, LayoutMode::SBNGD, MM_CFG_EXCEED);
        return;
    } else if (TILING_KEY_IS(10000000000000101099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(float, LayoutMode::BSNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000111099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(float, LayoutMode::SBNGD, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000001111099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(float, LayoutMode::SBNGD, MM_CFG_EXCEED);
        return;
    } else if (TILING_KEY_IS(10000000000000121099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(float, LayoutMode::BNGS1S2, MM_CFG_NORMAL);
        return;
    } else if (TILING_KEY_IS(10000000000000131099UL)) {
        INVOKE_FAG_GENERAL_S1S2_BNGS1S2_B_IMPL(float, LayoutMode::BSNGD, MM_CFG_NORMAL);
        return;
    }
#endif

    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingData, tiling_data_in, tiling_data);
    const FlashAttentionScoreGradTilingData *__restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(90)) {
        FlashAttentionScoreGradEmptyTensor<DTYPE_DQ> op;
        op.Init(dq, dk, dv, dpse, tilingData);
        op.Process();
    }
}
