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
                                         INPUT_LAYOUT, MM2_OUT_FORMAT)                                                 \
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
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT>                                      \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm4, bmm4tiling, op.mm3_1,             \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeOp);                                                                                  \
        op.Process();                                                                                                  \
        pipeOp.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT,          \
        input_format> opCast;                                                                                          \
        opCast.Init(dq, dk, dv, actual_seq_qlen, actual_seq_kvlen, user, tilingData, &pipeCast);                       \
        opCast.Process();                                                                                              \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(INPUT_TYPE, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,       \
                                                    DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT)                         \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataS1s2Bn2, tiling_data_in, tiling_data);            \
        const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData = &tiling_data_in;                       \
        const TCubeTiling *__restrict bmm1tiling = &(tilingData->mm1TilingData);                                       \
        const TCubeTiling *__restrict bmm31tiling = &(tilingData->mm31TilingData);                                     \
        const TCubeTiling *__restrict bmm4tiling = &(tilingData->mm4TilingData);                                       \
        FlashAttentionScoreGradS1s2Bn2<INPUT_TYPE, float, MM_CONFIG, CUBE_FORMAT, PSE_CFG, ATTEN_MASK_CFG,             \
                                       DROPOUT_CFG, INPUT_LAYOUT, MM2_OUT_FORMAT>                                      \
            op;                                                                                                        \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm4, bmm4tiling, op.mm3_1,             \
                          bmm31tiling);                                                                                \
        op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,       \
                prefix, softmax_in, actual_seq_qlen, actual_seq_kvlen, attention_in, dq, dk, dv, dpse, user,           \
                tilingData, &pipeIn);                                                                                  \
        op.Process();                                                                                                  \
        pipeIn.Destroy();                                                                                              \
        TPipe pipeCast;                                                                                                \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND;                       \
        FlashAttentionScoreGradPost<INPUT_TYPE, FlashAttentionScoreGradTilingDataS1s2Bn2, true, INPUT_LAYOUT,          \
        input_format> opCast;                                                                                          \
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

        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000000010111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111003434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011003434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101003434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001003434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110003434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010003434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100003434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000003434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BSNGD,
                                              MM_NZ_OUT_FORMAT);
        return;

        // 1 格式为SBNGD
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

        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000000010111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111013434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011013434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101013434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001013434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110013434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010013434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100013434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000013434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, SBNGD,
                                              MM_NZ_OUT_FORMAT);
        return;

        // 2 格式为BNGSD
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
        // --- mm2 out Nz
    } else if (TILING_KEY_IS(10000000010111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111023434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011023434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101023434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001023434UL)) {
        // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110023434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010023434UL)) {
        // attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100023434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000023434UL)) {
        // attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT, BNGSD,
                                              MM_NZ_OUT_FORMAT);
        return;

        // 3 格式为TND
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
        // -----------------------1.1 end---------------------------------

         // -----------------------1.2 start---------------------------------
        // For BSNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // For SBNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000000000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         TND, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         TND, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         TND, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         TND, MM_ND_OUT_NOALIGN);
        return;
        // For mm345 out
        // For BSNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000100000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111010000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111110000134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // For SBNGD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000100000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111010010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111001010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111011010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111110010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111101010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111111010134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000100000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111010020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111110020134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000100000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE, INPUT_EXIST,
                                         TND, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE, INPUT_EXIST,
                                         TND, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST, INPUT_EXIST,
                                         TND, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100030134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST, INPUT_EXIST,
                                         TND, MM_NZ_OUT_FORMAT);
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

        // mm2 out nz
    } else if (TILING_KEY_IS(10000000010111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111002434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011002434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101002434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001002434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110002434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010002434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100002434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000002434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // 1 for layout SBNGD
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

        // mm2 out nz
    } else if (TILING_KEY_IS(10000000010111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111012434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011012434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101012434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001012434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110012434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010012434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100012434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000012434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // 2 for layout BNGSD
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
        // mm2 out nz
    } else if (TILING_KEY_IS(10000000010111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                              MM_ND_OUT_NOALIGN, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111022434UL)) {
        // attention_mask:1, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011022434UL)) {
        // attention_mask:0, pse:1, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101022434UL)) {
        // attention_mask:1, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001022434UL)) {
        // // attention_mask:0, pse:0, drop:1, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_ENABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110022434UL)) {
        // attention_mask:1, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010022434UL)) {
        //  attention_mask:0, pse:1, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_ENABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011100022434UL)) {
        // attention_mask:1, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nz
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_NZ_OUT_FORMAT,
                                              BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // 3 for TND layout
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
        // -----------------------1.1 end---------------------------------

        // -----------------------1.2 start-------------------------------
        // For BSNGD
    } else if (TILING_KEY_IS(10000000000000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_FORMAT);
        return;
        // For SBNGD
    } else if (TILING_KEY_IS(10000000000000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_FORMAT);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000000110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000001110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000010110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000011110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_FORMAT);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000000000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_ND_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND, MM_ND_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_FORMAT);
        return;
       // for mm345 NZ out
        // -----------------------1.2 start-------------------------------
        // For BSNGD
    } else if (TILING_KEY_IS(10000000100000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111010002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111110002134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_NZ_OUT_FORMAT);
        return;
        // For SBNGD
    } else if (TILING_KEY_IS(10000000100000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111010012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111001012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111011012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111110012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111101012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111111012134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_EXCEED, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_NZ_OUT_FORMAT);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000100000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000100110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000101110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000110110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111010022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(10000000111110022134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_NZ_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_NZ_OUT_FORMAT);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000100000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000100100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000101000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000110000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000101100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_NZ_OUT_FORMAT);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000110100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_NZ_OUT_FORMAT);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000111000032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_NZ_OUT_FORMAT);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000111100032134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_NZ_OUT_FORMAT);
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
        // -----------------------1.1 end---------------------------------

        // -----------------------1.2 start-------------------------------
        // For BSNGD
    } else if (TILING_KEY_IS(10000000000000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010001001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010101001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011001001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011101001134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BSNGD, MM_ND_OUT_NOALIGN);
        return;
        // For SBNGD
    } else if (TILING_KEY_IS(10000000000000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011001011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011101011134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, SBNGD, MM_ND_OUT_NOALIGN);
        return;
        // For BNGSD
        // pse atten_mask dropout 均不存在
    } else if (TILING_KEY_IS(10000000000000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010001021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010101021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011001021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011101021134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, BNGSD, MM_ND_OUT_NOALIGN);
        return;
        // for TND
    } else if (TILING_KEY_IS(10000000000000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000001031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                                    INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse单独存在
    } else if (TILING_KEY_IS(10000000000100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000000101031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_NONE, INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask单独存在
    } else if (TILING_KEY_IS(10000000001000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001001031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // dropout单独存在
    } else if (TILING_KEY_IS(10000000010000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010001031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse atten_mask存在
    } else if (TILING_KEY_IS(10000000001100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000001101031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_NO_DROPOUT_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST,
                                                    INPUT_EXIST, INPUT_NONE, TND, MM_ND_OUT_NOALIGN);
        return;
        // pse dropout存在
    } else if (TILING_KEY_IS(10000000010100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000010101031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_NONE,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
        // atten_mask dropout存在
    } else if (TILING_KEY_IS(10000000011000031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011001031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_NONE, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
        // 均存在
    } else if (TILING_KEY_IS(10000000011100031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_NORMAL, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
        return;
    } else if (TILING_KEY_IS(10000000011101031134UL)) {
        INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(float, MM_CFG_EXCEED, MM_ND_OUT_FORMAT, INPUT_EXIST, INPUT_EXIST,
                                         INPUT_EXIST, TND, MM_ND_OUT_NOALIGN);
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
