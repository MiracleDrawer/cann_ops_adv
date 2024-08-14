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
 * \file prompt_flash_attention.cpp
 * \brief
 */
#include "kernel_operator.h"
#if (__CCE_AICORE__ > 200)
#include "prompt_flash_attention_base.h"
#include "prompt_flash_attention_split_n_s_no_tail.h"
#include "prompt_flash_attention_split_n_s_tail.h"
#include "prompt_flash_attention_bnstilling_n_s_no_tail.h"
#include "prompt_flash_attention_bnstilling_n_s_tail.h"
#include "prompt_flash_attention_bnstilling_n_s_no_tailWBNSD.h"
#include "prompt_flash_attention_bnstilling_n_s_tailWBNSD.h"
#include "prompt_flash_attention_s1s2_bns1_x910.h"
#include "prompt_flash_attention_empty_tensor.h"
#else
#include "prompt_flash_attention_s1s2_bns1_x310_base.h"
#include "prompt_flash_attention_s1s2_bns1_x310.h"
#include "prompt_flash_attention_bnstilling_n_s_no_tailWBNSD_KV_NZ.h"
#endif

#define INVOKE_PFA_GENERAL_OP_IMPL(templateClass, ...)                                                                  \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_INT8_OP_IMPL(templateClass, ...)                                                                     \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitQuant(deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2);                                \
        op.Process();                                                                                                   \
    } while (0)
#define INVOKE_PFA_KVANTIQUANT_OP_IMPL(templateClass, ...)                                                              \
    do {                                                                                                                \
        if (query == nullptr) {return;}                                                                                 \
        templateClass<__VA_ARGS__> op;                                                                                  \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                        \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,         \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                    \
        op.InitKvAntiquant(antiquant_scale, antiquant_offset);                                                          \
        op.InitQuant(deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2);                                \
        op.Process();                                                                                                   \
    } while (0)

constexpr uint32_t FLOATBYTENUM = 8;
constexpr uint32_t FLOAT16BYTENUM = 16;
constexpr uint32_t INT8BYTENUM = 32;

extern "C" __global__ __aicore__ void prompt_flash_attention_FIAS(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                             __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquant_scale,
                                                             __gm__ uint8_t* antiquant_offset, __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize,
                                                             __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* keySharedPrefix,
                                                             __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                                             __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                             __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    TPipe tPipe; // not zero
    /*
    拆解TilingData 数据；
    **/

#ifdef __DAV_C220_CUBE__
    // cube核只获取cube tiling数据，减少tiling Data的拷贝
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, bmm1TilingDataRect, bmm1TilingData, tiling);
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, bmm2TilingDataRect, bmm2TilingData, tiling);
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingData;
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingData;
    GET_TILING_DATA_MEMBER(PromptFlashAttentionTilingData, promptAttentionBaseParams, baseParams, tiling);
    auto maskByteNum = baseParams.maskTypeByteNum;
    auto outputTypeByteNum = baseParams.outputTypeByteNum;
    const PromptFlashAttentionTilingData* __restrict tiling_data = nullptr;
#else
    GET_TILING_DATA_WITH_STRUCT(PromptFlashAttentionTilingData, tiling_data_in, tiling);
    const PromptFlashAttentionTilingData* __restrict tiling_data = &tiling_data_in;
    const TCubeTiling* __restrict bmm1tiling = &(tiling_data->bmm1TilingDataRect);
    const TCubeTiling* __restrict bmm2tiling = &(tiling_data->bmm2TilingDataRect);
    auto maskByteNum = tiling_data->promptAttentionBaseParams.maskTypeByteNum;
    auto outputTypeByteNum = tiling_data->promptAttentionBaseParams.outputTypeByteNum;
#endif

    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    
#if (__CCE_AICORE__ > 200)
    #if (ORIG_DTYPE_QUERY == DT_FLOAT16)
        if (TILING_KEY_IS(1000000000000000000)) {
            // split NS no tail
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionSplitNSNoTail, half, half, CubeFormat::ND, half);
            }
            else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionSplitNSNoTail, half, bool, CubeFormat::ND, half);
            }
        } else if (TILING_KEY_IS(1000000000000000001)) {
            // split NS with tail
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionSplitNSTail, half, half, CubeFormat::ND, half);
            }
            else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionSplitNSTail, half, bool, CubeFormat::ND, half);
            }
        } else if (TILING_KEY_IS(1000000000000000010)) {
            // Non-BNSD layout, split NS no tail
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, half, half, CubeFormat::ND, half);
            }
            else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, half, bool, CubeFormat::ND, half);
            }
        } else if (TILING_KEY_IS(1000000000000000011)) {
            // Non-BNSD layout, split NS with tail
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, half, half, CubeFormat::ND, half);
            }
            else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, half, bool, CubeFormat::ND, half);
            }
        } else if (TILING_KEY_IS(1000000000000000015)) {
            // BNSD layout, split NS no tail
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, half, half, CubeFormat::ND,
                                        half);
            }
            else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, half, bool, CubeFormat::ND,
                                        half);
            }
        } else if (TILING_KEY_IS(1000000000000000016)) {
            // BNSD layout, split NS with tail
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, half, half, CubeFormat::ND, half);
            }
            else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, half, bool, CubeFormat::ND, half);
            }
        } else if (TILING_KEY_IS(1000000000000101612)) {
            // BSH layout HighPrecision
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, Mode::HighPrecision>);
        } else if (TILING_KEY_IS(1000000000010101612)) {
            // BSH layout HighPrecision, enable PA
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, Mode::HighPrecision, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100101612)) {
            // Prefix BSH layout HighPrecision
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, Mode::HighPrecision, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000001612)) {
            // BNSD layout HighPrecision
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, Mode::HighPrecision>);
        } else if (TILING_KEY_IS(1000000000010001612)) {
            // BNSD layout HighPrecision, enable PA
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, Mode::HighPrecision, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100001612)) {
            // Prefix BNSD layout HighPrecision
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, Mode::HighPrecision, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000800000101612)) {
            // BSH layout HighPrecision
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, Mode::HighPrecision>);
        } else if (TILING_KEY_IS(1000000800010101612)) {
            // BSH layout HighPrecision, enable PA
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, Mode::HighPrecision, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000800100101612)) {
            // Prefix BSH layout HighPrecision
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, Mode::HighPrecision, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000800000001612)) {
            // BNSD layout HighPrecision
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, Mode::HighPrecision>);
        } else if (TILING_KEY_IS(1000000800010001612)) {
            // BNSD layout HighPrecision, enable PA
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, Mode::HighPrecision, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000800100001612)) {
            // Prefix BNSD layout HighPrecision
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, Mode::HighPrecision, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000001001612)) {
            // BNSD layout HighPrecision
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, Mode::HighPrecision, MatMulType::MM_NORM>);
        } else if (TILING_KEY_IS(1000000000101001612)) {
            // Prefix BNSD layout HighPrecision
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, Mode::HighPrecision, MatMulType::MM_NORM, true>);
        } else if (TILING_KEY_IS(1000000000002001612)) {
            // BNSD layout HighPrecision 
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, Mode::HighPrecision, MatMulType::MM_IBSHARE_NORM>);
        } else if (TILING_KEY_IS(1000000000102001612)) {
            // Prefix BNSD layout HighPrecision 
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, half, Mode::HighPrecision, MatMulType::MM_IBSHARE_NORM, true>);
        } else if (TILING_KEY_IS(1000000000000101012)) {
            // no anti-quant path for CVDIFF-BSH, half in half out
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, half>);
            } else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool>);
            }
        } else if (TILING_KEY_IS(1000000000010101012)) {
            // no anti-quant path for CVDIFF-BSH, half in half out, enable PA
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100101012)) {
            // Prefix no anti-quant path for CVDIFF-BSH, half in half out
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, half, half, half, Mode::HighPerformance, MatMulType::MM_MDL, true>);
            } else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, half, Mode::HighPerformance, MatMulType::MM_MDL, true>);
            }
        } else if (TILING_KEY_IS(1000000800000101012)) {
            // anti-quant path for CVDIFF-BSH, half in half out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t>);
        } else if (TILING_KEY_IS(1000000800010101012)) {
            // anti-quant path for CVDIFF-BSH, half in half out, enable PA
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000800100101012)) {
            // Prefix anti-quant path for CVDIFF-BSH, half in half out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, half, int8_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000001012)) {
            // no anti-quant path for CVDIFF-BNSD, half in half out
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, half>);
            } else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t>);
            }
        } else if (TILING_KEY_IS(1000000000100001012)) {
            // Prefix no anti-quant path for CVDIFF-BNSD, half in half out
            if (maskByteNum == FLOAT16BYTENUM) {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, half, half, half, Mode::HighPerformance, MatMulType::MM_MDL, true>);
            } else {
                INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, Mode::HighPerformance, MatMulType::MM_MDL, true>);
            }
        } else if (TILING_KEY_IS(1000000000001001012)) {
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, Mode::HighPerformance, MatMulType::MM_NORM>);
        } else if (TILING_KEY_IS(1000000000010001012)) {
            // no anti-quant path for CVDIFF-BNSD, half in half out, enable PA
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000101001012)) {  // enable prefix
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, Mode::HighPerformance, MatMulType::MM_NORM, true>);
        } else if (TILING_KEY_IS(1000000000002001012)) {
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, Mode::HighPerformance, MatMulType::MM_IBSHARE_NORM>);
        } else if (TILING_KEY_IS(1000000000102001012)) {  // enable prefix
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, uint8_t, half, half, Mode::HighPerformance, MatMulType::MM_IBSHARE_NORM, true>);
        } else if (TILING_KEY_IS(1000000800000001012)) {
            // anti-quant path for CVDIFF-BNSD, half in half out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t>);
        } else if (TILING_KEY_IS(1000000800010001012)) {
            // anti-quant path for CVDIFF-BNSD, half in half out, enable PA
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000800100001012)) {
            // Prefix anti-quant path for CVDIFF-BNSD, half in half out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, half, int8_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000121012)) {
            // no anti-quant path for CVDIFF-BSH, half in int8 out
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t>);
        } else if (TILING_KEY_IS(1000000000010121012)) {
            // no anti-quant path for CVDIFF-BSH, half in int8 out, enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100121012)) {
            // Prefix no anti-quant path for CVDIFF-BSH, half in int8 out
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000800000121012)) {
            // anti-quant path for CVDIFF-BSH, half in int8 out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t>);
        } else if (TILING_KEY_IS(1000000800010121012)) {
            // anti-quant path for CVDIFF-BSH, half in int8 out, enable PA
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000800100121012)) {
            // Prefix anti-quant path for CVDIFF-BSH, half in int8 out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000021012)) {
            // no anti-quant path for CVDIFF-BNSD, half in int8 out
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t>);
        } else if (TILING_KEY_IS(1000000000010021012)) {
            // no anti-quant path for CVDIFF-BNSD, half in int8 out, enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100021012)) {
            // Prefix no anti-quant path for CVDIFF-BNSD, half in int8 out
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000800000021012)) {
            // anti-quant path for CVDIFF-BNSD, half in int8 out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t>);
        } else if (TILING_KEY_IS(1000000800010021012)) {
            // anti-quant path for CVDIFF-BNSD, half in int8 out, enable PA
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000800100021012)) {
            // Prefix anti-quant path for CVDIFF-BNSD, half in int8 out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000121612)) {
            // no anti-quant path for CVDIFF-BSH, half in int8 out
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, Mode::HighPrecision>);
        } else if (TILING_KEY_IS(1000000000010121612)) {
            // no anti-quant path for CVDIFF-BSH, half in int8 out, enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, Mode::HighPrecision, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100121612)) {
            // Prefix no anti-quant path for CVDIFF-BSH, half in int8 out
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, half, Mode::HighPrecision, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000800000121612)) {
            // anti-quant path for CVDIFF-BSH, half in int8 out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, Mode::HighPrecision>);
        } else if (TILING_KEY_IS(1000000800010121612)) {
            // anti-quant path for CVDIFF-BSH, half in int8 out, enable PA
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, Mode::HighPrecision, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000800100121612)) {
            // Prefix anti-quant path for CVDIFF-BSH, half in int8 out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, half, bool, int8_t, int8_t, Mode::HighPrecision, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000021612)) {
            // no anti-quant path for CVDIFF-BNSD, half in int8 out
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, Mode::HighPrecision>);
        } else if (TILING_KEY_IS(1000000000010021612)) {
            // no anti-quant path for CVDIFF-BNSD, half in int8 out, enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, Mode::HighPrecision, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100021612)) {
            // Prefix no anti-quant path for CVDIFF-BNSD, half in int8 out
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, half, Mode::HighPrecision, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000800000021612)) {
            // anti-quant path for CVDIFF-BNSD, half in int8 out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, Mode::HighPrecision>);
        } else if (TILING_KEY_IS(1000000800010021612)) {
            // anti-quant path for CVDIFF-BNSD, half in int8 out, enable PA
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, Mode::HighPrecision, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000800100021612)) {
            // Prefix anti-quant path for CVDIFF-BNSD, half in int8 out
            INVOKE_PFA_KVANTIQUANT_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, half, bool, int8_t, int8_t, Mode::HighPrecision, MatMulType::MM_MDL, true>);
        }
    #endif
    #if (ORIG_DTYPE_QUERY == DT_BF16)
        if (TILING_KEY_IS(1000000000000000100)) {
            // split NS no tail
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionSplitNSNoTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
        } else if (TILING_KEY_IS(1000000000000000101)) {
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionSplitNSTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
        } else if (TILING_KEY_IS(1000000000000000110)) {
            // Non-BNSD layout, split NS no tail
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
        } else if (TILING_KEY_IS(1000000000000000111)) {
            // Non-BNSD layout, split NS with tail
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
        } else if (TILING_KEY_IS(1000000000000000115)) {
            // BNSD layout, split NS no tail
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
        } else if (TILING_KEY_IS(1000000000000000116)) {
            // BNSD layout, split NS with tail
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, bfloat16_t, bool, CubeFormat::ND, bfloat16_t);
        } else if (TILING_KEY_IS(1000000000000111112)) {
            // BSH layout bf16 cvdiff
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t>);
        } else if (TILING_KEY_IS(1000000000010111112)) {
            // BSH layout bf16 cvdiff, enable PA
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, bfloat16_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100111112)) {  // enable prefix
            // BSH layout bf16 cvdiff
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, bfloat16_t, bfloat16_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000011112)) {
            // BNSD layout bf16 cvdiff
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t>);
        } else if (TILING_KEY_IS(1000000000010011112)) {
            // BNSD layout bf16 cvdiff, enable PA
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, bfloat16_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100011112)) {  // enable prefix
            // BNSD layout bf16 cvdiff
            INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, bfloat16_t, bfloat16_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000121112)) {
            // BSH layout bf16 in int8 out cvdiff
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t>);
        } else if (TILING_KEY_IS(1000000000010121112)) {
            // BSH layout bf16 in int8 out cvdiff, enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, bfloat16_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100121112)) {  // enable prefix
            // BSH layout bf16 in int8 out cvdiff
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, bfloat16_t, bool, int8_t, bfloat16_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000021112)) {
            // BNSD layout bf16 in int8 out cvdiff
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t>);
        } else if (TILING_KEY_IS(1000000000010021112)) {
            // BNSD layout bf16 in int8 out cvdiff, enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, bfloat16_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100021112)) {  // enable prefix
            // BNSD layout bf16 in int8 out cvdiff
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, bfloat16_t, bool, int8_t, bfloat16_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        }
    #endif
    #if (ORIG_DTYPE_QUERY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_INT8)
        if (TILING_KEY_IS(1000000000000020200)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionSplitNSNoTail, int8_t, bool, CubeFormat::ND, int8_t);
        } else if (TILING_KEY_IS(1000000000000020201)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionSplitNSTail, int8_t, bool, CubeFormat::ND, int8_t);
        } else if (TILING_KEY_IS(1000000000000020210)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, int8_t, bool, CubeFormat::ND, int8_t);
        } else if (TILING_KEY_IS(1000000000000020211)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, int8_t, bool, CubeFormat::ND, int8_t);
        } else if (TILING_KEY_IS(1000000000000020215)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, int8_t, bool, CubeFormat::ND, int8_t);
        } else if (TILING_KEY_IS(1000000000000020216)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, int8_t, bool, CubeFormat::ND, int8_t);
        } else if (TILING_KEY_IS(1000000000000021212)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, int8_t>);
        } else if (TILING_KEY_IS(1000000000010021212)) {  // enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, int8_t, int8_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100021212)) {  // enable prefix
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, int8_t, int8_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000000021217)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, int8_t>);
        } else if (TILING_KEY_IS(1000000000010021217)) {  // enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, int8_t, int8_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100021217)) {  // enable prefix
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, int8_t, int8_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        }
    #endif
    #if (ORIG_DTYPE_QUERY == DT_INT8) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
        if (TILING_KEY_IS(1000000000000000200)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionSplitNSNoTail, int8_t, bool, CubeFormat::ND, half);
        } else if (TILING_KEY_IS(1000000000000000201)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionSplitNSTail, int8_t, bool, CubeFormat::ND, half);
        } else if (TILING_KEY_IS(1000000000000000210)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSNoTail, int8_t, bool, CubeFormat::ND, half);
        } else if (TILING_KEY_IS(1000000000000000211)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSTail, int8_t, bool, CubeFormat::ND, half);
        } else if (TILING_KEY_IS(1000000000000000215)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDNoTail, int8_t, bool, CubeFormat::ND, half);
        } else if (TILING_KEY_IS(1000000000000000216)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionBNSTillingNSWithBNSDTail, int8_t, bool, CubeFormat::ND, half);
        } else if (TILING_KEY_IS(1000000000000001212)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, half>);
        } else if (TILING_KEY_IS(1000000000010001212)) {  // enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, half, int8_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000000001217)) {
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, half>);
        } else if (TILING_KEY_IS(1000000000010001217)) {  // enable PA
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, half, int8_t, Mode::HighPerformance, MatMulType::MM_PA>);
        } else if (TILING_KEY_IS(1000000000100001212)) {  // enable prefix
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BSH, int8_t, bool, half, int8_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        } else if (TILING_KEY_IS(1000000000100001217)) {  // enable prefix
            INVOKE_PFA_INT8_OP_IMPL(PromptFlashAttentionS1s2Bns1X910, PFAType<PFALayout::BNSD, int8_t, bool, half, int8_t, Mode::HighPerformance, MatMulType::MM_MDL, true>);
        }
    #endif
    if (TILING_KEY_IS(1000000000000000020)) {
        // kv is empty tensor, return zero output
        PromptFlashAttentionEmptyTensor<half> op;
        op.Init(attentionOut, tiling_data);
        op.Process();
        return;
    }
#else
    if (TILING_KEY_IS(1000000000000012288)){
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half>);
    } else if (TILING_KEY_IS(1000000000000022288)) {
        INVOKE_PFA_GENERAL_OP_IMPL(PromptFlashAttentionS1s2Bns1X310, PFATypeNZ<PFALayoutNZ::BSH, half, int8_t, half>);
    }
#endif
}

extern "C" __global__ __aicore__ void prompt_flash_attention(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                             __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2, __gm__ uint8_t* attentionOut,
                                                             __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    prompt_flash_attention_FIAS(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                                quant_offset2, nullptr, nullptr, nullptr, nullptr, nullptr,nullptr, nullptr, nullptr, attentionOut, nullptr, workspace, tiling);
}