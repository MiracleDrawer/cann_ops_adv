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
 * \file prompt_flash_attention_s1s2_bns1_x910_base.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_S1S2_BNS1_X910_BASE_H
#define PROMPT_FLASH_ATTENTION_S1S2_BNS1_X910_BASE_H
#include <type_traits>
#include "prompt_flash_attention_base.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "kernel_data_copy_transpose.h"

using namespace matmul;

enum class PFALayout {
    BSH = 0,
    BNSD,
};

enum class MatMulType {
    MM_MDL = 0,
    MM_NORM,
    MM_IBSHARE_NORM,
    MM_PA,
};

template <const MatMulType MM_TYPE>
struct GetMatmulConfig {
    static constexpr MatmulConfig mmcfg_value = CFG_MDL;
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_MDL> {
    static constexpr MatmulConfig mmcfg_value = CFG_MDL;
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_NORM> {
    static constexpr MatmulConfig mmcfg_value = CFG_NORM;
    static constexpr bool ibshare_value = false;
};

template <>
struct GetMatmulConfig<MatMulType::MM_IBSHARE_NORM> {
    static constexpr MatmulConfig mmcfg_value = CFG_IBSHARE_NORM;
    static constexpr bool ibshare_value = true;
};

template <>
struct GetMatmulConfig<MatMulType::MM_PA> {
    static constexpr MatmulConfig mmcfg_value = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, false);
    static constexpr bool ibshare_value = false;
};

template <PFALayout L, typename T, typename U, typename O = T, typename KV_T = T, Mode M = Mode::HighPerformance, const MatMulType MM_TYPE_TMP = MatMulType::MM_MDL, const bool F = false, typename...Args>
struct PFAType {
    using inputType = T;
    using maskType = U;
    using outputType = O;
    using kvInputType = KV_T;
    static constexpr PFALayout layout = L;
    static constexpr Mode calcMode = M;
    static constexpr MatMulType MM_TYPE = MM_TYPE_TMP;
    static constexpr MatmulConfig mmCFG = GetMatmulConfig<MM_TYPE>::mmcfg_value;
    static constexpr bool ibShare = GetMatmulConfig<MM_TYPE>::ibshare_value;
    static constexpr bool enablePrefix = F;
};

constexpr static uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xC77FE000;

#define TEMPLATE_LAYOUT template<PFALayout layout = PFAT::layout>
#define TYPENAME_BSH_VOID typename std::enable_if<layout == PFALayout::BSH, void>::type
#define TYPENAME_BNSD_VOID typename std::enable_if<layout == PFALayout::BNSD, void>::type
#define TYPENAME_BSH_INT64 typename std::enable_if<layout == PFALayout::BSH, int64_t>::type
#define TYPENAME_BNSD_INT64 typename std::enable_if<layout == PFALayout::BNSD, int64_t>::type

#define TEMPLATE_MASKTYPE template<typename _maskType>
#define TYPENAME_MASKTYPE_BOOL_VOID typename std::enable_if<std::is_same_v<_maskType, bool>, void>::type
#define TYPENAME_MASKTYPE_INT8_VOID typename std::enable_if<std::is_same_v<_maskType, uint8_t>, void>::type
#define TYPENAME_MASKTYPE_HALF_VOID typename std::enable_if<std::is_same_v<_maskType, half>, void>::type

struct PFAComputeParam {
    bool isFirstInnerIter;
    bool isSecondInnerIter;
    bool isLastInnerIter;
    bool isInnerTail;
    bool useMask;
    bool usePseShift;
    bool kernelInvalidRow;
    bool isPrefixInnerIter;

    uint32_t singleProcessSOuterSize;
    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSInnerSizeTail;
    uint32_t singleProcessSInnerPrefixSizeTail;
    uint32_t singleProcessSInnerSizeNow;
    uint32_t singleProcessSInnerBmmTail;
    uint32_t padSize;
    uint32_t padPrefixSize;
    uint32_t pseShiftPadSize;
    uint32_t pseShiftPadPrefixSize;
    uint32_t unalignSInner;
    uint32_t unalignSInnerPrefix;
    uint32_t maskCopyInCol;
    uint32_t pseShiftCopyInCol;
    uint32_t maskInnerTailAlign;
    uint32_t maskInnerPrefixTailAlign;
    uint32_t pseShiftInnerTailAlign;
    uint32_t pseShiftInnerPrefixTailAlign;
    uint32_t mm1SingleCoreN;
    int64_t tensorAOffset;
    int64_t tensorBOffset;
    int64_t attenMaskOffset;
    int64_t attenMaskOffsetPre;
    uint64_t pseShiftOffset;
    int64_t valueOffset;
    int64_t attentionOutOffset;
    int64_t sInnerOffsetDataSize;

    int64_t sOuterOffset;
    int64_t batchNOffset;
    uint32_t sInnerLoopOffset;
    int64_t multiSeqOffset;
    int64_t multiSeqOffsetBSNDOut;
    int gmPingpong;
    int64_t SoftMaxOffset;
    int taskBatch;

    bool sparseBandSelect0;         // 标志位，sparse band模式，select选0是否要做
    bool sparseBandSelect1;         // 标志位，sparse band模式，select选1是否要做
    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;
    int64_t actualSeqLengthPerBatch = 0;
    int64_t actualSeqLengthKVPerBatch = 0;
};
constexpr int32_t PFA_PARAMS_QUEUE_CAPBABILITY = 4;
constexpr uint32_t SPARSE_ATTENTION_MASK_SIZE = 2048;
constexpr event_t NULL_EVENT = static_cast<event_t>(INVALID_TEVENTID);
constexpr uint32_t MAX_SUBSOUTER_NUM = 64; // Souter最大为512，softmax按照8切块，不超过64块

template <typename PFAT>
class PromptFlashAttentionS1s2Bns1X910Base {
public:
    __aicore__ inline PromptFlashAttentionS1s2Bns1X910Base() {};
    __aicore__ inline void Init(__gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t*  attenMask,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                __gm__ uint8_t*  attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t*  workspace,
                                const PromptFlashAttentionTilingData* __restrict tiling,
                                __gm__ uint8_t* gmTiling, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void InitQuant(__gm__ uint8_t* deq_scale1, __gm__ uint8_t* scale1, __gm__ uint8_t* deq_scale2,
                                     __gm__ uint8_t* scale2, __gm__ uint8_t* offset2);
    __aicore__ inline void InitKvAntiquant(__gm__ uint8_t* antiq_scale, __gm__ uint8_t* antiq_offset);

    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using U = typename PFAT::maskType;
    using O = typename PFAT::outputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmOutputType;
    using computeType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::softmaxType;
    using pseShiftType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::pseShiftType;
    using pseShiftCastType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::pseShiftCastType;
    template <class SRC_T>
    static __aicore__ void CopyND2NZ(const LocalTensor<SRC_T>& dst, const GlobalTensor<SRC_T>& src, const int row, const int col, const int height,
                                    const int width, const int gCol, const int ndNum = 1, const int srcNdMatrixStride = 0,
                                    const int dstNzMatrixStride = 1, const bool kAlignToC0Size = false) {  // 参数取值范围最小为1
        int64_t srcOffset = (int64_t)row * (int64_t)gCol + (int64_t)col;
        int32_t alignNum = 16;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = ndNum;
        nd2nzParams.nValue = height;
        nd2nzParams.dValue = width;
        nd2nzParams.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzParams.srcDValue = gCol;
        if (kAlignToC0Size) {
            if constexpr (IsSameType<SRC_T, int8_t>::value) {
                alignNum = 32;
            } else if constexpr (IsSameType<SRC_T, float>::value) {
                alignNum = 8;
            }
        }
        nd2nzParams.dstNzC0Stride = Ceil(height, alignNum) * alignNum;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = dstNzMatrixStride;
        DataCopy(dst, src[srcOffset], nd2nzParams);
    }

    static __aicore__ void bmm1CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK, int useN,
                                      const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<uint32_t> bmm1LocalInfo;
        bmm1LocalInfo.SetGlobalBuffer((__gm__ uint32_t *)dataPtr, 8);  // 对齐到8个

        uint32_t bmm1BIdx = bmm1LocalInfo.GetValue(0);
        uint32_t bmm1NIdx = bmm1LocalInfo.GetValue(1);
        uint32_t s2SingleOffset = bmm1LocalInfo.GetValue(2);
        uint32_t bmm1TensorBAddr_high = bmm1LocalInfo.GetValue(3);
        uint32_t bmm1TensorBAddr_low = bmm1LocalInfo.GetValue(4);
        uint32_t bmm1BlockTableAddr_high = bmm1LocalInfo.GetValue(5);
        uint32_t bmm1BlockTableAddr_low = bmm1LocalInfo.GetValue(6);
        uint64_t bmm1TensorBAddr = (static_cast<uint64_t>(bmm1TensorBAddr_high) << 32) | static_cast<uint64_t>(bmm1TensorBAddr_low);
        uint64_t bmm1BlockTableAddr = (static_cast<uint64_t>(bmm1BlockTableAddr_high) << 32) | static_cast<uint64_t>(bmm1BlockTableAddr_low);

        // 其他场景用V侧设置的tilingptr里的tiling结果
        __gm__ PromptFlashAttentionTilingData* tilingDataPtr = reinterpret_cast<__gm__ PromptFlashAttentionTilingData*>(tilingPtr);
        // 静态编译时直接用栈上固定的 TilingData
        PromptFlashAttentionTilingData allTilingData;

        uint32_t blockTableDim2;
        uint32_t blockSize;
        uint32_t isLayoutBSH;  // BSH:1  BNSD:0
        uint32_t headNumSize;
        uint32_t headNumRatio;
        uint32_t kvD;
        uint32_t PABlockNumSum;
        uint32_t baseK;
        uint32_t baseN;
        uint32_t Kb;

        if (tilingDataPtr != nullptr) {
            blockTableDim2 = tilingDataPtr->promptAttentionBaseParams.blockTableDim2;
            blockSize = tilingDataPtr->promptAttentionBaseParams.blockSize;
            isLayoutBSH = tilingDataPtr->promptAttentionBaseParams.PAlayoutType;  // BSH:1 BNSD:0
            headNumSize = tilingDataPtr->promptAttentionBaseParams.headNumSize;
            headNumRatio = tilingDataPtr->promptAttentionBaseParams.headNumRatio;
            kvD = tilingDataPtr->promptAttentionBaseParams.headSize;
            PABlockNumSum = tilingDataPtr->promptAttentionBaseParams.PABlockNumSum;
            baseK = tilingDataPtr->bmm1TilingDataRect.baseK;
            baseN = tilingDataPtr->bmm1TilingDataRect.baseN;
            Kb = tilingDataPtr->bmm1TilingDataRect.Kb;
        } else {
            blockTableDim2 = allTilingData.promptAttentionBaseParams.blockTableDim2;
            blockSize = allTilingData.promptAttentionBaseParams.blockSize;
            isLayoutBSH = allTilingData.promptAttentionBaseParams.PAlayoutType;  // BSH:1 BNSD:0
            headNumSize = allTilingData.promptAttentionBaseParams.headNumSize;
            headNumRatio = allTilingData.promptAttentionBaseParams.headNumRatio;
            kvD = allTilingData.promptAttentionBaseParams.headSize;
            PABlockNumSum = allTilingData.promptAttentionBaseParams.PABlockNumSum;
            baseK = allTilingData.bmm1TilingDataRect.baseK;
            baseN = allTilingData.bmm1TilingDataRect.baseN;
            Kb = allTilingData.bmm1TilingDataRect.Kb;
        }

        // bmm1 row方向对应K轴、D轴；col方向对应N轴、S2轴
        uint32_t s2BaseOffset = col * baseN;  // S2方向当前useN块在single块中的偏移
        uint32_t s2AllOffset = s2SingleOffset + s2BaseOffset;  // S2方向总偏移
        uint32_t copyFinishRowCnt = 0;
        uint64_t blockTableIdx = 0;
        uint64_t offsetInBlock = 0;
        uint32_t blockRowOffsetInSingle = 0;
        uint32_t blockId = 0;
        uint32_t copyRowCnt = 0;
        uint64_t curOffset = 0;
        uint32_t baseRowOffsetInSingle = 0;
        uint32_t baseColOffsetInSingle = 0;
        uint32_t colElementCnt = 0;
        uint32_t ndNum = 0;
        uint32_t kvHeadNum = headNumSize / headNumRatio;

        colElementCnt = 32 / sizeof(T);
        int32_t alignUseN = (useN + colElementCnt - 1) / colElementCnt * colElementCnt;

        while (copyFinishRowCnt < useN) {  // 1. useN <= blockSize 拷blockSize的一部分 2. useN > blockSize 一个回调多次拷贝 3. 尾块
            blockTableIdx = s2AllOffset / blockSize;  // block table上的索引  s2AllOffset / blockSize
            offsetInBlock = s2AllOffset % blockSize;  // 最后一个block上的偏移

            blockRowOffsetInSingle = blockTableIdx * blockSize - s2SingleOffset;
            blockId = *(reinterpret_cast<__gm__ int32_t*>(bmm1BlockTableAddr) + bmm1BIdx * blockTableDim2 + blockTableIdx);
            copyRowCnt = blockSize - offsetInBlock;
            if (copyFinishRowCnt + copyRowCnt > useN) {  // 拷贝的比需要拷贝的多
                copyRowCnt = useN - copyFinishRowCnt;
            }

            if (isLayoutBSH == 1) {
                curOffset = blockId * blockSize * kvHeadNum * kvD + bmm1NIdx * kvD;
            } else {
                curOffset = blockId * blockSize * kvHeadNum * kvD + bmm1NIdx * blockSize * kvD;
            }

            GlobalTensor<T> src;  // 伪量化场景也是反量化到fp16存入GM
            src.SetGlobalBuffer((__gm__ T *)bmm1TensorBAddr, PABlockNumSum * blockSize * kvHeadNum * kvD);
            LocalTensor<T> dst = bMatrix.template ReinterpretCast<T>();

            baseRowOffsetInSingle = col * baseN;  // 当前base起始点在single中偏移
            baseColOffsetInSingle = row * baseK;

            if (blockRowOffsetInSingle > baseRowOffsetInSingle) {
                 baseRowOffsetInSingle = 0;
            } else {
                 baseRowOffsetInSingle -= blockRowOffsetInSingle;
            }

            if (blockSize < useN) {
                ndNum = useK / colElementCnt;
                CopyND2NZ(dst[copyFinishRowCnt * colElementCnt], src[curOffset], baseRowOffsetInSingle,
                           baseColOffsetInSingle, copyRowCnt, colElementCnt, Kb, ndNum, colElementCnt, alignUseN * colElementCnt);
            } else {
                CopyND2NZ(dst[0], src[curOffset], baseRowOffsetInSingle, baseColOffsetInSingle, copyRowCnt, useK, Kb);
            }

            // 更新循环变量
            copyFinishRowCnt += copyRowCnt;
            s2AllOffset += copyRowCnt;
        }
    }

    static __aicore__ void bmm2CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK, int useN,
                                      const uint64_t tilingPtr, const uint64_t dataPtr)
    {
        GlobalTensor<uint32_t> bmm2LocalInfo;
        bmm2LocalInfo.SetGlobalBuffer((__gm__ uint32_t*)dataPtr, 8);  // 对齐到8个

        uint32_t bmm2BIdx = bmm2LocalInfo.GetValue(0);
        uint32_t bmm2NIdx = bmm2LocalInfo.GetValue(1);
        uint32_t s2SingleOffset = bmm2LocalInfo.GetValue(2);
        uint32_t bmm2TensorBAddr_high = bmm2LocalInfo.GetValue(3);
        uint32_t bmm2TensorBAddr_low = bmm2LocalInfo.GetValue(4);
        uint32_t bmm2BlockTableAddr_high = bmm2LocalInfo.GetValue(5);
        uint32_t bmm2BlockTableAddr_low = bmm2LocalInfo.GetValue(6);
        uint64_t bmm2TensorBAddr = (static_cast<uint64_t>(bmm2TensorBAddr_high) << 32) | static_cast<uint64_t>(bmm2TensorBAddr_low);
        uint64_t bmm2BlockTableAddr = (static_cast<uint64_t>(bmm2BlockTableAddr_high) << 32) | static_cast<uint64_t>(bmm2BlockTableAddr_low);

        // 其他场景用V侧设置的tilingptr里的tiling结果
        __gm__ PromptFlashAttentionTilingData* tilingDataPtr = reinterpret_cast<__gm__ PromptFlashAttentionTilingData*>(tilingPtr);
        PromptFlashAttentionTilingData allTilingData;

        uint32_t blockTableDim2;
        uint32_t blockSize;
        uint32_t isLayoutBSH;  // BSH:1  BNSD:0
        uint32_t headNumSize;
        uint32_t headNumRatio;
        uint32_t kvD;
        uint32_t PABlockNumSum;
        uint32_t baseK;
        uint32_t baseN;
        uint32_t N;

        if (tilingDataPtr != nullptr) {
            blockTableDim2 = tilingDataPtr->promptAttentionBaseParams.blockTableDim2;
            blockSize = tilingDataPtr->promptAttentionBaseParams.blockSize;
            isLayoutBSH = tilingDataPtr->promptAttentionBaseParams.PAlayoutType;  // BSH:1  BNSD:0
            headNumSize = tilingDataPtr->promptAttentionBaseParams.headNumSize;
            headNumRatio = tilingDataPtr->promptAttentionBaseParams.headNumRatio;
            kvD = tilingDataPtr->promptAttentionBaseParams.headSize;
            PABlockNumSum = tilingDataPtr->promptAttentionBaseParams.PABlockNumSum;
            baseK = tilingDataPtr->bmm2TilingDataRect.baseK;
            baseN = tilingDataPtr->bmm2TilingDataRect.baseN;
            N = tilingDataPtr->bmm2TilingDataRect.N;

        } else {
            blockTableDim2 = allTilingData.promptAttentionBaseParams.blockTableDim2;
            blockSize = allTilingData.promptAttentionBaseParams.blockSize;
            isLayoutBSH = allTilingData.promptAttentionBaseParams.PAlayoutType;  // BSH:1  BNSD:0
            headNumSize = allTilingData.promptAttentionBaseParams.headNumSize;
            headNumRatio = allTilingData.promptAttentionBaseParams.headNumRatio;
            kvD = allTilingData.promptAttentionBaseParams.headSize;
            PABlockNumSum = allTilingData.promptAttentionBaseParams.PABlockNumSum;
            baseK = allTilingData.bmm2TilingDataRect.baseK;
            baseN = allTilingData.bmm2TilingDataRect.baseN;
            N = allTilingData.bmm2TilingDataRect.N;
        }
        // bmm2 row方向对应K轴、S2轴；col方向对应N轴、D轴
        uint32_t s2BaseOffset = row * baseK;
        uint32_t s2AllOffset = s2SingleOffset + s2BaseOffset;
        uint32_t kvHeadNum = headNumSize / headNumRatio;
        uint32_t copyFinishRowCnt = 0;
        uint64_t blockTableIdx = 0;
        uint64_t offsetInBlock = 0;
        uint32_t blockRowOffsetInSingle = 0;
        uint32_t blockId = 0;
        uint32_t copyRowCnt = 0;
        uint64_t curOffset = 0;
        uint32_t baseRowOffsetInSingle = 0;
        uint32_t baseColOffsetInSingle = 0;
        uint32_t colElementCnt = 0;
        uint32_t ndNum = 0;

        colElementCnt = 32 / sizeof(T);
        int32_t alignUseK = (useK + colElementCnt - 1) / colElementCnt * colElementCnt;

        while (copyFinishRowCnt < useK) {
            blockTableIdx = s2AllOffset / blockSize;
            offsetInBlock = s2AllOffset % blockSize;

            blockRowOffsetInSingle = blockTableIdx * blockSize - s2SingleOffset;
            blockId = *(reinterpret_cast<__gm__ int32_t*>(bmm2BlockTableAddr) + bmm2BIdx * blockTableDim2 + blockTableIdx);

            copyRowCnt = blockSize - offsetInBlock;
            if (copyFinishRowCnt + copyRowCnt > useK) {
                copyRowCnt = useK - copyFinishRowCnt;
            }

            if (isLayoutBSH == 1) {
                curOffset = blockId * blockSize * kvHeadNum * kvD + bmm2NIdx * kvD;
            } else {
                curOffset = blockId * blockSize * kvHeadNum * kvD + bmm2NIdx * blockSize * kvD;
            }

            GlobalTensor<T> src;
            src.SetGlobalBuffer((__gm__ T *)bmm2TensorBAddr, PABlockNumSum * blockSize * kvHeadNum * kvD);

            LocalTensor<T> dst = bMatrix.template ReinterpretCast<T>();

            baseRowOffsetInSingle = row * baseK;
            baseColOffsetInSingle = col * baseN;

            if (blockRowOffsetInSingle > baseRowOffsetInSingle) {
                 baseRowOffsetInSingle = 0;
            } else {
                 baseRowOffsetInSingle -= blockRowOffsetInSingle;
            }

            if (blockSize < useK) {
                ndNum = useN / colElementCnt;
                CopyND2NZ(dst[copyFinishRowCnt * colElementCnt], src[curOffset], baseRowOffsetInSingle,
                           baseColOffsetInSingle, copyRowCnt, colElementCnt, N, ndNum, colElementCnt, alignUseK * colElementCnt, true);
            } else {
                CopyND2NZ(dst[0], src[curOffset], baseRowOffsetInSingle,
                           baseColOffsetInSingle, copyRowCnt, useN, N, 1, 0, 1, true);
            }

            // 更新循环变量
            copyFinishRowCnt += copyRowCnt;
            s2AllOffset += copyRowCnt;
        }
    }
    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, T, true, LayoutMode::NONE, PFAT::ibShare>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, mmOutputType>;
    using PACBmm1 = typename AscendC::Conditional<PFAT::MM_TYPE == MatMulType::MM_PA,
                                                  Matmul<a1Type, b1Type, c1Type, bias1Type, PFAT::mmCFG, matmul::MatmulCallBackFunc<nullptr, nullptr, bmm1CopyB1>>,
                                                  Matmul<a1Type, b1Type, c1Type, bias1Type, PFAT::mmCFG>>::type;  // PA暂时不用大包搬运
    PACBmm1 mm;
    // define batchmatmul
    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LayoutMode::NONE, PFAT::ibShare>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmOutputType>;
    using PACBmm2 = typename AscendC::Conditional<PFAT::MM_TYPE == MatMulType::MM_PA,
                                                  Matmul<a2Type, b2Type, c2Type, bias2Type, PFAT::mmCFG, matmul::MatmulCallBackFunc<nullptr, nullptr, bmm2CopyB1>>,
                                                  Matmul<a2Type, b2Type, c2Type, bias2Type, PFAT::mmCFG>>::type;  // PA暂时不用大包搬运
    PACBmm2 bmm2;

protected:
    const PromptFlashAttentionTilingData* __restrict tilingData;
    TPipe* pipe;
    // define the que
    TQue<QuePosition::VECIN, 1> tempBmm2Queue;
    TQue<QuePosition::VECOUT, 1> Bmm1Queue;
    TQue<QuePosition::VECOUT, 1> softmaxOutQueue;

    TBuf<> PABmm1UB;
    TBuf<> PABmm2UB;
    TBuf<> selectSpaceUb;
    TBuf<> pseShiftCastUb;
    TBuf<> softmaxExpUb_;
    TBuf<> tempBmm2Ub;

    event_t pseShiftEvent = NULL_EVENT;
    event_t bmm1ResCopyInEvent[2];
    event_t bmm2ResCopyInEvent[2];
    event_t bmm1ResCopyOutEvent[2];
    event_t attenOutCopyOut;
    DataCopyParams mm1GmUbCopyParam[2];

    bool copyOutPrevIter = false;
    uint32_t softmaxSouterStepLen = 0;
    bool needAdd;

    LocalTensor<uint32_t> bmm1LocalInfo;
    LocalTensor<uint32_t> bmm2LocalInfo;
    LocalTensor<computeType> mmResUb[2];
    LocalTensor<float> softmaxMaxUb;
    LocalTensor<float> softmaxSumUb;
    LocalTensor<computeType> softmaxExpUb;
    LocalTensor<U> attenMaskUb;
    LocalTensor<pseShiftType> pseShiftUb;

    __gm__ uint8_t* key_ptr;
    __gm__ uint8_t* value_ptr;
    __gm__ uint8_t* currentKey;
    __gm__ uint8_t* currentValue;
    __gm__ uint8_t* blocktable_ptr;

    __gm__ uint32_t* bmm1CBDataPtr[2];
    __gm__ uint32_t* bmm2CBDataPtr[2];

    GlobalTensor<T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<U> attenMaskGm;
    GlobalTensor<O> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    GlobalTensor<pseShiftType> pseShiftGm;
    GlobalTensor<computeType> workspaceGm;
    GlobalTensor<computeType> bmm1ResGmDb[2];
    GlobalTensor<int8_t> quant1ResGmDb[2];
    GlobalTensor<computeType> bmm2ResGmDb[2];
    GlobalTensor<uint32_t> bmm1CBDataGm[2];
    GlobalTensor<uint32_t> bmm2CBDataGm[2];

    GlobalTensor<int64_t> queryPaddingSizeGm;
    GlobalTensor<int64_t> kvPaddingSizeGm;

    GlobalTensor<KV_T> keySharedPrefixGm;
    GlobalTensor<KV_T> valueSharedPrefixGm;
    GlobalTensor<int64_t> actualSharedPrefixLenGm;
    int64_t actualKVPrefixLen = 0;

    // quant: define quant variable
    uint64_t dequantScale1 = 0;
    float quantScale1 = 0;
    uint64_t dequantScale2 = 0;
    float quantScale2 = 0;
    float quantOffset2 = 0;

    GlobalTensor<uint32_t> deqScale1Fp32Gm;
    GlobalTensor<uint32_t> deqScale2Fp32Gm;

    // quant bf16 per-channel
    bool isQuant2PerChn = false;
    bool isQuant2BF16 = false;
    bool isQuantOffset2Exist = false;
    uint32_t perChannelQuantUBSize = 0;
    float quant2ScaleValue = 0;
    float quant2OffsetValue = 0;
    GlobalTensor<bfloat16_t> quantScale2BF16Gm;
    GlobalTensor<bfloat16_t> quantOffset2BF16Gm;

    GlobalTensor<float> quantScale2FP32Gm;
    GlobalTensor<float> quantOffset2FP32Gm;

    TBuf<> quantScale2BF16Ub;
    TBuf<> quantOffset2BF16Ub;
    TBuf<> quantScale2FloatUb;
    TBuf<> quantOffset2FloatUb;

    // kv antiquant
    bool isAntiquantSymmetric = false;
    T keyAntiquantScale;
    T keyAntiquantOffset;
    T valueAntiquantScale;
    T valueAntiquantOffset;
    TQue<QuePosition::VECIN, 1> kvAntiquantSrcQueue;
    TQue<QuePosition::VECOUT, 1> kvAntiquantDstQueue;
    TBuf<> antiquantScaleUb;
    TBuf<> antiquantOffsetUb;
    GlobalTensor<T> keyGmAntiquant;
    GlobalTensor<T> valueGmAntiquant;
    GlobalTensor<T> antiquantScaleGm;
    GlobalTensor<T> antiquantOffsetGm;

    PFAComputeParam pfaParamsQueue[PFA_PARAMS_QUEUE_CAPBABILITY];
    PFAComputeParam *tailParams;
    PFAComputeParam *headParams;
    PFAComputeParam *preHeadParams;
    int32_t headId = 0;
    int32_t tailId = 0;
    int32_t queSize = 0;
    int32_t queSizeLimit = PFA_PARAMS_QUEUE_CAPBABILITY - 2;

    int64_t tmp_block_idx = 0;
    int64_t maskOffset = 0;
    int64_t maskCoreOffset = 0;
    uint64_t attenMaskCoreOffset = 0;
    int64_t valueCoreOffset = 0;
    int64_t valuePrefixCoreOffset = 0;
    int64_t tensorACoreOffset = 0;
    int64_t tensorBCoreOffset = 0;
    int64_t tensorBPrefixCoreOffset = 0;
    int64_t offsetSS = 0;
    int64_t offsetSH = 0;
    int64_t offsetSTypeNum = 0;
    int64_t offsetNSTypeNum = 0;
    int64_t offsetNSS = 0;
    int64_t offsetNSH = 0;
    uint32_t maskDataType = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t s2InCurrentBatch = 0;
    AscendC::TensorDesc<__gm__ uint8_t> kvTensorDesc;

    uint32_t mm1SingleCoreNPrev = 0;
    uint32_t mm2MStridePrev = 0;
    uint32_t mm2KaStridePrev = 0;
    uint64_t pseShiftCoreOffset = 0;
    uint32_t pseShiftBatch = 0;

    // tilingdata
    uint32_t singleProcessSOuterSizeWhole = 0;
    uint32_t singleProcessSOuterSizeTail = 0;
    uint32_t mmResUbSize = 0;
    uint32_t attenMaskUbSize = 0;
    uint32_t maskSize = 0;
    uint32_t pseShiftUbSize = 0;
    uint32_t pseShiftTypeByteNum = 0;
    uint32_t pseShiftStride = 0;
    uint32_t softmaxMaxSize = 0;
    uint32_t softmaxSumSize = 0;
    uint32_t softmaxExpSize = 0;
    uint32_t spmTmpSize = 0;
    uint32_t scmTmpSize = 0;
    uint32_t bmm2ResUbSize = 0;
    uint32_t tmpMMResBmm2PreUbSize = 0;
    uint32_t tmpSoftmaxBmm2UbSize = 0;
    uint32_t typeByteNum = 0;
    uint32_t outputTypeByteNum = 0;
    uint32_t softmaxTypeByteNum = 0;
    uint32_t headNumRatio = 0;
    uint32_t maskTypeByteNum = 0;
    uint32_t selectSpaceUbSize = 0;
    uint32_t maskBmm2ShareSize = 0;

    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxFlashTilingData;
    CopyTransposeTiling transposeTilingData;
    uint32_t MultiHeadQ = 0;
    uint32_t MultiHeadKV = 0;
    uint32_t maxInnerLoopPrefixTimes = 0;
    uint32_t maxInnerLoopTimes = 0;
    int64_t seqListOffset = 0;

    uint32_t attentionMaskStride = 0;
    int32_t attentionMaskType = 0;
    uint32_t negativeScalar = NEGATIVE_MIN_VAULE_FP32;
    bool isSoftmaxResNeedUpdate;
    bool isSoftmaxLseNeedUpdate = false;
    bool isSoftmaxNeedUpdate[MAX_SUBSOUTER_NUM];

    bool isGlobalFirstCompute;

    bool isActualLenDimsNull;
    bool isActualLenDimsKVNull;
    uint64_t actualSeqOffsets[BATCH_NUM_MAX];
    uint32_t isKvContinuous = 0;
    uint32_t fromFused = 0;

    __aicore__ inline void SoftmaxBasicComputeFirstNoTail(LocalTensor<computeType>& mmResUb,
                                                          LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, uint32_t souterSize);

    __aicore__ inline void SoftmaxBasicComputeNoTail(LocalTensor<computeType>& mmResUb,
                                                     LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                   LocalTensor<computeType>& softmaxExpUb, uint32_t souterSize);

    __aicore__ inline void SoftmaxComputeFirstTail(LocalTensor<computeType>& mmResUb,
                                                   LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, uint32_t souterSize);

    __aicore__ inline void SoftmaxComputeTail(LocalTensor<computeType>& mmResUb,
                                              LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                              LocalTensor<computeType>& softmaxExpUb, uint32_t souterSize);

    __aicore__ inline void Bmm2UpdateDivNoTail(LocalTensor<computeType>& bmm2ResPreUb, LocalTensor<float>& softmaxSumUb);

    __aicore__ inline void UpdateVmul(LocalTensor<computeType>& softmaxExpUb);

    __aicore__ inline void Bmm2UpdateAdd(LocalTensor<computeType>& bmm2ResUb);

    __aicore__ inline void QuantCompute(LocalTensor<int8_t> quantResUb, LocalTensor<computeType> mmResUb, float scale,
                                        float offset, uint32_t computeSize);

    __aicore__ inline void CalPseShiftOffset(int sIdx);

    TEMPLATE_LAYOUT
    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float>& softmaxSumUb, LocalTensor<float>& softmaxMaxUb) {
        struct DataCopyParams dataCopyParams;
        uint32_t souterSize = this->headParams->singleProcessSOuterSize;

        dataCopyParams.blockCount = souterSize;
        dataCopyParams.blockLen = 4; // float 4 Bytes
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        LocalTensor<float> lseUb = this->softmaxExpUb_.template Get<float>(this->softmaxMaxSize);
        Log(lseUb, softmaxSumUb, souterSize * 8); // 8 :softmax的第二维
        pipe_barrier(PIPE_V);

        Add(lseUb, lseUb, softmaxMaxUb, souterSize * 8); // 8 :softmax的第二维
        pipe_barrier(PIPE_V);
        
        if(this->isSoftmaxLseNeedUpdate){
            SoftMaxShapeInfo softmaxShapeInfo = {
                static_cast<uint32_t>(souterSize),
                static_cast<uint32_t>(8),
                static_cast<uint32_t>(souterSize),
                static_cast<uint32_t>(8)
            };
            AdjustSoftMaxRes<float, float>(lseUb,  softmaxMaxUb, this->negativeScalar, 3e+99, softmaxShapeInfo);   
            pipe_barrier(PIPE_V);
        }
        
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(enQueEvtID);
        WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
        
        // lse datacopy
        DataCopyPad(softmaxLseGm[this->headParams->SoftMaxOffset], lseUb, dataCopyParams); // 8 :softmax的第二维
        event_t enQueEvtID0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(enQueEvtID0);
        WaitFlag<HardEvent::MTE3_V>(enQueEvtID0);
    }

    __aicore__ inline void PostQuant2PerChannelBF16(LocalTensor<computeType> &bmm2ResUb, LocalTensor<int8_t> &outputQuantRes);

    __aicore__ inline void PostQuant2PerChannelFP32(LocalTensor<computeType> &bmm2ResUb, LocalTensor<int8_t> &outputQuantRes);

    TEMPLATE_LAYOUT
    __aicore__ inline void DataCopyTransposeOutBSH(LocalTensor<computeType> &bmm2ResUb) {
        TransposeParams transposeParams;
        transposeParams.bIndex = 0;
        transposeParams.nIndex = this->preHeadParams->batchNOffset;
        transposeParams.sIndex = this->preHeadParams->sOuterOffset;
        transposeParams.hNIndex = 0;
        int64_t preTokensOffset = 0;
        int64_t nextTokensOffset = 0;
        if (this->preHeadParams->preTokensPerBatch < 0) {
            int64_t preTokenLength = this->preHeadParams->actualSeqLengthKVPerBatch + actualKVPrefixLen + this->preHeadParams->preTokensPerBatch;
            if (this->preHeadParams->sOuterOffset < preTokenLength &&
                (this->preHeadParams->sOuterOffset + this->preHeadParams->singleProcessSOuterSize) > preTokenLength) {
                preTokensOffset = this->preHeadParams->sOuterOffset + this->preHeadParams->singleProcessSOuterSize - preTokenLength;
            } else {
                preTokensOffset = 0;
            }
        }

        if (this->preHeadParams->sOuterOffset < this->preHeadParams->nextTokensPerBatch * (-1) &&
            (this->preHeadParams->sOuterOffset + this->preHeadParams->singleProcessSOuterSize) > this->preHeadParams->nextTokensPerBatch * (-1)) {
            nextTokensOffset = this->preHeadParams->nextTokensPerBatch * (-1) - this->preHeadParams->sOuterOffset;
        } else {
            nextTokensOffset = 0;
        }

        CopyTransposeTiling transposeTilingData22 = tilingData->transposeTilingDataRect;
        transposeTilingData22.srcShapeS = this->preHeadParams->singleProcessSOuterSize - preTokensOffset - nextTokensOffset;
        transposeTilingData22.invalidParamCopyTransposeTiling = 0;
        transposeParams.sIndex = transposeParams.sIndex + nextTokensOffset;

        int64_t multiSeqOffset = (this->tilingData->promptAttentionBaseParams.isQHasLeftPadding && this->tilingData->promptAttentionBaseParams.isBSNDOut) ?
            this->preHeadParams->multiSeqOffsetBSNDOut : this->preHeadParams->multiSeqOffset;
        if constexpr (IsSameType<O, int8_t>::value) {
            LocalTensor<int8_t> outputQuantRes;
            outputQuantRes = bmm2ResUb.template ReinterpretCast<int8_t>();
            outputQuantRes.SetSize(bmm2ResUb.GetSize());
            if (isQuant2PerChn) {                                         // per-channel
                if (isQuant2BF16) {                                       // scale2和offset2为bf16，此时qkv也为bf16，bmm2输出fp32，scale2和offset2需要做cast到fp32
                    PostQuant2PerChannelBF16(bmm2ResUb, outputQuantRes);
                } else {                                                  // scale2和offset2为fp32，高性能下需要cast到fp16，高精度/bf16下直接做quant
                    PostQuant2PerChannelFP32(bmm2ResUb, outputQuantRes);
                }
            } else {
                QuantCompute(outputQuantRes, bmm2ResUb, quantScale2, quantOffset2, bmm2ResUbSize);
            }
            event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
            DataCopyTranspose2<O> (attentionOutGm, outputQuantRes[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize],
                                CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams,
                                transposeTilingData22, multiSeqOffset);
        } else if constexpr (PFAT::calcMode == Mode::HighPrecision ||
            IsSameType<T, bfloat16_t>::value) {
            LocalTensor<T> FinalResUb = bmm2ResUb.template ReinterpretCast<T>();

            pipe_barrier(PIPE_V);
            Cast(FinalResUb, bmm2ResUb, RoundMode::CAST_ROUND, bmm2ResUb.GetSize());

            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopyTranspose2<O> (attentionOutGm, FinalResUb[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize],
                                   CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams,
                                   transposeTilingData22, multiSeqOffset);
        } else {
            // copyOut 前同步计算
            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopyTranspose2<O> (attentionOutGm, bmm2ResUb[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize],
                                   CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams,
                                   transposeTilingData22, multiSeqOffset);
        }
    }

    TEMPLATE_LAYOUT
    __aicore__ inline void DataCopyTransposeOutBNSD(LocalTensor<computeType> &bmm2ResUb) {
        int64_t preTokensOffset = 0;
        int64_t nextTokensOffset = 0;
        uint64_t copySize = this->preHeadParams->singleProcessSOuterSize * \
            tilingData->promptAttentionBaseParams.headSize;
        if (this->preHeadParams->preTokensPerBatch < 0) {
            int64_t preTokenLength = this->preHeadParams->actualSeqLengthKVPerBatch + actualKVPrefixLen + this->preHeadParams->preTokensPerBatch;
            if (this->preHeadParams->sOuterOffset < preTokenLength &&
                (this->preHeadParams->sOuterOffset + this->preHeadParams->singleProcessSOuterSize) > preTokenLength) {
                preTokensOffset = this->preHeadParams->sOuterOffset + this->preHeadParams->singleProcessSOuterSize - preTokenLength;
                copySize = copySize - preTokensOffset * tilingData->promptAttentionBaseParams.headSize;
            } else {
                preTokensOffset = 0;
            }
        }

        if (this->preHeadParams->sOuterOffset < this->preHeadParams->nextTokensPerBatch * (-1) &&
            (this->preHeadParams->sOuterOffset + this->preHeadParams->singleProcessSOuterSize) > this->preHeadParams->nextTokensPerBatch * (-1)) {
            nextTokensOffset = this->preHeadParams->nextTokensPerBatch * (-1) - this->preHeadParams->sOuterOffset;
            copySize = copySize - nextTokensOffset * tilingData->promptAttentionBaseParams.headSize;
        } else {
            nextTokensOffset = 0;
        }
        int64_t attentionOutTokenOffset = nextTokensOffset * tilingData->promptAttentionBaseParams.headSize;

        struct DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copySize / outputTypeByteNum;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        if constexpr (IsSameType<O, int8_t>::value) {
            LocalTensor<int8_t> outputQuantRes;
            outputQuantRes = bmm2ResUb.template ReinterpretCast<int8_t>();
            outputQuantRes.SetSize(bmm2ResUb.GetSize());
            if (isQuant2PerChn) {                                         // per-channel
                if (isQuant2BF16) {                                       // scale2和offset2为bf16，此时qkv也为bf16，bmm2输出fp32，scale2和offset2需要做cast到fp32
                    PostQuant2PerChannelBF16(bmm2ResUb, outputQuantRes);
                } else {                                                  // scale2和offset2为fp32，高性能下需要cast到fp16，高精度/bf16下直接做quant
                    PostQuant2PerChannelFP32(bmm2ResUb, outputQuantRes);
                }
            } else {
                QuantCompute(outputQuantRes, bmm2ResUb, quantScale2, quantOffset2, bmm2ResUbSize);
            }
            event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
            DataCopy(attentionOutGm[this->preHeadParams->attentionOutOffset + attentionOutTokenOffset],
                        outputQuantRes[attentionOutTokenOffset], dataCopyParams);
        } else if constexpr (PFAT::calcMode == Mode::HighPrecision ||
            IsSameType<T, bfloat16_t>::value) {
            LocalTensor<T> FinalResUb = bmm2ResUb.template ReinterpretCast<T>();

            pipe_barrier(PIPE_V);
            Cast(FinalResUb, bmm2ResUb, RoundMode::CAST_ROUND, bmm2ResUb.GetSize());

            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopy(attentionOutGm[this->preHeadParams->attentionOutOffset + attentionOutTokenOffset],
                     FinalResUb[attentionOutTokenOffset], dataCopyParams);
        } else {
            // copyOut 前同步计算
            SetFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            WaitFlag<HardEvent::V_MTE3>(attenOutCopyOut);
            DataCopy(attentionOutGm[this->preHeadParams->attentionOutOffset + attentionOutTokenOffset],
                     bmm2ResUb[attentionOutTokenOffset], dataCopyParams);
        }
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BSH_VOID ComputeOffset(PFAComputeParam *params,uint32_t sInnerLoopIdx, int32_t firstInnerMargin) {
        int64_t sInnerOffsetDataSize = 0;
        int64_t computeOffset = 0;
        if constexpr (PFAT::enablePrefix) {
            if (!params->isPrefixInnerIter) {
                int64_t prefixIdxOffset = (actualKVPrefixLen + this->tailParams->singleProcessSInnerSize - 1) / this->tailParams->singleProcessSInnerSize;
                sInnerOffsetDataSize = (((int64_t)sInnerLoopIdx - prefixIdxOffset) * this->tailParams->singleProcessSInnerSize + firstInnerMargin);
                computeOffset = sInnerOffsetDataSize + this->actualKVPrefixLen;
                this->tailParams->tensorBOffset = tensorBCoreOffset + (int64_t)sInnerOffsetDataSize * (int64_t)MultiHeadKV;
                this->tailParams->valueOffset = valueCoreOffset + (int64_t)sInnerOffsetDataSize * (int64_t)MultiHeadKV;
            } else {
                sInnerOffsetDataSize = ((int64_t)sInnerLoopIdx * this->tailParams->singleProcessSInnerSize + firstInnerMargin);
                computeOffset = sInnerOffsetDataSize;
                this->tailParams->tensorBOffset = tensorBPrefixCoreOffset + (int64_t)sInnerOffsetDataSize * (int64_t)MultiHeadKV;
                this->tailParams->valueOffset = valuePrefixCoreOffset + (int64_t)sInnerOffsetDataSize * (int64_t)MultiHeadKV;
            }
        } else {
            sInnerOffsetDataSize = (int64_t)sInnerLoopIdx * this->tailParams->singleProcessSInnerSize + firstInnerMargin;
            computeOffset = sInnerOffsetDataSize;
            this->tailParams->tensorBOffset = tensorBCoreOffset + (int64_t)sInnerOffsetDataSize * (int64_t)MultiHeadKV;
            this->tailParams->valueOffset = valueCoreOffset + (int64_t)sInnerOffsetDataSize * (int64_t)MultiHeadKV;
        }
        ComputePseShiftOffset(computeOffset);
        ComputeAttenMaskOffset(computeOffset);
        ComputeAttenMaskOffsetPre(computeOffset);

        this->tailParams->sInnerOffsetDataSize = sInnerOffsetDataSize;
        this->tailParams->tensorAOffset = tensorACoreOffset;
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BNSD_VOID ComputeOffset(PFAComputeParam *params,uint32_t sInnerLoopIdx, int32_t firstInnerMargin) {
        int64_t sInnerOffsetDataSize = 0;
        int64_t computeOffset = 0;
        if constexpr (PFAT::enablePrefix) {
            if (!params->isPrefixInnerIter) {
                int64_t prefixIdxOffset = (actualKVPrefixLen + this->tailParams->singleProcessSInnerSize - 1) / this->tailParams->singleProcessSInnerSize;
                sInnerOffsetDataSize = (((int64_t)sInnerLoopIdx - prefixIdxOffset) * this->tailParams->singleProcessSInnerSize + firstInnerMargin);
                computeOffset = sInnerOffsetDataSize + this->actualKVPrefixLen;
                this->tailParams->valueOffset = valueCoreOffset + (int64_t)sInnerOffsetDataSize * (int64_t)this->tilingData->promptAttentionBaseParams.headSize;
            } else {
                sInnerOffsetDataSize = ((int64_t)sInnerLoopIdx * this->tailParams->singleProcessSInnerSize + firstInnerMargin);
                computeOffset = sInnerOffsetDataSize;
                this->tailParams->valueOffset = valuePrefixCoreOffset + (int64_t)sInnerOffsetDataSize * (int64_t)this->tilingData->promptAttentionBaseParams.headSize;
            }
        } else {
            sInnerOffsetDataSize = (int64_t)sInnerLoopIdx * this->tailParams->singleProcessSInnerSize + firstInnerMargin;
            computeOffset = sInnerOffsetDataSize;
            this->tailParams->valueOffset = valueCoreOffset + (int64_t)sInnerOffsetDataSize * (int64_t)this->tilingData->promptAttentionBaseParams.headSize;
        }
        ComputePseShiftOffset(computeOffset);
        ComputeAttenMaskOffset(computeOffset);
        ComputeAttenMaskOffsetPre(computeOffset);

        this->tailParams->sInnerOffsetDataSize = sInnerOffsetDataSize;
    }

    __aicore__ inline int64_t GetQueryLeftPaddingSize(int sIdx) {
        if (!this->tilingData->promptAttentionBaseParams.isQHasLeftPadding) {
            return 0;
        }
        int64_t rightPaddingSize = this->queryPaddingSizeGm.GetValue(0) > 0 ? this->queryPaddingSizeGm.GetValue(0) : 0;
        int64_t leftPaddingSize = this->tilingData->promptAttentionBaseParams.seqSize - this->actualSeqLengthsGm.GetValue(sIdx) - rightPaddingSize;

        return leftPaddingSize > 0 ? leftPaddingSize : 0;
    }

    __aicore__ inline int64_t GetKVLeftPaddingSize(int sIdx) {
        if (!this->tilingData->promptAttentionBaseParams.isKVHasLeftPadding) {
            return 0;
        }
        int64_t rightPaddingSize = this->kvPaddingSizeGm.GetValue(0) > 0 ? this->kvPaddingSizeGm.GetValue(0) : 0;
        int64_t leftPaddingSize = this->tilingData->promptAttentionBaseParams.seqInnerSize - this->actualSeqLengthsKVGm.GetValue(sIdx) - rightPaddingSize;

        return leftPaddingSize > 0 ? leftPaddingSize : 0;
    }

    __aicore__ inline int64_t CalMultiSeqOffset(int sIdx) {
        int64_t multiSeqOffset = 0;
        int64_t queryLeftpaddingSize = this->GetQueryLeftPaddingSize(sIdx);
        if constexpr (PFAT::layout == PFALayout::BNSD) {
            multiSeqOffset = (int64_t)sIdx * this->tilingData->promptAttentionBaseParams.seqSize * (int64_t)this->MultiHeadQ +
                queryLeftpaddingSize * (int64_t)this->tilingData->promptAttentionBaseParams.headSize; // BNSD
        } else {
            multiSeqOffset = (int64_t)sIdx * this->tilingData->promptAttentionBaseParams.seqSize * (int64_t)this->MultiHeadQ +
                queryLeftpaddingSize * (int64_t)this->MultiHeadQ; // BSH
        }

        if (this->tilingData->promptAttentionBaseParams.isBSNDOut) {
            this->tailParams->multiSeqOffsetBSNDOut = (int64_t)sIdx * (int64_t)this->tilingData->promptAttentionBaseParams.seqSize *
                (int64_t)this->MultiHeadQ + queryLeftpaddingSize * (int64_t)this->MultiHeadQ;
        }

        return multiSeqOffset;
    }

    __aicore__ inline int64_t CalMultiSeqLseOffset(int sIdx, PFAComputeParam* params) {
      int64_t lseOffset = 0;
      int64_t queryLeftpaddingSize = this->GetQueryLeftPaddingSize(sIdx);
      // B offset
      lseOffset = (int64_t)sIdx * this->tilingData->promptAttentionBaseParams.seqSize *
                      (int64_t)this->tilingData->promptAttentionBaseParams.headNumSize +
                  (int64_t)params->batchNOffset * tilingData->promptAttentionBaseParams.seqSize +
                  (int64_t)params->sOuterOffset + queryLeftpaddingSize;
      return lseOffset;
    }

    TEMPLATE_LAYOUT
   __aicore__ inline TYPENAME_BSH_VOID LoopSOuterOffsetInit(int64_t seqListOffsetSize, int sIdx) {
        CalPseShiftOffset(sIdx);

        uint64_t attenMaskBatchOffset = 0;
        if (attenMaskBatch != 1) {
            attenMaskBatchOffset = (uint64_t)sIdx * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
                                (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
        }
        int64_t queryLeftpaddingSize = GetQueryLeftPaddingSize(sIdx);
        int64_t kvLeftPaddingSize = GetKVLeftPaddingSize(sIdx);
        attenMaskCoreOffset = attenMaskBatchOffset + (uint64_t)(this->tailParams->sOuterOffset + queryLeftpaddingSize) * \
            (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize + (uint64_t)kvLeftPaddingSize;

        tensorACoreOffset = (int64_t)seqListOffsetSize +
                            (int64_t)this->tailParams->sOuterOffset * (int64_t)MultiHeadQ +
                            (int64_t)this->tailParams->batchNOffset * (int64_t)tilingData->promptAttentionBaseParams.headSize;
        int64_t seqInnerOffsetSize;
        if (this->tilingData->promptAttentionBaseParams.isKVHasLeftPadding) {
            seqInnerOffsetSize = ((int64_t)sIdx * (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize + kvLeftPaddingSize) * (int64_t)MultiHeadKV;
        } else if (this->isKvContinuous == 1) {
            // 这是从KV的GM 到 每一个batch的开始地址 所需要的偏移量，即每一个batch需要偏移前面一整个batch的长度
            seqInnerOffsetSize =
                tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
                seqListOffsetSize / headNumRatio : (int64_t)sIdx * (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize * (int64_t)MultiHeadKV;
        } else {
            //KV tensorlist场景下，我们能直接将KV的GM设置成当前batch的开始地址，所以偏移量总是0
            seqInnerOffsetSize = 0;
        }
        tensorBCoreOffset = (int64_t)seqInnerOffsetSize +
                        this->tailParams->batchNOffset / headNumRatio * tilingData->promptAttentionBaseParams.headSize;

        valueCoreOffset = tensorBCoreOffset;
        this->tailParams->SoftMaxOffset = this->CalMultiSeqLseOffset(sIdx, this->tailParams);
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BNSD_VOID LoopSOuterOffsetInit(int64_t seqListOffsetSize, int sIdx) {
        uint64_t head_stride_q = tilingData->promptAttentionBaseParams.headSize *
                                tilingData->promptAttentionBaseParams.seqSize;
        uint32_t head_stride_kv;
        if (this->isKvContinuous == 1) {
            head_stride_kv = tilingData->promptAttentionBaseParams.headSize *
                                    tilingData->promptAttentionBaseParams.seqInnerSize;
        } else {
            head_stride_kv = tilingData->promptAttentionBaseParams.headSize *
                                    s2InCurrentBatch;
        }
        uint32_t seq_stride = tilingData->promptAttentionBaseParams.headSize;

        CalPseShiftOffset(sIdx);

        uint64_t attenMaskBatchOffset = 0;
        if (attenMaskBatch != 1) {
            attenMaskBatchOffset = (uint64_t)sIdx * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
                                (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
        }
        int64_t queryLeftpaddingSize = GetQueryLeftPaddingSize(sIdx);
        int64_t kvLeftPaddingSize = GetKVLeftPaddingSize(sIdx);
        attenMaskCoreOffset = attenMaskBatchOffset + ((uint64_t)this->tailParams->sOuterOffset + (uint64_t)queryLeftpaddingSize) * \
            (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize + (uint64_t)kvLeftPaddingSize;

        tensorACoreOffset = (int64_t)seqListOffsetSize + \
            (int64_t)this->tailParams->batchNOffset * (int64_t)head_stride_q + \
            (int64_t)this->tailParams->sOuterOffset * (int64_t)seq_stride;
        int64_t seqInnerOffsetSize;
        if (this->tilingData->promptAttentionBaseParams.isKVHasLeftPadding) {
            seqInnerOffsetSize = (int64_t)sIdx * (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize * (int64_t)MultiHeadKV +
                (int64_t)kvLeftPaddingSize * (int64_t)tilingData->promptAttentionBaseParams.headSize;
        } else if (this->isKvContinuous == 1) {
            seqInnerOffsetSize =
                tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
                (int64_t)seqListOffsetSize / (int64_t)headNumRatio : (int64_t)sIdx * (int64_t)head_stride_kv *
                (int64_t)tilingData->promptAttentionBaseParams.headNumSize / (int64_t)headNumRatio;
        } else {
            seqInnerOffsetSize = 0;
        }
        tensorBCoreOffset = (int64_t)seqInnerOffsetSize + \
            (int64_t)this->tailParams->batchNOffset / (int64_t)headNumRatio * (int64_t)head_stride_kv;

        valueCoreOffset = tensorBCoreOffset;

        this->tailParams->attentionOutOffset = (int64_t)seqListOffsetSize + \
            (int64_t)this->tailParams->batchNOffset * (int64_t)head_stride_q + (int64_t)this->tailParams->sOuterOffset * (int64_t)seq_stride;
        this->tailParams->SoftMaxOffset = this->CalMultiSeqLseOffset(sIdx, this->tailParams);
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BSH_INT64 GetBmm1TensorBOffset(PFAComputeParam *params,
        int32_t sInnerLoopIdx, int32_t firstInnerMargin) {
        int64_t prefixIdxOffset = 0;
        if constexpr (PFAT::enablePrefix) {
            if (params->isPrefixInnerIter) {
                return this->tensorBPrefixCoreOffset + ((int64_t)sInnerLoopIdx * (int64_t)params->singleProcessSInnerSize + (int64_t)firstInnerMargin) * (int64_t)this->MultiHeadKV;
            }
            prefixIdxOffset = (actualKVPrefixLen + (int64_t)params->singleProcessSInnerSize - 1) / (int64_t)params->singleProcessSInnerSize;
        }
        return this->tensorBCoreOffset + (((int64_t)sInnerLoopIdx  - prefixIdxOffset) * (int64_t)params->singleProcessSInnerSize + (int64_t)firstInnerMargin) * (int64_t)this->MultiHeadKV;
    }

    TEMPLATE_LAYOUT
    __aicore__ inline TYPENAME_BNSD_INT64 GetBmm1TensorBOffset(PFAComputeParam *params,
        int32_t sInnerLoopIdx, int32_t firstInnerMargin) {
        int64_t prefixIdxOffset = 0;
        if constexpr (PFAT::enablePrefix) {
            if (params->isPrefixInnerIter) {
                return this->tensorBPrefixCoreOffset + ((int64_t)sInnerLoopIdx * (int64_t)params->singleProcessSInnerSize + (int64_t)firstInnerMargin) * \
                    (int64_t)this->tilingData->promptAttentionBaseParams.headSize;
            }
            prefixIdxOffset = (actualKVPrefixLen + (int64_t)params->singleProcessSInnerSize - 1) / (int64_t)params->singleProcessSInnerSize;
        }
        return this->tensorBCoreOffset + (((int64_t)sInnerLoopIdx -  prefixIdxOffset) * (int64_t)params->singleProcessSInnerSize + 
            (int64_t)firstInnerMargin) * (int64_t)this->tilingData->promptAttentionBaseParams.headSize;
    }

    TEMPLATE_MASKTYPE
    __aicore__ inline TYPENAME_MASKTYPE_HALF_VOID ElewiseCompute(LocalTensor<computeType>& mmResUb, uint32_t sOuterSize,
        uint32_t sInnerSize, uint32_t maskCopyInCol, bool useMask, event_t &copyIn, uint32_t type)
    {
        uint32_t computeSize = sOuterSize * sInnerSize;
        if (useMask) {
            this->attenMaskUb = this->tempBmm2Queue.template DeQue<U>();
            Muls(this->attenMaskUb, this->attenMaskUb, static_cast<computeType>(-10000.0), computeSize);
            pipe_barrier(PIPE_V);
            Add(mmResUb, mmResUb, this->attenMaskUb, computeSize);
            pipe_barrier(PIPE_V);
            tempBmm2Queue.FreeTensor(this->attenMaskUb);
        }
    }

    TEMPLATE_MASKTYPE
    __aicore__ inline TYPENAME_MASKTYPE_BOOL_VOID ElewiseCompute(LocalTensor<computeType>& mmResUb, uint32_t sOuterSize,
        uint32_t sInnerSize, uint32_t maskCopyInCol, bool useMask, event_t &copyIn, uint32_t type)
    {
        if (useMask) {
            this->attenMaskUb = this->tempBmm2Queue.template DeQue<U>();
            this->attenMaskUb.SetSize(sOuterSize * maskCopyInCol);
            LocalTensor<uint8_t> selectSpace = selectSpaceUb.Get<uint8_t>(this->selectSpaceUbSize);
            computeType scalar;
            if constexpr (PFAT::calcMode == Mode::HighPrecision ||
                IsSameType<T, bfloat16_t>::value) {
                uint32_t tmp = 0xFF7FFFFF;  // fp32最小值
                scalar = *((float*)&tmp);
            } else {
                uint32_t tmp = 0xFBFF;  // fp16最小值
                scalar = *((half*)&tmp);
            }
            SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
            selectWithBytesMaskShapeInfo.firstAxis = sOuterSize;
            selectWithBytesMaskShapeInfo.srcLastAxis = sInnerSize;
            selectWithBytesMaskShapeInfo.maskLastAxis = maskCopyInCol;
            if(type == 0){
                SelectWithBytesMask(mmResUb, mmResUb, scalar, this->attenMaskUb, selectSpace,
                                    selectWithBytesMaskShapeInfo);
            } else if(type == 1) {
                SelectWithBytesMask(mmResUb, scalar, mmResUb, this->attenMaskUb, selectSpace,
                                    selectWithBytesMaskShapeInfo); // swape param 2 and param 3 of SelectWithBytesMask to compute attenMaskPre for band mode
            }
            pipe_barrier(PIPE_V);
            tempBmm2Queue.FreeTensor(this->attenMaskUb);
        }
    }

    TEMPLATE_MASKTYPE
    __aicore__ inline TYPENAME_MASKTYPE_INT8_VOID ElewiseCompute(LocalTensor<T>& mmResUb, uint32_t sOuterSize,
        uint32_t sInnerSize, uint32_t maskCopyInCol, bool useMask, event_t &copyIn, uint32_t type)
    {
        if (useMask) {
            this->attenMaskUb = this->tempBmm2Queue.template DeQue<U>();
            this->attenMaskUb.SetSize(sOuterSize * maskCopyInCol);
            LocalTensor<uint8_t> selectSpace = selectSpaceUb.Get<uint8_t>(this->selectSpaceUbSize);
            computeType scalar;
            if constexpr (PFAT::calcMode == Mode::HighPrecision ||
                IsSameType<T, bfloat16_t>::value) {
                uint32_t tmp = 0xFF7FFFFF;  // fp32最小值
                scalar = *((float*)&tmp);
            } else {
                uint32_t tmp = 0xFBFF;  // fp16最小值
                scalar = *((half*)&tmp);
            }
            SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
            selectWithBytesMaskShapeInfo.firstAxis = sOuterSize;
            selectWithBytesMaskShapeInfo.srcLastAxis = sInnerSize;
            selectWithBytesMaskShapeInfo.maskLastAxis = maskCopyInCol;
            if(type == 0){
                SelectWithBytesMask(mmResUb, mmResUb, scalar, this->attenMaskUb, selectSpace,
                                    selectWithBytesMaskShapeInfo);
            } else if(type == 1) {
                SelectWithBytesMask(mmResUb, scalar, mmResUb, this->attenMaskUb, selectSpace,
                                    selectWithBytesMaskShapeInfo); // swape param 2 and param 3 of SelectWithBytesMask to compute attenMaskPre for band mode
            }
            pipe_barrier(PIPE_V);
            tempBmm2Queue.FreeTensor(this->attenMaskUb);
        }
    }

    __aicore__ inline void ComputePseShiftOffset(int sInnerOffsetDataSize) {
        if (!(this->tailParams->usePseShift)) {
            return;
        }

        this->tailParams->pseShiftOffset = pseShiftCoreOffset + (uint64_t)sInnerOffsetDataSize;
    }

    __aicore__ inline void ComputeAttenMaskOffset(int64_t sInnerOffsetDataSize) {
        int64_t delta;
        if (attentionMaskType == 2 || attentionMaskType == 3 || attentionMaskType == 4) { // 2:leftUp mode of sparseMode, 3:rightdown mode of sparseMode, 4:band mode of sparseMode
            if (attentionMaskType == 2) {
                delta = this->tailParams->sOuterOffset - \
                    sInnerOffsetDataSize + tilingData->promptAttentionBaseParams.nextTokens;
            } else {
                delta = this->tailParams->sOuterOffset - \
                    sInnerOffsetDataSize + this->tailParams->nextTokensPerBatch;
            }

            if (delta < 0) {
                this->tailParams->attenMaskOffset = ((int32_t)singleProcessSOuterSizeWhole + delta) > 0
                    ? (-delta) : singleProcessSOuterSizeWhole;
            }
            else {
                this->tailParams->attenMaskOffset = (((int32_t)this->tailParams->singleProcessSInnerSize - delta) > 0
                    ? delta : this->tailParams->singleProcessSInnerSize) * attentionMaskStride;
            }
        } else {
            this->tailParams->attenMaskOffset = attenMaskCoreOffset + (uint64_t)sInnerOffsetDataSize;
        }
    }

    __aicore__ inline void ComputeAttenMaskOffsetPre(int64_t sInnerOffsetDataSize) {
        if (attentionMaskType == 0 || attentionMaskType == 1) {
            return;
        }
        int64_t delta;
        delta = this->tailParams->sOuterOffset - sInnerOffsetDataSize - this->tailParams->preTokensPerBatch - 1;
        if (delta < 0) {
            this->tailParams->attenMaskOffsetPre = ((int32_t)singleProcessSOuterSizeWhole + delta) > 0
                ? (-delta) : singleProcessSOuterSizeWhole;
        }
        else {
            this->tailParams->attenMaskOffsetPre = (((int32_t)this->tailParams->singleProcessSInnerSize - delta) > 0
                ? delta : this->tailParams->singleProcessSInnerSize) * attentionMaskStride;
        }
    }

    __aicore__ inline void initOffset();

    __aicore__ inline void InitTensorSize(const PromptAttentionSingleCoreTensorSize* tensorSizeTiling);

    __aicore__ inline void GetSingleCoreParam(int sIdx);

    __aicore__ inline void GetSparseParam(int64_t* preTokens, int64_t* nextTokens, int sIdx, PFAComputeParam *&params);

    __aicore__ inline void InitOutputSingleCore();
    __aicore__ inline void InitLseOutputSingleCore();
};

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::PostQuant2PerChannelBF16(LocalTensor<computeType> &bmm2ResUb, LocalTensor<int8_t> &outputQuantRes) {
    if constexpr (IsSameType<computeType, float>::value) {
        LocalTensor<bfloat16_t> quantScale2Ub = quantScale2BF16Ub.Get<bfloat16_t>(perChannelQuantUBSize);
        LocalTensor<bfloat16_t> quantOffset2Ub;
        DataCopy(quantScale2Ub, quantScale2BF16Gm[(uint64_t)this->preHeadParams->batchNOffset * perChannelQuantUBSize], perChannelQuantUBSize);
        if (isQuantOffset2Exist) {
            quantOffset2Ub = quantOffset2BF16Ub.Get<bfloat16_t>(perChannelQuantUBSize);
            DataCopy(quantOffset2Ub, quantOffset2BF16Gm[(uint64_t)this->preHeadParams->batchNOffset * perChannelQuantUBSize], perChannelQuantUBSize);
        }
        auto quantParamCast = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(quantParamCast);
        WaitFlag<HardEvent::MTE2_V>(quantParamCast);

        LocalTensor<float> quantScale2UbFloat = quantScale2FloatUb.Get<float>(perChannelQuantUBSize);
        LocalTensor<float> quantOffset2UbFloat;
        Cast(quantScale2UbFloat, quantScale2Ub, RoundMode::CAST_NONE, quantScale2Ub.GetSize());
        if (isQuantOffset2Exist) {
            quantOffset2UbFloat = quantOffset2FloatUb.Get<float>(perChannelQuantUBSize);
            Cast(quantOffset2UbFloat, quantOffset2Ub, RoundMode::CAST_NONE, quantOffset2Ub.GetSize());
            pipe_barrier(PIPE_V);
            AscendQuant(outputQuantRes, bmm2ResUb, quantScale2UbFloat, quantOffset2UbFloat);
        } else {
            pipe_barrier(PIPE_V);
            AscendQuant(outputQuantRes, bmm2ResUb, quantScale2UbFloat, static_cast<float>(0));
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::PostQuant2PerChannelFP32(LocalTensor<computeType> &bmm2ResUb, LocalTensor<int8_t> &outputQuantRes) {
    LocalTensor<float> quantScale2UbTmp = quantScale2FloatUb.Get<float>(perChannelQuantUBSize);
    LocalTensor<computeType> quantScale2Ub = quantScale2UbTmp.template ReinterpretCast<computeType>();
    LocalTensor<float> quantOffset2UbTmp;
    LocalTensor<computeType> quantOffset2Ub;
    if (isQuantOffset2Exist) {
        quantOffset2UbTmp = quantOffset2FloatUb.Get<float>(perChannelQuantUBSize);
        quantOffset2Ub = quantOffset2UbTmp.template ReinterpretCast<computeType>();
    }

    DataCopy(quantScale2UbTmp, quantScale2FP32Gm[(uint64_t)this->preHeadParams->batchNOffset * perChannelQuantUBSize], perChannelQuantUBSize);
    if (isQuantOffset2Exist) {
        DataCopy(quantOffset2UbTmp, quantOffset2FP32Gm[(uint64_t)this->preHeadParams->batchNOffset * perChannelQuantUBSize], perChannelQuantUBSize);
    }
    auto quantParamCast = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(quantParamCast);
    WaitFlag<HardEvent::MTE2_V>(quantParamCast);

    if constexpr (IsSameType<computeType, half>::value) {
        Cast(quantScale2Ub, quantScale2UbTmp, RoundMode::CAST_ROUND, perChannelQuantUBSize);
        if (isQuantOffset2Exist) {
            Cast(quantOffset2Ub, quantOffset2UbTmp, RoundMode::CAST_ROUND, perChannelQuantUBSize);
        }
    }

    pipe_barrier(PIPE_V);
    if (isQuantOffset2Exist) {
        AscendQuant(outputQuantRes, bmm2ResUb, quantScale2Ub, quantOffset2Ub, perChannelQuantUBSize, perChannelQuantUBSize, bmm2ResUb.GetSize());
    } else {
        AscendQuant(outputQuantRes, bmm2ResUb, quantScale2Ub, static_cast<computeType>(0), perChannelQuantUBSize, bmm2ResUb.GetSize());
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::CalPseShiftOffset(int sIdx) {
    if (!(this->tailParams->usePseShift)) {
        return;
    }
    int64_t queryLeftpaddingSize = GetQueryLeftPaddingSize(sIdx);
    int64_t kvLeftPaddingSize = GetKVLeftPaddingSize(sIdx);
    uint64_t pseShiftBatchOffset = 0;
    uint64_t pseShiftN = (uint64_t)tilingData->promptAttentionBaseParams.headNumSize;
    uint64_t pseShiftS1 = (uint64_t)tilingData->promptAttentionBaseParams.pseShiftS1Size;
    uint64_t pseShiftS2 = (uint64_t)tilingData->promptAttentionBaseParams.pseShiftS2Size;

    if (pseShiftBatch != 1) {
        pseShiftBatchOffset = (uint64_t)sIdx * pseShiftN * pseShiftS1 * pseShiftS2;
    }

    pseShiftCoreOffset = pseShiftBatchOffset + (uint64_t)this->tailParams->batchNOffset * pseShiftS1 * pseShiftS2 +
                         ((uint64_t)this->tailParams->sOuterOffset + queryLeftpaddingSize) * pseShiftS2 + kvLeftPaddingSize;
}

// quant: add quant functions
template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::QuantCompute(LocalTensor<int8_t> quantResUb, LocalTensor<computeType> mmResUb,
                                                                                    float scale, float offset, uint32_t computeSize) {
    pipe_barrier(PIPE_V);
    AscendQuant(quantResUb, mmResUb, scale, offset, computeSize);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::InitQuant(__gm__ uint8_t* deq_scale1,
                                                             __gm__ uint8_t* scale1, __gm__ uint8_t* deq_scale2,
                                                             __gm__ uint8_t* scale2, __gm__ uint8_t* offset2) {
    if (deq_scale1 != nullptr) {
        if(tilingData->promptAttentionBaseParams.deqScale2Flag == 1){
            deqScale1Fp32Gm.SetGlobalBuffer((__gm__ uint32_t*)deq_scale1);
            dequantScale1 = deqScale1Fp32Gm(0);
        } else {
            dequantScale1 = *(reinterpret_cast<__gm__ uint64_t*>(deq_scale1));
        }
    }
    if (scale1 != nullptr) { quantScale1 = *(reinterpret_cast<__gm__ float*>(scale1));}
    if (deq_scale2 != nullptr) {
        if(tilingData->promptAttentionBaseParams.deqScaleFlag == 1){
            deqScale2Fp32Gm.SetGlobalBuffer((__gm__ uint32_t*)deq_scale2);
            dequantScale2 = deqScale2Fp32Gm(0);
        } else {
            dequantScale2 = *(reinterpret_cast<__gm__ uint64_t*>(deq_scale2));
        }
    }
    isQuant2PerChn = tilingData->promptAttentionBaseParams.isQuant2Perchannel == 0 ? false : true;
    isQuant2BF16 = tilingData->promptAttentionBaseParams.isQuant2BF16 == 0 ? false : true;
    isQuantOffset2Exist = offset2 == nullptr ? false : true;

    // per-tensor是否要支持scale2和offset输入bf16，当前的修改把这个功能删掉了，有功能回退
    perChannelQuantUBSize = this->tilingData->promptAttentionBaseParams.headSize;
    if (scale2 != nullptr) {
        if (isQuant2PerChn) {       // per-channel, scale2为tensor
            if (isQuant2BF16) {     // scale2类型为bf16
                quantScale2BF16Gm.SetGlobalBuffer((__gm__ bfloat16_t*)(scale2));
                pipe->InitBuffer(quantScale2BF16Ub, perChannelQuantUBSize * sizeof(bfloat16_t));
                pipe->InitBuffer(quantScale2FloatUb, perChannelQuantUBSize * sizeof(float));
            } else {                // scale2类型为fp32
                quantScale2FP32Gm.SetGlobalBuffer((__gm__ float*)(scale2));
                pipe->InitBuffer(quantScale2FloatUb, perChannelQuantUBSize * sizeof(float));
            }
        } else {                    // per-tensor, scale2为scalar
            if (isQuant2BF16) {     // scale2类型为bf16
                quantScale2BF16Gm.SetGlobalBuffer((__gm__ bfloat16_t*)(scale2));
                quantScale2 = ToFloat(quantScale2BF16Gm.GetValue(0));
            } else {                // scale2类型为fp32
                quantScale2 = *(reinterpret_cast<__gm__ float*>(scale2));
            }
        }
    }

    if (offset2 != nullptr) {
        if (isQuant2PerChn) {       // per-channel, offset2为tensor
            if (isQuant2BF16) {     // offset2类型为bf16
                quantOffset2BF16Gm.SetGlobalBuffer((__gm__ bfloat16_t*)(offset2));
                pipe->InitBuffer(quantOffset2BF16Ub, perChannelQuantUBSize * sizeof(bfloat16_t));
                pipe->InitBuffer(quantOffset2FloatUb, perChannelQuantUBSize * sizeof(float));
            } else {                // offset2类型为fp32
                quantOffset2FP32Gm.SetGlobalBuffer((__gm__ float*)(offset2));
                pipe->InitBuffer(quantOffset2FloatUb, perChannelQuantUBSize * sizeof(float));
            }
        } else {                    // per-tensor, offset2为scalar
            if (isQuant2BF16) {
                quantOffset2BF16Gm.SetGlobalBuffer((__gm__ bfloat16_t*)(offset2));
                quantOffset2 = ToFloat(quantOffset2BF16Gm.GetValue(0));
            } else {
                quantOffset2 = *(reinterpret_cast<__gm__ float*>(offset2));
            }
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::InitKvAntiquant(__gm__ uint8_t* antiq_scale, __gm__ uint8_t* antiq_offset) {
    pipe->InitBuffer(kvAntiquantSrcQueue, 1, tilingData->promptAttentionTensorSizeRect.kvAntiquantUbSize * sizeof(int8_t));
    pipe->InitBuffer(kvAntiquantDstQueue, 1, tilingData->promptAttentionTensorSizeRect.kvAntiquantUbSize * sizeof(T));
    pipe->InitBuffer(antiquantScaleUb, tilingData->promptAttentionBaseParams.alignedHeadSize * sizeof(T));
    pipe->InitBuffer(antiquantOffsetUb, tilingData->promptAttentionBaseParams.alignedHeadSize * sizeof(T));

    antiquantScaleGm.SetGlobalBuffer((__gm__ T*)antiq_scale);
    if (antiq_offset != nullptr) {
        antiquantOffsetGm.SetGlobalBuffer((__gm__ T*)antiq_offset);
    } else {
        isAntiquantSymmetric = true;
    }
    if (!tilingData->promptAttentionBaseParams.isAntiPerchannel) {
        keyAntiquantScale = antiquantScaleGm(0);
        valueAntiquantScale = antiquantScaleGm(1);
        if (antiq_offset != nullptr) {
            keyAntiquantOffset = antiquantOffsetGm(0);
            valueAntiquantOffset = antiquantOffsetGm(1);
        } else {
            keyAntiquantOffset = 0;
            valueAntiquantOffset = 0;
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                        __gm__ uint8_t* value, __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                        __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                        const PromptFlashAttentionTilingData* __restrict tiling,
                                        __gm__ uint8_t* gmTiling, TPipe* tPipe) {
    tmp_block_idx = GetBlockIdx();
    // init global buffer
    tilingData = tiling;
    key_ptr = key;
    value_ptr = value;

    // 针对小 B*N 进行跳核优化
    if (tilingData->promptAttentionSingleCoreParams.actualCoreNums <= (GetBlockNum() * GetTaskRation() / 2 + 1)) {
        if (tmp_block_idx & 0x1) {
            tmp_block_idx = (tmp_block_idx + GetBlockNum() * GetTaskRation()) / 2;
        } else {
            tmp_block_idx = tmp_block_idx / 2;
        }
    }

    queryGm.SetGlobalBuffer((__gm__ T*)query);
    attentionOutGm.SetGlobalBuffer((__gm__ O*)attentionOut);
    attentionOutInitGm.SetGlobalBuffer((__gm__ half*)attentionOut);
    softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse);
    workspaceGm.SetGlobalBuffer((__gm__ computeType*)workspace);

    pipe = tPipe;
    typeByteNum = tilingData->promptAttentionBaseParams.typeByteNum;
    outputTypeByteNum = tilingData->promptAttentionBaseParams.outputTypeByteNum;
    softmaxTypeByteNum = tilingData->promptAttentionBaseParams.softmaxTypeByteNum;
    headNumRatio = tilingData->promptAttentionBaseParams.headNumRatio;
    maskDataType = tilingData->promptAttentionBaseParams.attenMaskElemType;
    maskTypeByteNum = tilingData->promptAttentionBaseParams.maskTypeByteNum;
    attenMaskBatch = tilingData->promptAttentionSingleCoreParams.attenMaskBatch;
    pseShiftTypeByteNum = tilingData->promptAttentionBaseParams.pseShiftTypeByteNum;
    pseShiftBatch = tilingData->promptAttentionSingleCoreParams.pseShiftBatch;
    isKvContinuous = tilingData->promptAttentionBaseParams.isKvContinuous;
    fromFused = tilingData->promptAttentionBaseParams.fromFused;

    if (fromFused) {
        ListTensorDesc keyListTensorDescInit((__gm__ void*)key_ptr);
        ListTensorDesc valueListTensorDescInit((__gm__ void*)value_ptr);
        currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
            blocktable_ptr = blocktable;
            mm.SetUserDefInfo(reinterpret_cast<uint64_t>(gmTiling));
            bmm2.SetUserDefInfo(reinterpret_cast<uint64_t>(gmTiling));
        }
        keyGm.SetGlobalBuffer((__gm__ KV_T*)currentKey);
        valueGm.SetGlobalBuffer((__gm__ KV_T*)currentValue);
    } else {
        keyGm.SetGlobalBuffer((__gm__ KV_T*)key);
        valueGm.SetGlobalBuffer((__gm__ KV_T*)value);
    }
    initOffset();

    isActualLenDimsNull = true;
    isActualLenDimsKVNull = true;
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsNull) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsNull = false;
    }
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsKVNull) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsKVNull = false;
    }

    if (tilingData->promptAttentionBaseParams.isQHasLeftPadding) {
        queryPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t*)queryPaddingSize);
    }

    if (tilingData->promptAttentionBaseParams.isKVHasLeftPadding) {
        kvPaddingSizeGm.SetGlobalBuffer((__gm__ int64_t*)kvPaddingSize);
    }
    if constexpr (PFAT::enablePrefix) {
        keySharedPrefixGm.SetGlobalBuffer((__gm__ KV_T*)keySharedPrefix);
        valueSharedPrefixGm.SetGlobalBuffer((__gm__ KV_T*)valueSharedPrefix);
        actualSharedPrefixLenGm.SetGlobalBuffer((__gm__ int64_t*)actualSharedPrefixLen);
    }

    uint32_t preAccumSOuter = 0;
    uint32_t h = tilingData->promptAttentionBaseParams.headNumSize * tilingData->promptAttentionBaseParams.headSize;
    uint32_t s = tilingData->promptAttentionBaseParams.seqSize;
    if constexpr ((PFAT::calcMode != Mode::HighPrecision) && 
                  (IsSameType<T, half>::value || IsSameType<T, int8_t>::value)) {
        this->negativeScalar = NEGATIVE_MIN_VAULE_FP16;
    }
    uint64_t maskSize = (tilingData->promptAttentionTensorSizeRect.attenMaskUbSize) * sizeof(U);
    maskBmm2ShareSize = (tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize) * sizeof(computeType);
    if (maskBmm2ShareSize < maskSize) {
        maskBmm2ShareSize = maskSize;
    }

    if ((pseShift != NULL) && (tilingData->promptAttentionBaseParams.usePseShift == 1)) {
        uint32_t pseShiftSize = (tilingData->promptAttentionTensorSizeRect.pseShiftUbSize) * sizeof(pseShiftType);
        if (maskBmm2ShareSize < pseShiftSize) {
            maskBmm2ShareSize = pseShiftSize;
        }
    }

    pipe->InitBuffer(softmaxOutQueue, 1, 2 * tilingData->promptAttentionTensorSizeRect.softmaxMaxSize * sizeof(float));
    pipe->InitBuffer(tempBmm2Ub, tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize * sizeof(computeType));
    pipe->InitBuffer(softmaxExpUb_, tilingData->promptAttentionTensorSizeRect.softmaxExpSize * sizeof(computeType));
    pipe->InitBuffer(tempBmm2Queue, 1, maskBmm2ShareSize);
    pipe->InitBuffer(Bmm1Queue, 2, tilingData->promptAttentionTensorSizeRect.mmResUbSize * sizeof(computeType));
    if (tilingData->promptAttentionTensorSizeRect.selectSpaceUbSize != 0) {
        pipe->InitBuffer(selectSpaceUb, tilingData->promptAttentionTensorSizeRect.selectSpaceUbSize);
    }
    if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
        pipe->InitBuffer(PABmm1UB, 64);  // dcci刷新64B
        pipe->InitBuffer(PABmm2UB, 64);  // dcci刷新64B
    }
    // 使用队列预取参数，每次计算外在tail入队一个新的计算参数，计算时使用队列head的参数，计算后出队head
    tailId = 0;
    headId = 0;
    queSize = 0;
    tailParams = &pfaParamsQueue[tailId];
    headParams = &pfaParamsQueue[headId];
    preHeadParams = &pfaParamsQueue[headId];
    isGlobalFirstCompute = true;
    mm1SingleCoreNPrev = 0;
    mm2MStridePrev = 0;
    mm2KaStridePrev = 0;

    tailParams->gmPingpong = 0;
    tailParams->useMask = false;
    tailParams->usePseShift = false;

    attentionMaskType = tilingData->promptAttentionBaseParams.sparseMode;
    if ((attenMask != NULL) && (tilingData->promptAttentionBaseParams.useMask == 1)) {
        tailParams->useMask = true;
        attenMaskGm.SetGlobalBuffer((__gm__ U*)attenMask);
        attentionMaskStride = tilingData->promptAttentionBaseParams.maskKVsSize;
    }

    if ((pseShift != NULL) && (tilingData->promptAttentionBaseParams.usePseShift == 1)) {
        tailParams->usePseShift = true;
        pseShiftGm.SetGlobalBuffer((__gm__ pseShiftType*)pseShift);
        pseShiftStride = tilingData->promptAttentionBaseParams.pseShiftS2Size;

        if constexpr (AscendC::IsSameType<pseShiftCastType, float>::value) {
            pipe->InitBuffer(pseShiftCastUb,
                             (tilingData->promptAttentionTensorSizeRect.pseShiftUbSize) * sizeof(float));
        }
    }

    softmaxExpUb = softmaxExpUb_.Get<computeType>(tilingData->promptAttentionTensorSizeRect.softmaxExpSize);

    if (tilingData->promptAttentionInitOutputParams.needInit == 1) {
        InitOutputSingleCore();
    }
    InitLseOutputSingleCore();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::InitLseOutputSingleCore() {
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    if (this->tilingData->promptAttentionBaseParams.isSoftMaxLseEnable == true) {
        int64_t coreNum = GetBlockNum() * GetTaskRation();
        int64_t singleCoreLseSize = initParams.totalSoftMaxLseOutputSize / coreNum;
        if (tmp_block_idx == coreNum - 1) {
            singleCoreLseSize += initParams.totalSoftMaxLseOutputSize % coreNum;
        }
        InitOutput<float>(softmaxLseGm[tmp_block_idx * (initParams.totalSoftMaxLseOutputSize / coreNum)], singleCoreLseSize, 3e+99);
        SyncAll();
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::InitOutputSingleCore()
{
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    int64_t tailSize = (int64_t)initParams.totalOutputSize - tmp_block_idx * (int64_t)initParams.singleCoreSize;
    if (tailSize > 0) {
        int64_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
        if constexpr (IsSameType<O, int8_t>::value) {
            // InitOutput instr do not support int8/uint8, we Use half replace int8/uint8
            // because D is 32bytes aligned, it will has no remainder even if we Converted into half
            InitOutput<half>(attentionOutInitGm[tmp_block_idx * (int64_t)initParams.singleCoreSize / 2], singleInitOutputSize / 2, 0);
        } else {
            InitOutput<O>(attentionOutGm[tmp_block_idx * (int64_t)initParams.singleCoreSize], singleInitOutputSize, 0);
        }
    }
    SyncAll();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::initOffset() {
    offsetSS = tilingData->promptAttentionBaseParams.seqSize * tilingData->promptAttentionBaseParams.seqSize;
    offsetSH = tilingData->promptAttentionBaseParams.seqSize * tilingData->promptAttentionBaseParams.headSize;
    offsetSTypeNum = tilingData->promptAttentionBaseParams.seqSize * typeByteNum;
    offsetNSTypeNum = tilingData->promptAttentionBaseParams.headNumSize * offsetSTypeNum;
    offsetNSS = tilingData->promptAttentionBaseParams.headNumSize * offsetSS;
    offsetNSH = tilingData->promptAttentionBaseParams.headNumSize * offsetSH;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::InitTensorSize(
                const PromptAttentionSingleCoreTensorSize* tensorSizeTiling) {
    mmResUbSize = tensorSizeTiling->mmResUbSize;
    attenMaskUbSize = tensorSizeTiling->attenMaskUbSize;
    pseShiftUbSize = tensorSizeTiling->pseShiftUbSize;
    maskSize = tensorSizeTiling->maskSize;
    softmaxMaxSize = tensorSizeTiling->softmaxMaxSize;
    softmaxSumSize = tensorSizeTiling->softmaxSumSize;
    softmaxExpSize = tensorSizeTiling->softmaxExpSize;
    spmTmpSize = tensorSizeTiling->spmTmpSize;
    scmTmpSize = tensorSizeTiling->scmTmpSize;
    bmm2ResUbSize = tensorSizeTiling->bmm2ResUbSize;
    tmpMMResBmm2PreUbSize = tensorSizeTiling->tmpMMResBmm2PreUbSize;
    tmpSoftmaxBmm2UbSize = tensorSizeTiling->tmpSoftmaxBmm2UbSize;
    selectSpaceUbSize = tensorSizeTiling->selectSpaceUbSize;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::SoftmaxBasicComputeFirstNoTail(LocalTensor<computeType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, uint32_t souterSize) {
    LocalTensor<computeType> null;
    SoftMaxShapeInfo softmaxShapeInfo;
    if (this->headParams->isInnerTail) {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerBmmTail)
        };
    } else {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow)
        };
    }
    SoftmaxFlashV2<computeType, false, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                         mmResUb, null, softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<computeType, float>(mmResUb,
            softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
        this->isSoftmaxLseNeedUpdate = true;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::SoftmaxBasicComputeNoTail(LocalTensor<computeType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<computeType>& softmaxExpUb, uint32_t souterSize) {
    SoftMaxShapeInfo softmaxShapeInfo;
    if (this->headParams->isInnerTail) {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerBmmTail)
        };
    } else {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow)
        };
    }
    SoftmaxFlashV2<computeType, true, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                        mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<computeType, float>(mmResUb,
            softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
        this->isSoftmaxLseNeedUpdate = true;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::SoftmaxComputeFirstTail(LocalTensor<computeType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, uint32_t souterSize) {
    LocalTensor<computeType> null;
    SoftMaxShapeInfo softmaxShapeInfo;
    if (this->headParams->isInnerTail) {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerBmmTail)
        };
    } else {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow)
        };
    }
    SoftmaxFlashV2<computeType, false, true, false>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                          mmResUb, null, softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<computeType, float>(mmResUb,
            softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
        this->isSoftmaxLseNeedUpdate = true;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::SoftmaxComputeTail(LocalTensor<computeType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<computeType>& softmaxExpUb, uint32_t souterSize) {
    SoftMaxShapeInfo softmaxShapeInfo;
    if (this->headParams->isInnerTail) {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerBmmTail)
        };
    } else {
        softmaxShapeInfo = {
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow),
            static_cast<uint32_t>(souterSize),
            static_cast<uint32_t>(this->headParams->singleProcessSInnerSizeNow)
        };
    }
    SoftmaxFlashV2<computeType, true, true, false>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                         mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<computeType, float>(mmResUb,
            softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
        this->isSoftmaxLseNeedUpdate = true;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::Bmm2UpdateDivNoTail(LocalTensor<computeType>& bmm2ResPreUb,
                                            LocalTensor<float>& softmaxSumUb) {
    PFAComputeParam *params = this->preHeadParams;
    int32_t headLoop = (tilingData->promptAttentionBaseParams.headSize + softmaxTypeByteNum - 1) / softmaxTypeByteNum;
    constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(computeType);

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = headLoop;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = headLoop;

    int32_t loop = tilingData->promptAttentionBaseParams.headSize / REPEAT_DATA_NUM;
    int32_t remain = tilingData->promptAttentionBaseParams.headSize % REPEAT_DATA_NUM;
    if constexpr (IsSameType<computeType, half>::value) {
        constexpr int32_t FP32_BLOCK_NUM = 8;
        constexpr int32_t FP32_MASK_NUM = 64;
        CopyRepeatParams copyRepeatParams{2, 1, 16, 8};
        int32_t calcSize = params->singleProcessSOuterSize * FP32_BLOCK_NUM;
        LocalTensor<float> tmpBuffer = tempBmm2Queue.template AllocTensor<float>();
        LocalTensor<half> tmpHalfBuffer = tmpBuffer[calcSize * 2].template ReinterpretCast<half>();

        int32_t repeat = (calcSize + FP32_MASK_NUM - 1) / FP32_MASK_NUM;
        Copy(tmpBuffer, softmaxSumUb, FP32_MASK_NUM, repeat, copyRepeatParams);
        Copy(tmpBuffer[FP32_BLOCK_NUM], softmaxSumUb, FP32_MASK_NUM, repeat, copyRepeatParams);
        pipe_barrier(PIPE_V);
        Cast(tmpHalfBuffer, tmpBuffer, RoundMode::CAST_ROUND, calcSize * 2);
        pipe_barrier(PIPE_V);

        for (int i = 0; i < loop; i++) {
            Div(bmm2ResPreUb[i * REPEAT_DATA_NUM], bmm2ResPreUb[i * REPEAT_DATA_NUM], tmpHalfBuffer,
                REPEAT_DATA_NUM, params->singleProcessSOuterSize, repeatParams);
        }
        if (remain) {
            Div(bmm2ResPreUb[loop * REPEAT_DATA_NUM], bmm2ResPreUb[loop * REPEAT_DATA_NUM], tmpHalfBuffer,
                remain, params->singleProcessSOuterSize, repeatParams);
        }
        tempBmm2Queue.FreeTensor(tmpBuffer);
    } else {
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResPreUb[i * REPEAT_DATA_NUM], bmm2ResPreUb[i * REPEAT_DATA_NUM], softmaxSumUb,
                REPEAT_DATA_NUM, params->singleProcessSOuterSize, repeatParams);
        }
        if (remain) {
            Div(bmm2ResPreUb[loop * REPEAT_DATA_NUM], bmm2ResPreUb[loop * REPEAT_DATA_NUM], softmaxSumUb,
                remain, params->singleProcessSOuterSize, repeatParams);
        }
    }
}


template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::UpdateVmul(LocalTensor<computeType>& softmaxExpUb) {
    LocalTensor<computeType> bmm2ResPreUb = tempBmm2Ub.Get<computeType>(bmm2ResUbSize);

    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = (
        tilingData->promptAttentionBaseParams.headSize + softmaxTypeByteNum - 1) / softmaxTypeByteNum;
    repeatParams.dstRepStride = (
        tilingData->promptAttentionBaseParams.headSize + softmaxTypeByteNum - 1) / softmaxTypeByteNum;

    // only support singleProcessSOuterSize <=255, headsize 32B align
    int32_t numOneRep = 256 / sizeof(computeType);
    int32_t loop = tilingData->promptAttentionBaseParams.headSize / numOneRep;
    int32_t remain =  tilingData->promptAttentionBaseParams.headSize % numOneRep;

    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResPreUb[i * numOneRep], softmaxExpUb, bmm2ResPreUb[i * numOneRep],
            numOneRep, this->headParams->singleProcessSOuterSize, repeatParams);
    }
    if (remain) {
        Mul(bmm2ResPreUb[loop * numOneRep], softmaxExpUb, bmm2ResPreUb[loop * numOneRep],
            remain, this->headParams->singleProcessSOuterSize, repeatParams);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::Bmm2UpdateAdd(LocalTensor<computeType>& bmm2ResUb) {
    LocalTensor<computeType> bmm2ResPreUb = tempBmm2Ub.Get<computeType>(bmm2ResUbSize);
    Add(bmm2ResPreUb, bmm2ResUb, bmm2ResPreUb, bmm2ResUbSize);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::GetSingleCoreParam(int sIdx) {
    int64_t actualSeqLengthPerBatch = 0;
    int64_t actualSeqLengthKVPerBatch = 0;
    actualSeqLengthPerBatch = isActualLenDimsNull ? tilingData->promptAttentionBaseParams.seqSize :
                              actualSeqLengthsGm.GetValue(sIdx);
    if (isKvContinuous == 1) {
        actualSeqLengthKVPerBatch = isActualLenDimsKVNull ? tilingData->promptAttentionBaseParams.seqInnerSize :
                                    actualSeqLengthsKVGm.GetValue(sIdx);
    } else {
        actualSeqLengthKVPerBatch = this->isActualLenDimsKVNull ? s2InCurrentBatch : this->actualSeqLengthsKVGm.GetValue(sIdx);
    }

    this->tailParams->singleProcessSInnerSize = tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
    singleProcessSOuterSizeWhole = tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
    MultiHeadQ = tilingData->promptAttentionBaseParams.headSize * tilingData->promptAttentionBaseParams.headNumSize;
    MultiHeadKV = MultiHeadQ / headNumRatio;

    if (attentionMaskType != 4) {
        this->tailParams->preTokensPerBatch = (int64_t)tilingData->promptAttentionBaseParams.preTokens;
        this->tailParams->nextTokensPerBatch = (int64_t)tilingData->promptAttentionBaseParams.nextTokens;
        if (isKvContinuous == 1) {
            actualSeqLengthPerBatch = ((int64_t)actualSeqLengthPerBatch >
                                    (int64_t)actualSeqLengthKVPerBatch + actualKVPrefixLen +
                                    (int64_t)tilingData->promptAttentionBaseParams.preTokens) && (attentionMaskType != 4)?
                                    (int64_t)actualSeqLengthKVPerBatch + actualKVPrefixLen + (int64_t)tilingData->promptAttentionBaseParams.preTokens :
                                    actualSeqLengthPerBatch;
        } else {
            actualSeqLengthPerBatch = ((int64_t)actualSeqLengthPerBatch >
                                    (int64_t)s2InCurrentBatch + (int64_t)tilingData->promptAttentionBaseParams.preTokens) && (attentionMaskType != 4)?
                                    (int64_t)s2InCurrentBatch + (int64_t)tilingData->promptAttentionBaseParams.preTokens :
                                    actualSeqLengthPerBatch;
        }
    } else {
        this->tailParams->preTokensPerBatch = (int64_t)tilingData->promptAttentionBaseParams.preTokens - actualSeqLengthKVPerBatch - actualKVPrefixLen + actualSeqLengthPerBatch;
        this->tailParams->nextTokensPerBatch = (int64_t)tilingData->promptAttentionBaseParams.nextTokens + actualSeqLengthKVPerBatch + actualKVPrefixLen - actualSeqLengthPerBatch;
        if (isKvContinuous == 1) {
            actualSeqLengthPerBatch = ((int64_t)actualSeqLengthPerBatch >
                                    (int64_t)actualSeqLengthKVPerBatch + actualKVPrefixLen +
                                    (int64_t)this->tailParams->preTokensPerBatch) ?
                                    (int64_t)actualSeqLengthKVPerBatch + actualKVPrefixLen + (int64_t)this->tailParams->preTokensPerBatch :
                                    actualSeqLengthPerBatch;
        } else {
            actualSeqLengthPerBatch = ((int64_t)actualSeqLengthPerBatch >
                                    (int64_t)s2InCurrentBatch + (int64_t)this->tailParams->preTokensPerBatch) ?
                                    (int64_t)s2InCurrentBatch + (int64_t)this->tailParams->preTokensPerBatch :
                                    actualSeqLengthPerBatch;
        }
    }

    singleProcessSOuterSizeTail = (actualSeqLengthPerBatch % singleProcessSOuterSizeWhole != 0) ?
                                   actualSeqLengthPerBatch % singleProcessSOuterSizeWhole : singleProcessSOuterSizeWhole;
    this->tailParams->unalignSInner = (actualSeqLengthKVPerBatch % this->tailParams->singleProcessSInnerSize != 0) ?
                     actualSeqLengthKVPerBatch % this->tailParams->singleProcessSInnerSize : this->tailParams->singleProcessSInnerSize;
    uint32_t maxInnerLoopPromptTimes = (actualSeqLengthKVPerBatch + this->tailParams->singleProcessSInnerSize - 1) / this->tailParams->singleProcessSInnerSize;    this->tailParams->singleProcessSInnerSizeTail = \
        (this->tailParams->unalignSInner + typeByteNum - 1) / typeByteNum * typeByteNum;
    this->tailParams->maskInnerTailAlign = \
        (this->tailParams->unalignSInner + maskTypeByteNum - 1) / maskTypeByteNum * maskTypeByteNum;
    this->tailParams->padSize = this->tailParams->maskInnerTailAlign - this->tailParams->unalignSInner;

    if (pseShiftTypeByteNum != 0) {
        this->tailParams->pseShiftInnerTailAlign = (this->tailParams->unalignSInner + pseShiftTypeByteNum - 1) /
                                                pseShiftTypeByteNum * pseShiftTypeByteNum;
        this->tailParams->pseShiftPadSize = this->tailParams->pseShiftInnerTailAlign - this->tailParams->unalignSInner;
    }

    if constexpr (PFAT::enablePrefix) {
        this->tailParams->unalignSInnerPrefix = (actualKVPrefixLen % this->tailParams->singleProcessSInnerSize != 0) ?
                        actualKVPrefixLen % this->tailParams->singleProcessSInnerSize : this->tailParams->singleProcessSInnerSize;
        maxInnerLoopPrefixTimes = (actualKVPrefixLen + this->tailParams->singleProcessSInnerSize - 1) / this->tailParams->singleProcessSInnerSize;
        this->tailParams->singleProcessSInnerPrefixSizeTail = \
            (this->tailParams->unalignSInnerPrefix + typeByteNum - 1) / typeByteNum * typeByteNum;
        this->tailParams->maskInnerPrefixTailAlign = \
            (this->tailParams->unalignSInnerPrefix + maskTypeByteNum - 1) / maskTypeByteNum * maskTypeByteNum;
        this->tailParams->padPrefixSize = this->tailParams->maskInnerPrefixTailAlign - this->tailParams->unalignSInnerPrefix;
        if (pseShiftTypeByteNum != 0) {
            this->tailParams->pseShiftInnerPrefixTailAlign = (this->tailParams->unalignSInnerPrefix + pseShiftTypeByteNum - 1) /
                                                    pseShiftTypeByteNum * pseShiftTypeByteNum;
            this->tailParams->pseShiftPadPrefixSize = this->tailParams->pseShiftInnerPrefixTailAlign - this->tailParams->unalignSInnerPrefix;
        }
    }
    maxInnerLoopTimes = maxInnerLoopPromptTimes + maxInnerLoopPrefixTimes;
    InitTensorSize(&tilingData->promptAttentionTensorSizeRect);
    transposeTilingData = tilingData->transposeTilingDataRect;
    softmaxTilingData = tilingData->softmaxTilingDataRect;
    softmaxFlashTilingData = tilingData->softmaxFlashTilingDataRect;

    this->tailParams->actualSeqLengthPerBatch = actualSeqLengthPerBatch;
    this->tailParams->actualSeqLengthKVPerBatch = actualSeqLengthKVPerBatch;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910Base<PFAT>::GetSparseParam(int64_t* preTokens, int64_t* nextTokens, int sIdx, PFAComputeParam *&params) {
    if (attentionMaskType == 3) {
        *preTokens = 214748647;
        *nextTokens = params->actualSeqLengthKVPerBatch + actualKVPrefixLen - params->actualSeqLengthPerBatch;
    }
    if (attentionMaskType == 4) {
        int64_t actualSeqLengthKVPerBatchUser;
        int64_t actualSeqLengthPerBatchUser = isActualLenDimsNull ? tilingData->promptAttentionBaseParams.seqSize :
                                              actualSeqLengthsGm.GetValue(sIdx);
        if (isKvContinuous == 1) {
            actualSeqLengthKVPerBatchUser = isActualLenDimsKVNull ? tilingData->promptAttentionBaseParams.seqInnerSize :
                                            actualSeqLengthsKVGm.GetValue(sIdx);
        } else {
            actualSeqLengthKVPerBatchUser = this->isActualLenDimsKVNull ? s2InCurrentBatch : this->actualSeqLengthsKVGm.GetValue(sIdx);
        }
        *preTokens = (int64_t)tilingData->promptAttentionBaseParams.preTokens - actualSeqLengthKVPerBatchUser - actualKVPrefixLen + actualSeqLengthPerBatchUser;
        *nextTokens = (int64_t)tilingData->promptAttentionBaseParams.nextTokens + actualSeqLengthKVPerBatchUser + actualKVPrefixLen - actualSeqLengthPerBatchUser;
    }
    params->preTokensPerBatch = *preTokens;
    params->nextTokensPerBatch = *nextTokens;
}

#endif  // PROMPT_FLASH_ATTENTION_S1S2_BNS1_X910_BASE_H
