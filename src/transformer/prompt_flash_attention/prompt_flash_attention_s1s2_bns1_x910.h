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
 * \file prompt_flash_attention_s1s2_bns1_x910.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_S1S2_BNS1_X910_H
#define PROMPT_FLASH_ATTENTION_S1S2_BNS1_X910_H

#include "prompt_flash_attention_s1s2_bns1_x910_base.h"

using namespace matmul;
template<typename PFAT>
class PromptFlashAttentionS1s2Bns1X910 : public PromptFlashAttentionS1s2Bns1X910Base<PFAT> {
public:
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using U = typename PFAT::maskType;
    using O = typename PFAT::outputType;
    using computeType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::softmaxType;
    using pseShiftType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::pseShiftType;
    using pseShiftCastType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::pseShiftCastType;

    __aicore__ inline PromptFlashAttentionS1s2Bns1X910() {};
    __aicore__ inline void Process();

protected:
    __aicore__ inline void AllocGlobalResources();

    __aicore__ inline void FreeGlobalResources();

    __aicore__ inline void PseOrMaskCopyIn(uint64_t offset, uint32_t souterSize,
        bool isInnerTail, uint32_t alignSInner, uint32_t unalignSInner, uint32_t padSize, bool isMask);

    __aicore__ inline void SparseBandElewiseCompute(int32_t ubPingpong, uint32_t souterSize, uint64_t attenMaskOffsetPre);

    __aicore__ inline void Bmm1VecInputCopyIn();

    __aicore__ inline void Bmm1ResDoVecBmm2Compute();

    __aicore__ inline void ComputeEachCoreSInnerLoop();

    __aicore__ inline void SInnerLoopFunc(int64_t sInnerFirstToken, int64_t sInnerEndToken, int curBatch, int32_t preTokens, int32_t nextTokens);

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);

    __aicore__ inline void ComputeEachCoreBalance(uint32_t coreIdx);

    __aicore__ inline void InitEachCoreWorkspace(uint32_t coreIdx, int32_t blockNum);

    __aicore__ inline void ComputeEachCoreSplitSeqOneN(uint32_t coreIdx);

    __aicore__ inline void ProcessLastSouterLoopFinalRes();

    __aicore__ inline void CheckRowInvalid(int64_t preTokens, int64_t nextTokens, PFAComputeParam* params);

    __aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue);

    __aicore__ inline void Bmm2Antiquant(PFAComputeParam *params) {
        int step = this->tilingData->promptAttentionSingleCoreParams.kvAntiquantSInnerSize;
        int kvAntiquantLoopTimes = (params->singleProcessSInnerBmmTail + step -1) / step;
        int headSize = this->tilingData->promptAttentionBaseParams.alignedHeadSize;

        LocalTensor<T> scaleLocal = this->antiquantScaleUb.template Get<T>(headSize);
        LocalTensor<T> offsetLocal = this->antiquantOffsetUb.template Get<T>(headSize);
        if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel) {
            scaleLocal.SetSize(headSize);
            offsetLocal.SetSize(headSize);
            DataCopy(scaleLocal, this->antiquantScaleGm[((int64_t)this->tilingData->promptAttentionBaseParams.headNumSize + (int64_t)params->batchNOffset) / this->headNumRatio * headSize], headSize);
            if (!this->isAntiquantSymmetric) {
                DataCopy(offsetLocal, this->antiquantOffsetGm[((int64_t)this->tilingData->promptAttentionBaseParams.headNumSize + (int64_t)params->batchNOffset) / this->headNumRatio * headSize], headSize);
            } else {
                Duplicate(offsetLocal, static_cast<T>(0), headSize);    // 对称量化
            }
        }

        DataCopyParams kvCopyParam;
        kvCopyParam.blockLen = headSize / 32;   // KV int8 dtype  32 : 32B对齐
        if constexpr (PFAT::layout == PFALayout::BNSD) {
            kvCopyParam.srcStride = 0;  // BNSD
        } else {
            kvCopyParam.srcStride = (this->MultiHeadKV - headSize) / 32;    // BSH  32 : 32B对齐
        }
        kvCopyParam.dstStride = 0;

        for (int loopIdx = 0; loopIdx < kvAntiquantLoopTimes; loopIdx++) {
            int64_t kvComputeSInnerSize = (loopIdx == kvAntiquantLoopTimes - 1) ? (params->singleProcessSInnerBmmTail - loopIdx * step) : step;
            kvComputeSInnerSize = kvComputeSInnerSize > step ? step : kvComputeSInnerSize;
            int64_t vOffset;
            if constexpr (PFAT::layout == PFALayout::BNSD) {
                vOffset = params->valueOffset + loopIdx * step * headSize;  // BNSD
            } else {
                vOffset = params->valueOffset + loopIdx * step * this->MultiHeadKV; // BSH
            }

            LocalTensor<int8_t> srcLocal = this->kvAntiquantSrcQueue.template AllocTensor<int8_t>();
            LocalTensor<T> dstLocal = this->kvAntiquantDstQueue.template AllocTensor<T>();

            kvCopyParam.blockCount = kvComputeSInnerSize;
            if (this->isKvContinuous == 0) {
                ListTensorDesc valueListDesc((__gm__ void*)this->value_ptr);
                __gm__ uint8_t* tempValueGm = (__gm__ uint8_t*)valueListDesc.GetDataPtr<__gm__ uint8_t>(params->taskBatch);
                this->valueGm.SetGlobalBuffer((__gm__ KV_T*)tempValueGm);
            }
            DataCopy(srcLocal, this->valueGm[vOffset], kvCopyParam);
            this->kvAntiquantSrcQueue.EnQue(srcLocal);
            srcLocal = this->kvAntiquantSrcQueue.template DeQue<int8_t>();

            AntiQuantShapeInfo antiquantShape = {static_cast<uint32_t>(kvComputeSInnerSize), static_cast<uint32_t>(headSize),
                                                 static_cast<uint32_t>(kvComputeSInnerSize), static_cast<uint32_t>(headSize)};
            srcLocal.SetSize(kvComputeSInnerSize * headSize);
            dstLocal.SetSize(kvComputeSInnerSize * headSize);

            if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel) {
                AscendAntiQuant<KV_T, T, false>(dstLocal, srcLocal, offsetLocal, scaleLocal, kvComputeSInnerSize, antiquantShape);
            } else {
                AscendAntiQuant<KV_T, T, false>(dstLocal, srcLocal, this->valueAntiquantOffset, this->valueAntiquantScale, kvComputeSInnerSize, antiquantShape);
            }
            this->kvAntiquantDstQueue.EnQue(dstLocal);

            dstLocal = this->kvAntiquantDstQueue.template DeQue<T>();
            DataCopy(this->valueGmAntiquant[loopIdx * step * headSize], dstLocal, kvComputeSInnerSize * headSize);

            this->kvAntiquantSrcQueue.FreeTensor(srcLocal);
            this->kvAntiquantDstQueue.FreeTensor(dstLocal);
        }
        event_t bmmWaitAntiEvt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(bmmWaitAntiEvt);
        WaitFlag<HardEvent::MTE3_MTE2>(bmmWaitAntiEvt);
    }

    __aicore__ inline void Bmm1Antiquant(PFAComputeParam *params){
        int step = this->tilingData->promptAttentionSingleCoreParams.kvAntiquantSInnerSize;
        int kvAntiquantLoopTimes = (params->singleProcessSInnerBmmTail + step -1) / step;
        int headSize = this->tilingData->promptAttentionBaseParams.alignedHeadSize;

        LocalTensor<T> scaleLocal = this->antiquantScaleUb.template Get<T>(headSize);
        LocalTensor<T> offsetLocal = this->antiquantOffsetUb.template Get<T>(headSize);
        if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel) {
            scaleLocal.SetSize(headSize);
            offsetLocal.SetSize(headSize);
            DataCopy(scaleLocal, this->antiquantScaleGm[params->batchNOffset / this->headNumRatio * headSize], headSize);
            if (!this->isAntiquantSymmetric) {
                DataCopy(offsetLocal, this->antiquantOffsetGm[params->batchNOffset / this->headNumRatio * headSize], headSize);
            } else {
                Duplicate(offsetLocal, static_cast<T>(0), headSize);    // 对称量化
            }
        }

        DataCopyParams kvCopyParam;
        kvCopyParam.blockLen = headSize / 32;   // KV int8 dtype  32 : 32B对齐
        if constexpr (PFAT::layout == PFALayout::BNSD) {
            kvCopyParam.srcStride = 0;  // BNSD
        } else {
            kvCopyParam.srcStride = (this->MultiHeadKV - headSize) / 32;    // BSH  32 : 32B对齐
        }
        kvCopyParam.dstStride = 0;

        for (int loopIdx = 0; loopIdx < kvAntiquantLoopTimes; loopIdx++) {
            int kvComputeSInnerSize = (loopIdx == kvAntiquantLoopTimes - 1) ? (params->singleProcessSInnerBmmTail - loopIdx * step) : step;
            kvComputeSInnerSize = kvComputeSInnerSize > step ? step : kvComputeSInnerSize;
            int64_t kOffset;
            if constexpr (PFAT::layout == PFALayout::BNSD) {
                kOffset = params->tensorBOffset + loopIdx * step * headSize;  // BNSD
            } else {
                kOffset = params->tensorBOffset + loopIdx * step * this->MultiHeadKV; // BSH
            }

            LocalTensor<int8_t> srcLocal = this->kvAntiquantSrcQueue.template AllocTensor<int8_t>();
            LocalTensor<T> dstLocal = this->kvAntiquantDstQueue.template AllocTensor<T>();

            kvCopyParam.blockCount = kvComputeSInnerSize;
            if (this->isKvContinuous == 0) {
                ListTensorDesc keyListDesc((__gm__ void*)this->key_ptr);
                __gm__ uint8_t* tempKeyGm = (__gm__ uint8_t*)keyListDesc.GetDataPtr<__gm__ uint8_t>(params->taskBatch);
                this->keyGm.SetGlobalBuffer((__gm__ KV_T*)tempKeyGm);
            }
            DataCopy(srcLocal, this->keyGm[kOffset], kvCopyParam);
            this->kvAntiquantSrcQueue.EnQue(srcLocal);
            srcLocal = this->kvAntiquantSrcQueue.template DeQue<int8_t>();

            AntiQuantShapeInfo antiquantShape = {static_cast<uint32_t>(kvComputeSInnerSize), static_cast<uint32_t>(headSize),
                                                 static_cast<uint32_t>(kvComputeSInnerSize), static_cast<uint32_t>(headSize)};
            srcLocal.SetSize(kvComputeSInnerSize * headSize);
            dstLocal.SetSize(kvComputeSInnerSize * headSize);

            if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel) {
                AscendAntiQuant<KV_T, T, false>(dstLocal, srcLocal, offsetLocal, scaleLocal, kvComputeSInnerSize, antiquantShape);
            } else {
                AscendAntiQuant<KV_T, T, false>(dstLocal, srcLocal, this->keyAntiquantOffset, this->keyAntiquantScale, kvComputeSInnerSize, antiquantShape);
            }
            this->kvAntiquantDstQueue.EnQue(dstLocal);

            dstLocal = this->kvAntiquantDstQueue.template DeQue<T>();
            DataCopy(this->keyGmAntiquant[loopIdx * step * headSize], dstLocal, kvComputeSInnerSize * headSize);

            this->kvAntiquantSrcQueue.FreeTensor(srcLocal);
            this->kvAntiquantDstQueue.FreeTensor(dstLocal);
        }
        event_t bmmWaitAntiEvt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(bmmWaitAntiEvt);
        WaitFlag<HardEvent::MTE3_MTE2>(bmmWaitAntiEvt);
    }
    __aicore__ inline void Bmm1ComputeIterate(int64_t qOffset, int64_t kOffset, int32_t singleCoreM, int32_t singleCoreN, int32_t unalignSingleCoreN, int pingpong, int taskBatch) {
       if (this->mm1SingleCoreNPrev != singleCoreN) {
            // 减少SetOrgShape调用次数，可以减少cv通信次数
            this->mm.SetOrgShape(this->tilingData->bmm1TilingDataRect.M, this->tilingData->bmm1TilingDataRect.N,
                                 this->tilingData->bmm1TilingDataRect.Ka, this->tilingData->bmm1TilingDataRect.Kb,
                                 singleCoreN);
            this->mm1SingleCoreNPrev = singleCoreN;
        }
        this->mm.SetTail(singleCoreM, unalignSingleCoreN);
        this->mm.SetTensorA(this->queryGm[qOffset]);

        if (this->isKvContinuous == 0) {
            ListTensorDesc keyListDesc((__gm__ void*)this->key_ptr);
            __gm__ uint8_t* tempKeyGm = (__gm__ uint8_t*)keyListDesc.GetDataPtr<__gm__ uint8_t>(taskBatch);
            this->keyGm.SetGlobalBuffer((__gm__ KV_T*)tempKeyGm);
        }

        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            this->mm.SetTensorB(this->keyGmAntiquant, true);
        } else {
            this->mm.SetTensorB(this->keyGm[kOffset], true);
        }

        // quant:
        if constexpr (IsSameType<T, int8_t>::value) {
            this->mm.SetQuantScalar(this->dequantScale1);
        }

        this->mm.template IterateAll<false>(this->bmm1ResGmDb[pingpong], false, false, true);
    }

    __aicore__ inline void Bmm1GmResCopyInUb(LocalTensor<computeType> &mmResUb, int64_t gmOffset,
        int32_t blockCount, int32_t blockLen, int32_t srcStride, int pingpong, int ubPingpong,
        uint32_t souterSize, bool unalign, uint32_t alignSInner, uint32_t unalignSInner, bool setCopyIn) {
        if (unalign) {
            mmResUb.SetSize(souterSize * alignSInner);
        }
        else {
            mmResUb.SetSize(souterSize * alignSInner);
        }

        this->mm1GmUbCopyParam[ubPingpong].blockCount = blockCount;
        this->mm1GmUbCopyParam[ubPingpong].blockLen = blockLen;
        this->mm1GmUbCopyParam[ubPingpong].srcStride = srcStride;
        this->mm1GmUbCopyParam[ubPingpong].dstStride = 0;

        WaitFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[ubPingpong]);
        DataCopy(mmResUb, this->bmm1ResGmDb[pingpong][gmOffset], this->mm1GmUbCopyParam[ubPingpong]);
        SetFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[ubPingpong]);
    }

    __aicore__ inline void SoftmaxResCopyOut(LocalTensor<computeType> &mmResUb, int64_t gmOffset, int pingpong, int ubPingpong, uint32_t singleProcessSInnerSizeNow) {
        if constexpr (PFAT::calcMode == Mode::HighPrecision ||
            IsSameType<T, bfloat16_t>::value) {
            // 原地cast
            LocalTensor<T> tmpSoftmaxResUb = mmResUb.template ReinterpretCast<T>();
            pipe_barrier(PIPE_V);
            Cast(tmpSoftmaxResUb, mmResUb, RoundMode::CAST_ROUND, mmResUb.GetSize());

            // cast之后有效数据长度减半，datacopy数据长度要改
            this->mm1GmUbCopyParam[ubPingpong].blockLen = singleProcessSInnerSizeNow / (32 / sizeof(T));   // 32 : 32B对齐
        }

        this->Bmm1Queue.EnQue(mmResUb);  // 不能挪前，必须放在这，否则精度误差。
        this->mm1GmUbCopyParam[ubPingpong].dstStride = this->mm1GmUbCopyParam[ubPingpong].srcStride;
        this->mm1GmUbCopyParam[ubPingpong].srcStride = 0;

        mmResUb = this->Bmm1Queue.template DeQue<computeType>();

        if constexpr (PFAT::calcMode == Mode::HighPrecision ||
            IsSameType<T, bfloat16_t>::value) {
            // 把fp16的数据拷贝到fp32的gm上，紧凑排列offset要改
            GlobalTensor<T> tmpBmm1ResGmDb;
            tmpBmm1ResGmDb.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->bmm1ResGmDb[pingpong][gmOffset / 2].address_), mmResUb.GetSize());     // 这里可能有数据对齐的问题，是否有bug待验证
            LocalTensor<T> tmpSoftmaxResUb = mmResUb.template ReinterpretCast<T>();
            DataCopy(tmpBmm1ResGmDb, tmpSoftmaxResUb, this->mm1GmUbCopyParam[ubPingpong]);
        } else {
            DataCopy(this->bmm1ResGmDb[pingpong][gmOffset], mmResUb, this->mm1GmUbCopyParam[ubPingpong]);
        }

        SetFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[ubPingpong]);
    }

    __aicore__ inline void Res1VecCompute(PFAComputeParam *params) {
        int64_t mm1ResGmOffset = 0;
        int64_t nextMm1ResGmOffset = 0;
        int64_t attenMaskOffset = params->attenMaskOffset;
        int64_t attenMaskOffsetPre = params->attenMaskOffsetPre;
        uint64_t pseShiftOffset = params->pseShiftOffset;
        LocalTensor<float> softmaxMaxUbSub;
        LocalTensor<float> softmaxSumUbSub;
        LocalTensor<computeType> softmaxExpUbSub;

        int ubPingpong = 0;
        int64_t nextSouterOffset;
        uint32_t computeSize;
        for (int64_t souterOffset = 0; souterOffset < params->singleProcessSOuterSize; souterOffset = nextSouterOffset) {     // 待整改
            int64_t leftSouterSize = params->singleProcessSOuterSize - souterOffset;
            int64_t souterSize = (leftSouterSize >= this->softmaxSouterStepLen) ? this->softmaxSouterStepLen : leftSouterSize;
            nextSouterOffset = souterOffset + this->softmaxSouterStepLen;
            bool noLastSoftmaxLoop = (nextSouterOffset < params->singleProcessSOuterSize);
            bool noLastLastSoftmaxLoop = (nextSouterOffset + this->softmaxSouterStepLen < params->singleProcessSOuterSize);
            int64_t nextLeftSouterSize = params->singleProcessSOuterSize - nextSouterOffset;
            int64_t nextSouterSize = (nextLeftSouterSize >= this->softmaxSouterStepLen) ? this->softmaxSouterStepLen : nextLeftSouterSize;
            nextMm1ResGmOffset = mm1ResGmOffset + souterSize * params->mm1SingleCoreN;

            // mm1 + mask*-10000
            softmaxMaxUbSub = this->softmaxMaxUb[souterOffset * 8];  // 8 softmaxShapeArray第二维的长度
            softmaxSumUbSub = this->softmaxSumUb[souterOffset * 8];  // 8 softmaxShapeArray第二维的长度

            // mul scaleValue
            computeSize = souterSize * params->singleProcessSInnerSizeNow;
            WaitFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[ubPingpong]);  // 同步CopyIn
            Muls(this->mmResUb[ubPingpong], this->mmResUb[ubPingpong],
                 static_cast<computeType>(this->tilingData->promptAttentionBaseParams.scaleValue), computeSize);
            pipe_barrier(PIPE_V);

            if (params->usePseShift) {
                this->pseShiftUb = this->tempBmm2Queue.template DeQue<pseShiftType>();
                if constexpr (AscendC::IsSameType<pseShiftCastType, float>::value) {
                    LocalTensor<float> pseShiftCastTensor = this->pseShiftCastUb.template Get<float>(this->pseShiftUbSize);
                    Cast(pseShiftCastTensor, this->pseShiftUb, RoundMode::CAST_NONE, computeSize);
                    pipe_barrier(PIPE_V);
                    Add(this->mmResUb[ubPingpong], this->mmResUb[ubPingpong], pseShiftCastTensor, computeSize);
                } else {
                    Add(this->mmResUb[ubPingpong], this->mmResUb[ubPingpong], this->pseShiftUb, computeSize);
                }
                pipe_barrier(PIPE_V);
                this->tempBmm2Queue.FreeTensor(this->pseShiftUb);
                if (params->useMask && params->sparseBandSelect0) {  // mask 预取，非band模式sparseBandSelect0为true，只需要关注前面的useMask
                    this->PseOrMaskCopyIn(attenMaskOffset, souterSize, params->isInnerTail,
                        params->maskCopyInCol, params->singleProcessSInnerBmmTail, params->padSize, true);
                }
            }

            if(this->attentionMaskType == 4) { // 4:band mode of sparseMode
                SparseBandElewiseCompute(ubPingpong, souterSize, attenMaskOffsetPre);
            } else {
                this->template ElewiseCompute<U>(this->mmResUb[ubPingpong], souterSize, params->singleProcessSInnerSizeNow,
                                             params->maskCopyInCol, params->useMask, this->bmm1ResCopyInEvent[ubPingpong], 0);
            }

            this->isSoftmaxResNeedUpdate = (params->isFirstInnerIter ||
                                            this->softmaxSouterStepLen == 0 ||
                                            souterOffset / this->softmaxSouterStepLen >= MAX_SUBSOUTER_NUM) ?
                                            this->tilingData->promptAttentionBaseParams.isRowInvalid :
                                            this->isSoftmaxNeedUpdate[souterOffset / this->softmaxSouterStepLen];
            if (params->kernelInvalidRow) {
                this->isSoftmaxResNeedUpdate = params->kernelInvalidRow;
            }
            // softmaxflash
            const uint32_t basicSoftmaxSinner = 64;
            const uint32_t basicSoftmaxSouter = 8;
            const uint32_t basicSoftmaxK = 1024;
            if (params->isFirstInnerIter) {
                if ((params->singleProcessSInnerBmmTail % basicSoftmaxSinner == 0)
                    && (params->singleProcessSInnerBmmTail <= basicSoftmaxK)
                    && (souterSize % basicSoftmaxSouter == 0)) {
                    this->SoftmaxBasicComputeFirstNoTail(this->mmResUb[ubPingpong], softmaxMaxUbSub,
                                                         softmaxSumUbSub, souterSize);
                } else {
                    this->SoftmaxComputeFirstTail(this->mmResUb[ubPingpong], softmaxMaxUbSub,
                                                  softmaxSumUbSub, souterSize);
                }
            } else {
                softmaxExpUbSub = this->softmaxExpUb[souterOffset * this->softmaxTypeByteNum];
                if ((params->singleProcessSInnerBmmTail % basicSoftmaxSinner == 0)
                    && (params->singleProcessSInnerBmmTail <= basicSoftmaxK)
                    && (souterSize % basicSoftmaxSouter == 0)) {
                    this->SoftmaxBasicComputeNoTail(this->mmResUb[ubPingpong], softmaxMaxUbSub,
                                                    softmaxSumUbSub, softmaxExpUbSub, souterSize);
                } else {
                    this->SoftmaxComputeTail(this->mmResUb[ubPingpong], softmaxMaxUbSub,
                                             softmaxSumUbSub, softmaxExpUbSub, souterSize);
                }
                pipe_barrier(PIPE_V);
            }
            if (this->softmaxSouterStepLen != 0 && souterOffset / this->softmaxSouterStepLen < MAX_SUBSOUTER_NUM) {
                this->isSoftmaxNeedUpdate[souterOffset / this->softmaxSouterStepLen] = this->isSoftmaxResNeedUpdate;
            }

            if (noLastSoftmaxLoop) {
                // 有可复用的函数(Bmm1VecInputCopyIn)，待整改
                if (params->useMask) {
                    attenMaskOffset += souterSize * this->attentionMaskStride;
                    attenMaskOffsetPre += souterSize * this->attentionMaskStride;
                }
                if (params->usePseShift) {  // pse 预取
                    pseShiftOffset += souterSize * this->pseShiftStride;
                    this->PseOrMaskCopyIn(pseShiftOffset, nextSouterSize, params->isInnerTail,
                        params->pseShiftCopyInCol, params->singleProcessSInnerBmmTail, params->pseShiftPadSize, false);
                } else if (params->useMask && params->sparseBandSelect0) {  // mask 预取，非band模式sparseBandSelect0为true，只需要关注前面的useMask
                    this->PseOrMaskCopyIn(attenMaskOffset, nextSouterSize, params->isInnerTail, params->maskCopyInCol,
                        params->singleProcessSInnerBmmTail, params->padSize, true);
                }

                // mm1 result copyIn 预取
                bool flag = (souterOffset != 0);
                this->Bmm1GmResCopyInUb(this->mmResUb[ubPingpong^1], nextMm1ResGmOffset,
                    nextSouterSize, params->singleProcessSInnerSizeNow / this->softmaxTypeByteNum,
                    (params->mm1SingleCoreN - params->singleProcessSInnerSizeNow) / this->softmaxTypeByteNum,
                    params->gmPingpong, ubPingpong^1, nextSouterSize, params->isInnerTail,
                    params->singleProcessSInnerSizeNow, params->singleProcessSInnerBmmTail, flag);
            }

            if constexpr (IsSameType<T, int8_t>::value) {
                LocalTensor<int8_t> softmaxQuantResUb;
                softmaxQuantResUb = this->mmResUb[ubPingpong].template ReinterpretCast<int8_t>();
                softmaxQuantResUb.SetSize(this->mmResUb[ubPingpong].GetSize());
                this->QuantCompute(softmaxQuantResUb, this->mmResUb[ubPingpong], this->quantScale1, 0, souterSize * params->singleProcessSInnerSizeNow);
                // copy前同步vector quant计算，并修改copy参数为int8属性
                event_t enQueEvtId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(enQueEvtId);
                WaitFlag<HardEvent::V_MTE3>(enQueEvtId);
                this->mm1GmUbCopyParam[ubPingpong].blockLen = params->singleProcessSInnerSizeNow / this->typeByteNum;
                this->mm1GmUbCopyParam[ubPingpong].dstStride = (params->mm1SingleCoreN - params->singleProcessSInnerSizeNow) / this->typeByteNum;
                this->mm1GmUbCopyParam[ubPingpong].srcStride = 0;
                DataCopy(this->quant1ResGmDb[params->gmPingpong][mm1ResGmOffset], softmaxQuantResUb, this->mm1GmUbCopyParam[ubPingpong]);
                SetFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[ubPingpong]);
            } else {
                // softmax res copyOut
                this->SoftmaxResCopyOut(this->mmResUb[ubPingpong], mm1ResGmOffset, params->gmPingpong, ubPingpong, params->singleProcessSInnerSizeNow);
            }

            mm1ResGmOffset = nextMm1ResGmOffset;
            ubPingpong ^= 1;
        }
    }

    __aicore__ inline void Bmm2ComputeIterate() {
        PFAComputeParam *params = this->headParams;
        if ((this->mm2MStridePrev != params->singleProcessSOuterSize)
            || (this->mm2KaStridePrev != params->mm1SingleCoreN)) {
             // 减少SetOrgShape调用次数，可以减少cv通信次数
             this->bmm2.SetOrgShape(params->singleProcessSOuterSize,  // M stride for trans a
                 this->tilingData->bmm2TilingDataRect.N,   // N stride for b
                 params->mm1SingleCoreN,  // Ka stride for a
                 this->tilingData->bmm2TilingDataRect.Kb,   // Kb stride for trans b
                 this->tilingData->promptAttentionBaseParams.headSize);  // Kc
             this->mm2MStridePrev = params->singleProcessSOuterSize;
             this->mm2KaStridePrev = params->mm1SingleCoreN;
        }
        this->bmm2.SetTail(params->singleProcessSOuterSize,
            this->tilingData->promptAttentionBaseParams.headSize, params->singleProcessSInnerBmmTail);
        if constexpr (IsSameType<T, int8_t>::value) {
            this->bmm2.SetTensorA(this->quant1ResGmDb[params->gmPingpong]);
        } else if constexpr (PFAT::calcMode == Mode::HighPrecision ||
            IsSameType<T, bfloat16_t>::value) {
            uint64_t gmSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize *
                              this->tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
            GlobalTensor<T> tmpBmm1ResGmDb;
            tmpBmm1ResGmDb.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->bmm1ResGmDb[params->gmPingpong].address_), gmSize);
            this->bmm2.SetTensorA(tmpBmm1ResGmDb);
        } else {
            this->bmm2.SetTensorA(this->bmm1ResGmDb[params->gmPingpong]);
        }

        if (this->isKvContinuous == 0) {
            ListTensorDesc valueListDesc((__gm__ void*)this->value_ptr);
            __gm__ uint8_t* tempValueGm = (__gm__ uint8_t*)valueListDesc.GetDataPtr<__gm__ uint8_t>(params->taskBatch);
            this->valueGm.SetGlobalBuffer((__gm__ KV_T*)tempValueGm);
        }

        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            this->bmm2.SetTensorB(this->valueGmAntiquant);
        } else {
            this->bmm2.SetTensorB(this->valueGm[params->valueOffset]);
        }

        if constexpr (IsSameType<T, int8_t>::value) {
            this->bmm2.SetQuantScalar(this->dequantScale2);
        }

        this->bmm2.template IterateAll<false>(this->bmm2ResGmDb[params->gmPingpong], false, false, true);
    }

    __aicore__ inline LocalTensor<computeType> AllocBmm2UbRes(PFAComputeParam *params, bool useTbuf, uint32_t& resShapeSize) {
        LocalTensor<computeType> bmm2ResUb;
        // 使用真实需要的Q_S，针对小Q_S优化
        uint32_t neededSouterSize = params->singleProcessSOuterSize;
        if (this->tilingData->promptAttentionBaseParams.seqSize < neededSouterSize) {
            neededSouterSize = this->tilingData->promptAttentionBaseParams.seqSize;
        }

        if (useTbuf) {
            bmm2ResUb = this->tempBmm2Ub.template Get<computeType>(this->bmm2ResUbSize);
        } else {
            bmm2ResUb = this->tempBmm2Queue.template AllocTensor<computeType>();
        }

        resShapeSize = neededSouterSize * this->tilingData->promptAttentionBaseParams.headSize;
        return bmm2ResUb;
    }
    __aicore__ inline void CopyParamsAttrOutOfInnerLoop(PFAComputeParam *dst, PFAComputeParam *src) {
        dst->isFirstInnerIter = src->isFirstInnerIter;
        dst->isSecondInnerIter = src->isSecondInnerIter;
        dst->useMask = src->useMask;
        dst->usePseShift = src->usePseShift;
        dst->singleProcessSOuterSize = src->singleProcessSOuterSize;
        dst->singleProcessSInnerSize = src->singleProcessSInnerSize;
        dst->singleProcessSInnerSizeTail = src->singleProcessSInnerSizeTail;
        dst->maskCopyInCol = src->maskCopyInCol;
        dst->maskInnerTailAlign = src->maskInnerTailAlign;
        dst->padSize = src->padSize;
        dst->pseShiftCopyInCol = src->pseShiftCopyInCol;
        dst->pseShiftInnerTailAlign = src->pseShiftInnerTailAlign;
        dst->pseShiftPadSize = src->pseShiftPadSize;

        dst->unalignSInner = src->unalignSInner;
        dst->tensorAOffset = src->tensorAOffset;
        dst->attentionOutOffset = src->attentionOutOffset;
        dst->batchNOffset = src->batchNOffset;
        dst->sOuterOffset = src->sOuterOffset;
        dst->multiSeqOffset = src->multiSeqOffset;
        dst->multiSeqOffsetBSNDOut = src->multiSeqOffsetBSNDOut;
        dst->SoftMaxOffset = src->SoftMaxOffset;
        dst->taskBatch = src->taskBatch;
        dst->preTokensPerBatch = src->preTokensPerBatch;
        dst->nextTokensPerBatch = src->nextTokensPerBatch;
        dst->actualSeqLengthPerBatch = src->actualSeqLengthPerBatch;
        dst->actualSeqLengthKVPerBatch = src->actualSeqLengthKVPerBatch;
    }
};

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::AllocGlobalResources() {
    for (int i = 0; i < 2; ++i) {
        this->mmResUb[i] = this->Bmm1Queue.template AllocTensor<computeType>();
    }
    for (int i = 0; i < 2; ++i) {
        this->bmm1ResCopyInEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        this->bmm1ResCopyOutEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        this->bmm2ResCopyInEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[i]);
    }
    this->attenOutCopyOut = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

    this->softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();
    this->softmaxSumUb = this->softmaxMaxUb[this->tilingData->promptAttentionTensorSizeRect.softmaxMaxSize];
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::FreeGlobalResources() {
    this->softmaxOutQueue.FreeTensor(this->softmaxMaxUb);

    for (int i = 0; i < 2; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[i]);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(this->attenOutCopyOut);
    for (int i = 0; i < 2; ++i) {
        this->Bmm1Queue.FreeTensor(this->mmResUb[i]);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::Process() {
    AllocGlobalResources();

    if constexpr (PFAT::MM_TYPE == MatMulType::MM_IBSHARE_NORM) {
        ComputeEachCoreSplitSeqOneN(this->tmp_block_idx);
    } else {
        if (this->headNumRatio != 1 || this->tilingData->promptAttentionInitOutputParams.needInit ||
            this->tilingData->promptAttentionBaseParams.batchSize != 1) {
            ComputeEachCore(this->tmp_block_idx);
        }
        else {
            ComputeEachCoreBalance(this->tmp_block_idx);
        }
    }

    // 清空队列剩余参数
    while (this->queSize > 0) {
        this->queSize--;
        ComputeEachCoreSInnerLoop();

        this->preHeadParams = this->headParams;

        // 出队
        this->headId = (this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
        this->headParams = &this->pfaParamsQueue[this->headId];
    }
    ProcessLastSouterLoopFinalRes();

    FreeGlobalResources();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::PseOrMaskCopyIn(uint64_t offset, uint32_t souterSize,
    bool isInnerTail, uint32_t alignSInner, uint32_t unalignSInner, uint32_t padSize, bool isMask) {
    // 使用真实需要的Q_S，针对小Q_S优化
    uint32_t neededSouterSize = souterSize;
    if (this->tilingData->promptAttentionBaseParams.seqSize < neededSouterSize) {
        neededSouterSize = this->tilingData->promptAttentionBaseParams.seqSize;
    }

    uint32_t lenOfType = 1;  // 每个数据的长度
    uint32_t dataStride = 0; // stride大小

    if (isMask) {  // pse和mask复用该函数，复用ub
        this->attenMaskUb = this->tempBmm2Queue.template AllocTensor<U>();
        this->attenMaskUb.SetSize(souterSize * alignSInner);
        lenOfType = sizeof(U);
        dataStride = this->attentionMaskStride;
    } else {
        this->pseShiftUb = this->tempBmm2Queue.template AllocTensor<pseShiftType>();
        this->pseShiftUb.SetSize(souterSize * alignSInner);
        lenOfType = sizeof(pseShiftType);
        dataStride = this->pseShiftStride;
    }

    DataCopyExtParams intriParams;
    intriParams.blockCount = neededSouterSize;  // 此处应该是非对齐
    intriParams.blockLen = alignSInner * lenOfType;
    intriParams.srcStride = (dataStride - alignSInner) * lenOfType;
    if (isInnerTail) {
        intriParams.blockLen = unalignSInner * lenOfType;
        intriParams.srcStride = (dataStride - unalignSInner) * lenOfType;
    }
    intriParams.dstStride = 0;

    if (isMask) {
        DataCopyPadExtParams<U> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = 1;
        if (isInnerTail) {
            padParams.rightPadding = padSize;
        } else {
            padParams.rightPadding = 0;
        }
        DataCopyPad(this->attenMaskUb, this->attenMaskGm[offset], intriParams, padParams);
        this->tempBmm2Queue.template EnQue<U>(this->attenMaskUb);
    } else {
        DataCopyPadExtParams<pseShiftType> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = 1;
        if (isInnerTail) {
            padParams.rightPadding = padSize;
            if constexpr (IsSameType<T, int8_t>::value) {
                if (((intriParams.blockLen / lenOfType + padSize) % 32) != 0) {
                    intriParams.dstStride = 1;  // 如果qkv是int8，此时pad的长度相差一个block，需跳存
                }
            }
        } else {
            padParams.rightPadding = 0;
        }

        DataCopyPad(this->pseShiftUb, this->pseShiftGm[offset], intriParams, padParams);
        this->tempBmm2Queue.template EnQue<pseShiftType>(this->pseShiftUb);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::Bmm1VecInputCopyIn() {
    PFAComputeParam *params = this->headParams;

    this->softmaxSouterStepLen = this->softmaxFlashTilingData.srcM;
    // 优化尾块，softmax循环次数
    if (this->softmaxFlashTilingData.srcK != params->singleProcessSInnerSizeNow) {
        this->softmaxSouterStepLen = this->softmaxFlashTilingData.srcSize / params->singleProcessSInnerSizeNow / 8 * 8; // 8对齐
        this->softmaxSouterStepLen = ((this->softmaxSouterStepLen > params->singleProcessSOuterSize) ||
        (this->softmaxSouterStepLen == 0)) ?
        params->singleProcessSOuterSize : this->softmaxSouterStepLen;
    }
    uint32_t souterSize = this->softmaxSouterStepLen;

    if (params->usePseShift) {
        this->PseOrMaskCopyIn(params->pseShiftOffset, souterSize, params->isInnerTail, params->pseShiftCopyInCol,
            params->singleProcessSInnerBmmTail, params->pseShiftPadSize, false);
    } else if (params->useMask && params->sparseBandSelect0) {  // 非band模式sparseBandSelect0为true，只需要关注前面的useMask
        this->PseOrMaskCopyIn(params->attenMaskOffset, souterSize, params->isInnerTail, params->maskCopyInCol,
            params->singleProcessSInnerBmmTail, params->padSize, true);
    }

    this->Bmm1GmResCopyInUb(this->mmResUb[0], 0,
        souterSize, params->singleProcessSInnerSizeNow / this->softmaxTypeByteNum,
        (params->mm1SingleCoreN - params->singleProcessSInnerSizeNow) / this->softmaxTypeByteNum,
        params->gmPingpong, 0, souterSize, params->isInnerTail, params->singleProcessSInnerSizeNow,
        params->singleProcessSInnerBmmTail, false);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::SparseBandElewiseCompute(int32_t ubPingpong, uint32_t souterSize, uint64_t attenMaskOffsetPre) {
    PFAComputeParam *params = this->headParams;
    if (params->sparseBandSelect0) {    // 选0的部分
        this->template ElewiseCompute<U>(this->mmResUb[ubPingpong], souterSize, params->singleProcessSInnerSizeNow,
                                        params->maskCopyInCol, params->useMask, this->bmm1ResCopyInEvent[ubPingpong], 0);
    }
    if (params->sparseBandSelect1) {    // 选1的部分
        this->PseOrMaskCopyIn(attenMaskOffsetPre, souterSize, params->isInnerTail, params->maskCopyInCol,
            params->singleProcessSInnerBmmTail, params->padSize, true);

        pipe_barrier(PIPE_V);
        this->template ElewiseCompute<U>(this->mmResUb[ubPingpong], souterSize, params->singleProcessSInnerSizeNow,
                                        params->maskCopyInCol, params->useMask, this->bmm1ResCopyInEvent[ubPingpong], 1);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::Bmm1ResDoVecBmm2Compute() {
    PFAComputeParam *params = this->headParams;
    LocalTensor<computeType> bmm2ResUb;
    uint32_t resShapeSize;

    // 处理当前循环的softmax，使用headParams
    this->Res1VecCompute(params);

    if (params->isFirstInnerIter) {
        ProcessLastSouterLoopFinalRes();    // 处理上次任务souter循环的输出，内部所有调用需要使用preHeadParams
        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            this->Bmm2Antiquant(params);
        }
        this->Bmm2ComputeIterate();
    } else if (params->isSecondInnerIter) {                      
        bmm2ResUb = AllocBmm2UbRes(this->headParams, true, resShapeSize);    // 第二次不需要做加法，用Tbuf
        this->bmm2.WaitIterateAll();
        DataCopy(bmm2ResUb, this->bmm2ResGmDb[this->headParams->gmPingpong ^ 1], resShapeSize);  // 处理上次循环的bmm2结果，所以是^1
        SetFlag<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[this->headParams->gmPingpong ^ 1]);
        WaitFlag<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[this->headParams->gmPingpong ^ 1]);
        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            this->Bmm2Antiquant(params);
        }
        this->Bmm2ComputeIterate();    // 触发当前循环的bmm2计算，使用headParams
        this->UpdateVmul(this->softmaxExpUb);
    } else {
        bmm2ResUb = AllocBmm2UbRes(this->headParams, false, resShapeSize);
        this->bmm2.WaitIterateAll();
        DataCopy(bmm2ResUb, this->bmm2ResGmDb[this->headParams->gmPingpong ^ 1], resShapeSize);  // 处理上次循环的bmm2结果，所以是^1
        this->tempBmm2Queue.template EnQue<computeType>(bmm2ResUb);
        bmm2ResUb = this->tempBmm2Queue.template DeQue<computeType>();
        this->Bmm2UpdateAdd(bmm2ResUb);
        this->tempBmm2Queue.FreeTensor(bmm2ResUb);
        pipe_barrier(PIPE_V);
        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            this->Bmm2Antiquant(params);
        }
        this->Bmm2ComputeIterate();    // 触发当前循环的bmm2计算，使用headParams
        this->UpdateVmul(this->softmaxExpUb);
    }

    if (params->isLastInnerIter) {
        // copy sle
        if (this->tilingData->promptAttentionBaseParams.isSoftMaxLseEnable) {
            this->SoftmaxLseCopyOut(this->softmaxSumUb, this->softmaxMaxUb);
        }
        // 复用softmaxExp Ub空间，拷贝sum值
        LocalTensor<float> softmaxSumTmp = this->softmaxExpUb_.template Get<float>(this->softmaxSumSize);
        DataCopy(softmaxSumTmp, this->softmaxSumUb, this->softmaxSumSize);
        this->copyOutPrevIter = true;
        this->needAdd = !params->isFirstInnerIter;    // 第一次循环即为最后一次循环时，不需要做add
    }
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::CheckRowInvalid(int64_t preTokens, int64_t nextTokens,
                                                                             PFAComputeParam* params) {
    // 1. nextToken cross souter   2. preToken cross souter   3. preToken cross souter 2
    //  |____            |          |____  \        |           |___\_   \      |
    //  | \  |           |          |   |   \       |           |    \|   \     |
    //  |_ \ |           |          | \ |    \      |           |     |\   \    |
    //  |   \            |          |  \|     \     |           |____ | \   \   |
    //  |\   \           |          |___|\     \    |           |     |  \   \  |
    //  | \              |          |               |           |     |         |

    bool nextokenCrossSouter = nextTokens < 0 && abs(nextTokens) > params->sOuterOffset &&
                             abs(nextTokens) < (params->sOuterOffset + params->singleProcessSOuterSize);
    int32_t sinnerSize = params->isInnerTail ? params->singleProcessSInnerBmmTail : params->singleProcessSInnerSize;
    bool pretokenCrossSouter = preTokens > 0 && preTokens > params->sOuterOffset &&
                             preTokens < (params->sOuterOffset + params->singleProcessSOuterSize) &&
                             (params->sOuterOffset + params->singleProcessSOuterSize - preTokens) > sinnerSize;
    bool pretokenCrossSouter2 =
         preTokens < 0 && preTokens + params->actualSeqLengthKVPerBatch > params->sOuterOffset &&
         preTokens + params->actualSeqLengthKVPerBatch < (params->sOuterOffset + params->singleProcessSOuterSize);
    if (params->isFirstInnerIter && (nextokenCrossSouter || pretokenCrossSouter || pretokenCrossSouter2)) {
        params->kernelInvalidRow = 1;
    } else {
        params->kernelInvalidRow = 0;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::SInnerLoopFunc(int64_t sInnerFirstToken, int64_t sInnerLastToken, int curBatch,
                                                                            int32_t preTokens, int32_t nextTokens) {
    // params 传引用，当tailParams, params也跟着更新
    PFAComputeParam *&params = this->tailParams;                // 配置新任务，新任务会放到队列尾，使用tailParams
    int32_t basicSInnerSize = (int32_t)(params->singleProcessSInnerSize);
    int32_t startIndex = sInnerFirstToken / basicSInnerSize;
    int32_t endIndex = (sInnerLastToken + basicSInnerSize - 1) / basicSInnerSize;
    bool isS2Load = (this->maxInnerLoopTimes == 1);
    if (endIndex > this->maxInnerLoopTimes) {
        endIndex = this->maxInnerLoopTimes;
    }

    constexpr int32_t softmaxInnerBasicSize = 64;
    // 上三角mask场景，动态last sinnersize， 根据上三角mask，计算出当前last inner iter的最紧凑sinnersize（包含所有mask 0值的最小sinnersize）
    int64_t firstInnerMargin = (sInnerFirstToken - startIndex * basicSInnerSize) / softmaxInnerBasicSize * softmaxInnerBasicSize;
    int64_t lastInnerMargin = (endIndex * basicSInnerSize - sInnerLastToken) / softmaxInnerBasicSize * softmaxInnerBasicSize;
    params->tensorAOffset = this->tensorACoreOffset;
    params->mm1SingleCoreN = params->singleProcessSInnerSize;
    params->isFirstInnerIter = true;
    params->isSecondInnerIter = true;
    params->taskBatch = curBatch;
    this->isSoftmaxLseNeedUpdate = false;
    for (int32_t sInnerLoopIdx = startIndex; sInnerLoopIdx < endIndex; sInnerLoopIdx++) {
        params->isFirstInnerIter = (sInnerLoopIdx == startIndex);
        params->isSecondInnerIter = (sInnerLoopIdx == (startIndex + 1));
        params->isLastInnerIter = (sInnerLoopIdx == endIndex - 1);
        if (unlikely(isS2Load)) {
            params->isInnerTail = true;
        } else {
            params->isInnerTail = (sInnerLoopIdx == this->maxInnerLoopTimes - 1);
        }

        if (unlikely(params->isInnerTail)) {
            lastInnerMargin = (sInnerLoopIdx * basicSInnerSize + params->unalignSInner - sInnerLastToken)
                / softmaxInnerBasicSize * softmaxInnerBasicSize;
            lastInnerMargin = (lastInnerMargin > 0) ? lastInnerMargin : 0;
            params->mm1SingleCoreN = params->singleProcessSInnerSizeTail - lastInnerMargin;
            params->singleProcessSInnerSizeNow = params->singleProcessSInnerSizeTail - lastInnerMargin;
            params->singleProcessSInnerBmmTail = params->unalignSInner - lastInnerMargin;
            params->maskCopyInCol = params->maskInnerTailAlign - lastInnerMargin;
            params->pseShiftCopyInCol = params->pseShiftInnerTailAlign - lastInnerMargin;
        } else {
            params->mm1SingleCoreN = params->singleProcessSInnerSize;
            params->singleProcessSInnerSizeNow = params->singleProcessSInnerSize;
            params->singleProcessSInnerBmmTail = params->singleProcessSInnerSize;
            params->maskCopyInCol = params->singleProcessSInnerSize;
            params->pseShiftCopyInCol = params->singleProcessSInnerSize;
            if (params->isLastInnerIter) {
                params->mm1SingleCoreN -= lastInnerMargin;
                params->singleProcessSInnerSizeNow -= lastInnerMargin;
                params->singleProcessSInnerBmmTail -= lastInnerMargin;
                params->maskCopyInCol -= lastInnerMargin;
                params->pseShiftCopyInCol -= lastInnerMargin;
            }
        }
        if (params->isFirstInnerIter) {
            params->mm1SingleCoreN -= firstInnerMargin;
            params->singleProcessSInnerSizeNow -= firstInnerMargin;
            params->singleProcessSInnerBmmTail -= firstInnerMargin;
            params->maskCopyInCol -= firstInnerMargin;
            params->pseShiftCopyInCol -= firstInnerMargin;
            params->tensorBOffset = this->GetBmm1TensorBOffset(params, sInnerLoopIdx, firstInnerMargin);
            this->ComputeOffset(sInnerLoopIdx, firstInnerMargin);
        } else {
            params->tensorBOffset = this->GetBmm1TensorBOffset(params, sInnerLoopIdx);
            this->ComputeOffset(sInnerLoopIdx, 0);
        }

        if (this->attentionMaskType == 2 || this->attentionMaskType == 3) {
            params->useMask = ((sInnerFirstToken + params->singleProcessSOuterSize) > ((int64_t)sInnerLoopIdx * (int64_t)basicSInnerSize)
                || (sInnerLastToken - params->singleProcessSOuterSize < ((int64_t)(sInnerLoopIdx + 1) * (int64_t)basicSInnerSize)));
        }

        // 判断在核内是否开启行无效
        CheckRowInvalid(preTokens, nextTokens, params);
 
        if (this->attentionMaskType == 4) {
            int32_t sOuterOffset = params->attenMaskOffset / SPARSE_ATTENTION_MASK_SIZE;
            int32_t sInnerOffset = params->attenMaskOffset % SPARSE_ATTENTION_MASK_SIZE;
            params->sparseBandSelect0 = (sOuterOffset < (sInnerOffset + (int32_t)params->maskCopyInCol));
            sOuterOffset = params->attenMaskOffsetPre / SPARSE_ATTENTION_MASK_SIZE;
            sInnerOffset = params->attenMaskOffsetPre % SPARSE_ATTENTION_MASK_SIZE;
            params->sparseBandSelect1 = (sOuterOffset > (sInnerOffset - (int32_t)params->singleProcessSOuterSize));
            params->useMask = params->sparseBandSelect0 || params->sparseBandSelect1;
        } else {        // 非band模式，不涉及sparseBandSelect0和sparseBandSelect1，都设置为true，保证不影响公共流程
            params->sparseBandSelect0 = true;
            params->sparseBandSelect1 = true;
        }

        if (this->queSize >= this->queSizeLimit) {
            // 队列满，触发任务，headParams指向的任务开始发指令
            ComputeEachCoreSInnerLoop();

            // prehead更新
            this->preHeadParams = this->headParams;

            // head出队
            this->headId = (this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
            this->headParams = &this->pfaParamsQueue[this->headId];

            // tail入队
            this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
            PFAComputeParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
            if ((sInnerLoopIdx - startIndex) < PFA_PARAMS_QUEUE_CAPBABILITY - 1) {
                // 将旧head参数覆盖。当前下一个tail未在Inner循环外赋值，没有参数，需要拷贝一下循环外会记录的参数
                this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
            }
            nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
            this->tailParams = nextTailParams;
        }
        else {// tail入队
            this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
            PFAComputeParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
            // 将旧head参数覆盖。当前下一个tail未在Inner循环外赋值，没有参数，需要拷贝一下循环外会记录的参数
            this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
            nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
            this->tailParams = nextTailParams;
            this->queSize++;
        }
    }
}

template<typename PFAT>
__aicore__ inline int64_t PromptFlashAttentionS1s2Bns1X910<PFAT>::ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue) {
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::ComputeEachCoreSInnerLoop() {
    PFAComputeParam *params = this->headParams;
    PFAComputeParam *nextParams = &(this->pfaParamsQueue[(this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY]);

    // mm1 compute
    if (this->isGlobalFirstCompute) {
        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            this->Bmm1Antiquant(params);
        }
        this->Bmm1ComputeIterate(params->tensorAOffset, params->tensorBOffset,
            params->singleProcessSOuterSize, params->mm1SingleCoreN, params->singleProcessSInnerBmmTail, params->gmPingpong,
            params->taskBatch);
    }
    this->mm.WaitIterateAll();

    Bmm1VecInputCopyIn();

    if (this->queSize > 0) {
        // 预取下一个mm1计算
        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            this->Bmm1Antiquant(nextParams);
        }
        this->Bmm1ComputeIterate(nextParams->tensorAOffset, nextParams->tensorBOffset,
            nextParams->singleProcessSOuterSize, nextParams->mm1SingleCoreN, nextParams->singleProcessSInnerBmmTail, nextParams->gmPingpong,
            nextParams->taskBatch);
    }

    Bmm1ResDoVecBmm2Compute();
    this->isGlobalFirstCompute = false;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::ComputeEachCore(uint32_t coreIdx) {
    int64_t blockNum = GetBlockNum() * GetTaskRation();

    InitEachCoreWorkspace(coreIdx, blockNum);

    int actualCoreNums = this->tilingData->promptAttentionSingleCoreParams.actualCoreNums;
    if (g_coreType == AIV && coreIdx >= actualCoreNums) {
        return;
    }
    int sNum = this->tilingData->promptAttentionBaseParams.dimNumOfseq;

    // 临时复用
    // CoreHeadNumTail to coreNStart
    // actualS1 to coreNEnd
    // actualCoreNums to coreSidStart
    // singleCoreHeadNumSize to coreSidEnd
    int sIdStart = this->tilingData->promptAttentionSeqParams.actualCoreNums[coreIdx];
    int sIdEnd = this->tilingData->promptAttentionSeqParams.singleCoreHeadNumSize[coreIdx];
    int outerLoopStart = this->tilingData->promptAttentionSeqParams.coreSeqPosStart[coreIdx];
    int outerLoopEnd = this->tilingData->promptAttentionSeqParams.coreSeqPosEnd[coreIdx];
    int nLoopStart = this->tilingData->promptAttentionSeqParams.CoreHeadNumTail[coreIdx];
    int nLoopEnd = this->tilingData->promptAttentionSeqParams.actualS1[coreIdx];
    int64_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int64_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);

    int tmpOuterLoopEnd;
    int tmpSLoopEnd;
    bool isLast = false;
    int64_t actualSeqLengthsIdx = 0;
    // 必须传引用赋值params，因为head地址在内部更新了
    PFAComputeParam *&params = this->tailParams;

    for (uint32_t loopNIdx = nLoopStart; loopNIdx < nLoopEnd; loopNIdx++) {
        params->batchNOffset = loopNIdx;
        if (loopNIdx != nLoopEnd - 1) {
            tmpSLoopEnd = sNum;
        } else {
            tmpSLoopEnd = sIdEnd;
            isLast = true;
        }
        for (int sIdx = sIdStart; sIdx < tmpSLoopEnd; sIdx++) {
            if (this->isKvContinuous == 0) {
                ListTensorDesc keyListTensorDesc((__gm__ void*)this->key_ptr);

                uint64_t dimInfo[4];
                this->kvTensorDesc.SetShapeAddr(&dimInfo[0]);
                keyListTensorDesc.GetDesc(this->kvTensorDesc, sIdx);
                if (PFAT::layout == PFALayout::BNSD) {
                    this->s2InCurrentBatch = this->kvTensorDesc.GetShape(2);
                } else {
                    this->s2InCurrentBatch = this->kvTensorDesc.GetShape(1);
                }
            }
            this->GetSingleCoreParam(sIdx);
            this->GetSparseParam(&preTokens, &nextTokens, sIdx, params);

            int sOuterBlockNum = (params->actualSeqLengthPerBatch + this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                                  this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;

            if (this->tilingData->promptAttentionBaseParams.isLayoutSH) {    // SH格式偏移量
                params->multiSeqOffset = 0;
                for (int i = 0; i < sIdx; i++) {
                    params->multiSeqOffset += this->actualSeqLengthsGm.GetValue(i);
                }
                params->multiSeqOffset *= this->MultiHeadQ;
            } else {    // 非SH格式偏移量
                params->multiSeqOffset = this->CalMultiSeqOffset(sIdx);
            }

            if (isLast && sIdx == tmpSLoopEnd - 1) {
                tmpOuterLoopEnd = outerLoopEnd;
            } else {
                tmpOuterLoopEnd = sOuterBlockNum;
            }
            for (uint32_t sOuterLoopIdx = outerLoopStart; sOuterLoopIdx < tmpOuterLoopEnd; sOuterLoopIdx++) {
                if (sOuterLoopIdx == sOuterBlockNum - 1) {
                    params->singleProcessSOuterSize = this->singleProcessSOuterSizeTail;
                } else {
                    params->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
                }
                params->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSizeWhole;
                if (nextTokens < 0 && params->sOuterOffset < ((nextTokens * (-1)) /
                    this->singleProcessSOuterSizeWhole * this->singleProcessSOuterSizeWhole)) {
                        continue;
                }
                int64_t sInnerFirstToken = ClipSInnerToken(params->sOuterOffset - (int64_t)preTokens, 0, params->actualSeqLengthKVPerBatch);
                int64_t sInnerLastToken = ClipSInnerToken(params->sOuterOffset + (int64_t)nextTokens + params->singleProcessSOuterSize, 0, params->actualSeqLengthKVPerBatch);
                if (sInnerLastToken <= sInnerFirstToken) {
                    continue;
                }

                this->LoopSOuterOffsetInit(params->multiSeqOffset, sIdx);
                SInnerLoopFunc(sInnerFirstToken, sInnerLastToken, sIdx, preTokens, nextTokens);
            }
            outerLoopStart = 0;
        }
        sIdStart = 0;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::ComputeEachCoreBalance(uint32_t coreIdx) {
    int sNum = this->tilingData->promptAttentionBaseParams.dimNumOfseq;
    if (sNum == 0) {
	    return;
    }
    int64_t blockNum = GetBlockNum() * GetTaskRation();
    if (coreIdx % 2 == 1) {
        coreIdx = blockNum - coreIdx;
    }

    InitEachCoreWorkspace(coreIdx, blockNum);

    int64_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int64_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);

    int32_t sIdx = 0;
    // batch内所有seq的length相同
    int64_t actualSeqLengthsIdx = this->isActualLenDimsNull ? this->tilingData->promptAttentionBaseParams.seqSize : this->actualSeqLengthsGm.GetValue(sIdx);

    PFAComputeParam *&params = this->tailParams;
    if (this->attentionMaskType == 4) {
        this->GetSparseParam(&preTokens, &nextTokens, sIdx, params);
        actualSeqLengthsIdx = ((int64_t)actualSeqLengthsIdx >
                               (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)preTokens) ?
                            this->tilingData->promptAttentionBaseParams.seqInnerSize + preTokens :
                            actualSeqLengthsIdx;  // 该分核不会传actualseqlenkv, 不用改seqInnerSize
    } else {
        actualSeqLengthsIdx = (this->attentionMaskType == 0 && (int64_t)actualSeqLengthsIdx >
                            (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize +
                            (int64_t)this->tilingData->promptAttentionBaseParams.preTokens) ?
                            (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize + (int64_t)this->tilingData->promptAttentionBaseParams.preTokens :
                            actualSeqLengthsIdx;
    }

    int64_t sOuterBlockNum = (actualSeqLengthsIdx +
                              this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                              this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
    int64_t sNumMulHeadNum = this->tilingData->promptAttentionBaseParams.headNumSize * sNum;
    int64_t totalTilingN = sNumMulHeadNum * sOuterBlockNum;

    for (int64_t tilingIdx = coreIdx; tilingIdx < totalTilingN; tilingIdx += (blockNum - (tilingIdx % blockNum)) * 2 - 1) {
        int64_t sIdxMulbatchNOffset = tilingIdx % sNumMulHeadNum;
        sIdx = sIdxMulbatchNOffset % sNum;
        params->batchNOffset = sIdxMulbatchNOffset / sNum;
        int64_t sOuterLoopIdx = sOuterBlockNum - 1 - (tilingIdx / sNumMulHeadNum);
        this->GetSingleCoreParam(sIdx);
        this->GetSparseParam(&preTokens, &nextTokens, sIdx, params);
        if (this->tilingData->promptAttentionBaseParams.isLayoutSH) {    // SH格式偏移量
            params->multiSeqOffset = 0;
            for (int i = 0; i < sIdx; i++) {
                params->multiSeqOffset += this->actualSeqLengthsGm.GetValue(i);
            }
            params->multiSeqOffset *= this->MultiHeadQ;
        } else {    // 非SH格式偏移量
            params->multiSeqOffset = this->CalMultiSeqOffset(sIdx);
        }

        if (sOuterLoopIdx == 0) {
            params->singleProcessSOuterSize = this->singleProcessSOuterSizeTail;
            params->sOuterOffset = 0;
        } else {
            params->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
            params->sOuterOffset = this->singleProcessSOuterSizeTail + (sOuterLoopIdx-1) * this->singleProcessSOuterSizeWhole;
        }
        if (nextTokens < 0 && params->sOuterOffset < ((nextTokens * (-1)) /
            this->singleProcessSOuterSizeWhole * this->singleProcessSOuterSizeWhole)) {
                continue;
        }
        int64_t sInnerFirstToken = ClipSInnerToken(params->sOuterOffset - preTokens, 0, params->actualSeqLengthKVPerBatch);
        int64_t sInnerLastToken = ClipSInnerToken(params->sOuterOffset + nextTokens + params->singleProcessSOuterSize, 0, params->actualSeqLengthKVPerBatch);
        if (sInnerLastToken <= sInnerFirstToken) {
            continue;
        }

        this->LoopSOuterOffsetInit(params->multiSeqOffset, sIdx);
        this->SInnerLoopFunc(sInnerFirstToken, sInnerLastToken, sIdx, preTokens, nextTokens);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::InitEachCoreWorkspace(uint32_t coreIdx, int32_t blockNum) {
    this->spmTmpSize = this->tilingData->promptAttentionTensorSizeRect.spmTmpSize;
    this->mmResUbSize = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize;
    this->bmm2ResUbSize = this->tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize;

    int reuseWorkspaceRatio = 2;
    int64_t mm1ResSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize * \
        this->tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
    int64_t mm2ResSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize * \
        this->tilingData->promptAttentionBaseParams.headSize;
    this->bmm1ResGmDb[0].SetGlobalBuffer((__gm__ computeType*)this->workspaceGm[blockNum * this->spmTmpSize +
        coreIdx * mm1ResSize * reuseWorkspaceRatio].GetPhyAddr());
    this->bmm1ResGmDb[1].SetGlobalBuffer((__gm__ computeType*)this->workspaceGm[blockNum * this->spmTmpSize +
        coreIdx * mm1ResSize * reuseWorkspaceRatio + mm1ResSize].GetPhyAddr());
    if constexpr (IsSameType<T, int8_t>::value) {
        this->quant1ResGmDb[0].SetGlobalBuffer((__gm__ int8_t*)this->workspaceGm[blockNum * this->spmTmpSize +
            coreIdx * mm1ResSize * reuseWorkspaceRatio].GetPhyAddr());
        this->quant1ResGmDb[1].SetGlobalBuffer((__gm__ int8_t*)this->workspaceGm[blockNum * this->spmTmpSize +
            coreIdx * mm1ResSize * reuseWorkspaceRatio + mm1ResSize].GetPhyAddr());
    }

    int64_t buff_offset = blockNum * (this->spmTmpSize + mm1ResSize * reuseWorkspaceRatio);
    this->bmm2ResGmDb[0].SetGlobalBuffer((__gm__ computeType*)this->workspaceGm[buff_offset +
        coreIdx * mm2ResSize * reuseWorkspaceRatio].GetPhyAddr());
    this->bmm2ResGmDb[1].SetGlobalBuffer((__gm__ computeType*)this->workspaceGm[buff_offset +
        coreIdx * mm2ResSize * reuseWorkspaceRatio + mm2ResSize].GetPhyAddr());

    if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
        GlobalTensor<T> workspaceGmAntiquant;
        buff_offset = blockNum * (this->spmTmpSize + (mm1ResSize + mm2ResSize) * reuseWorkspaceRatio);
        //高精度模式workspace为fp32类型，但是antiquant结果为fp16类型
        workspaceGmAntiquant.SetGlobalBuffer((__gm__ T*)this->workspaceGm[buff_offset].GetPhyAddr());
        int64_t kvAntiquantSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize * \
            this->tilingData->promptAttentionBaseParams.headSize;
        this->keyGmAntiquant.SetGlobalBuffer((__gm__ T*)workspaceGmAntiquant[
            coreIdx * kvAntiquantSize * reuseWorkspaceRatio].GetPhyAddr());
        this->valueGmAntiquant.SetGlobalBuffer((__gm__ T*)workspaceGmAntiquant[
            coreIdx * kvAntiquantSize * reuseWorkspaceRatio + kvAntiquantSize].GetPhyAddr());
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::ComputeEachCoreSplitSeqOneN(uint32_t coreIdx) {
    int32_t blockNum = GetBlockNum() * GetTaskRation();
    InitEachCoreWorkspace(coreIdx, blockNum);

    int actualCoreNums = this->tilingData->promptAttentionSingleCoreParams.actualCoreNums;
    if (g_coreType == AIV && coreIdx >= actualCoreNums) {
        return;
    }
    int sNum = this->tilingData->promptAttentionBaseParams.dimNumOfseq;
    int headNum = this->tilingData->promptAttentionBaseParams.headNumSize;
    int64_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int64_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);

    // 本分核方案不支持有actualSeq的场景
    uint32_t actualSeqLengths = this->tilingData->promptAttentionBaseParams.seqSize;

    // 必须传引用赋值params，因为head地址在内部更新了
    PFAComputeParam *&params = this->tailParams;

    int64_t bnIdx = 0;
    for (int sIdx = 0; sIdx < sNum; sIdx++) {
        this->GetSingleCoreParam(sIdx);
        this->GetSparseParam(&preTokens, &nextTokens, sIdx, params);
        actualSeqLengths = (this->attentionMaskType == 0 && (int64_t)actualSeqLengths >
                           (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize +
                           (int64_t)this->tilingData->promptAttentionBaseParams.preTokens) ?
                           (this->tilingData->promptAttentionBaseParams.seqInnerSize +
                           this->tilingData->promptAttentionBaseParams.preTokens) :
                           actualSeqLengths;

        uint32_t cubeSOuterSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize * 2;
        int sOuterBlockNum = (params->actualSeqLengthPerBatch + cubeSOuterSize - 1) / cubeSOuterSize;
        params->multiSeqOffset = sIdx * this->tilingData->promptAttentionBaseParams.seqSize * this->MultiHeadQ;

        for (uint32_t loopNIdx = 0; loopNIdx < headNum; loopNIdx++) {
            params->batchNOffset = loopNIdx;
            // 为了让计算量在每个核上尽量均匀
            uint32_t coreIdxCube = coreIdx / 2;
            uint32_t sOutPolicyIdx = (coreIdxCube + bnIdx) % (actualCoreNums / 2);      // actualCoreNums是vector核的个数，cube分核时需要算出cube核个数
            int outerLoopStart = this->tilingData->promptAttentionSeqParams.coreSeqPosStart[sOutPolicyIdx];
            int outerLoopEnd = this->tilingData->promptAttentionSeqParams.coreSeqPosEnd[sOutPolicyIdx];
            for (uint32_t sOuterLoopIdxByCube = outerLoopStart; sOuterLoopIdxByCube < outerLoopEnd; sOuterLoopIdxByCube++) {
                uint32_t sOuterLoopIdx = (coreIdx % 2 == 0) ? (sOuterLoopIdxByCube * 2) : (sOuterLoopIdxByCube * 2 + 1);
                if (sOuterLoopIdxByCube == sOuterBlockNum - 1) {     // TODO: 尾块场景需要考虑优化方案
                    params->singleProcessSOuterSize = this->singleProcessSOuterSizeTail;
                } else {
                    params->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
                }
                params->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSizeWhole;
                int64_t sInnerFirstToken = ClipSInnerToken(params->sOuterOffset - preTokens, 0, params->actualSeqLengthKVPerBatch);
                int64_t sInnerLastToken = ClipSInnerToken(params->sOuterOffset + nextTokens + params->singleProcessSOuterSize, 0, params->actualSeqLengthKVPerBatch);
                if (sInnerLastToken <= sInnerFirstToken) {
                    continue;
                }
                this->LoopSOuterOffsetInit(params->multiSeqOffset, sIdx);
                SInnerLoopFunc(sInnerFirstToken, sInnerLastToken, sIdx, preTokens, nextTokens);
            }
            bnIdx++;
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X910<PFAT>::ProcessLastSouterLoopFinalRes() {
    if (this->copyOutPrevIter) {
        LocalTensor<float> softmaxSumTmp = this->softmaxExpUb_.template Get<float>(this->softmaxSumSize);
        LocalTensor<computeType> bmm2ResPreUb = this->tempBmm2Ub.template Get<computeType>(this->bmm2ResUbSize);
        LocalTensor<computeType>& FinalResUb = bmm2ResPreUb;
        uint32_t resShapeSize;

        LocalTensor<computeType> bmm2ResUb = AllocBmm2UbRes(this->preHeadParams, !this->needAdd, resShapeSize);    // 不做加法时直接用Tbuf
        this->bmm2.WaitIterateAll();
        DataCopy(bmm2ResUb, this->bmm2ResGmDb[this->preHeadParams->gmPingpong], resShapeSize);

        if (this->needAdd) {
            this->tempBmm2Queue.template EnQue<computeType>(bmm2ResUb);
            bmm2ResUb = this->tempBmm2Queue.template DeQue<computeType>();
            this->Bmm2UpdateAdd(bmm2ResUb);
            this->tempBmm2Queue.FreeTensor(bmm2ResUb);
            pipe_barrier(PIPE_V);
        } else {
            event_t tmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));  // set wait连续使用，可以用Fetch，否则要用Alloc
            SetFlag<HardEvent::MTE2_V>(tmp);
            WaitFlag<HardEvent::MTE2_V>(tmp);
        }

        this->Bmm2UpdateDivNoTail(bmm2ResPreUb, softmaxSumTmp);
        if ((PFAT::layout == PFALayout::BSH) ||
            (PFAT::layout == PFALayout::BNSD && this->tilingData->promptAttentionBaseParams.isBSNDOut == 1)) {
            this->DataCopyTransposeOutBSH(FinalResUb);
        } else {
            this->DataCopyTransposeOutBNSD(FinalResUb);
        }
        this->copyOutPrevIter = false;
    }
}

#endif  // PROMPT_FLASH_ATTENTION_S1S2_BNS1_X910_H