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
 * \file prompt_flash_attention_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_H_
#include <cstdint>
#include <vector>
#include <queue>
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/data_copy_transpose_tiling_def.h"
#include "tiling/data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#ifdef ASCENDC_OP_TEST
#define PFA_EXTERN_C extern "C"
#else
#define PFA_EXTERN_C
#endif

#include "prompt_flash_attention_tiling_compile_info.h"
#include "prompt_flash_attention_tiling_const.h"
#include "prompt_flash_attention_tiling_context.h"
#include "prompt_flash_attention_tiling_struct.h"

namespace optiling { 

BEGIN_TILING_DATA_DEF(PromptAttentionBaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, headNumSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqSize);
  TILING_DATA_FIELD_DEF(uint32_t, headSize);
  TILING_DATA_FIELD_DEF(float, scaleValue);
  TILING_DATA_FIELD_DEF(int32_t, preTokens);
  TILING_DATA_FIELD_DEF(int32_t, nextTokens);
  TILING_DATA_FIELD_DEF(int32_t, blockSize);
  TILING_DATA_FIELD_DEF(int32_t, blockTableDim2);
  TILING_DATA_FIELD_DEF(int32_t, PABlockNumSum);
  TILING_DATA_FIELD_DEF(uint32_t, dimNumOfseq);
  TILING_DATA_FIELD_DEF(uint32_t, typeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, seqInnerSize);
  TILING_DATA_FIELD_DEF(uint32_t, prefixSeqInnerSize);
  TILING_DATA_FIELD_DEF(uint32_t, usePseShift);
  TILING_DATA_FIELD_DEF(uint32_t, useMask);
  TILING_DATA_FIELD_DEF(uint32_t, headNumRatio);
  TILING_DATA_FIELD_DEF(uint32_t, attenMaskElemType);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, pseMaskMaxSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, outputTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxTypeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
  TILING_DATA_FIELD_DEF(uint32_t, alignedHeadSize);
  TILING_DATA_FIELD_DEF(uint32_t, splitS2);
  TILING_DATA_FIELD_DEF(uint32_t, splitD);
  TILING_DATA_FIELD_DEF(uint32_t, layoutType);
  TILING_DATA_FIELD_DEF(uint32_t, PAlayoutType);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1Size);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2Size);
  TILING_DATA_FIELD_DEF(uint32_t, maskKVsSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskQsSize);
  TILING_DATA_FIELD_DEF(uint32_t, isLayoutSH);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsNull);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsKVNull);
  TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);
  TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);
  TILING_DATA_FIELD_DEF(uint32_t, deqScaleFlag);
  TILING_DATA_FIELD_DEF(uint32_t, deqScale2Flag);
  TILING_DATA_FIELD_DEF(uint32_t, isAntiPerchannel);
  TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxOuterSize);
  TILING_DATA_FIELD_DEF(uint32_t, isQuant2Perchannel);
  TILING_DATA_FIELD_DEF(uint32_t, isQuant2BF16);
  TILING_DATA_FIELD_DEF(uint32_t, isKvContinuous);
  TILING_DATA_FIELD_DEF(uint32_t, fromFused);
  TILING_DATA_FIELD_DEF(uint32_t, isBSNDOut);
  TILING_DATA_FIELD_DEF(uint32_t, isSoftMaxLseEnable);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSharedPrefixLenNull);
  TILING_DATA_FIELD_DEF(uint32_t, isQHasLeftPadding);
  TILING_DATA_FIELD_DEF(uint32_t, isKVHasLeftPadding);
  TILING_DATA_FIELD_DEF(int64_t, keyAntiquantMode);
  TILING_DATA_FIELD_DEF(int64_t, valueAntiquantMode);
  TILING_DATA_FIELD_DEF(uint32_t, hasKeyAntiquantOffset);
  TILING_DATA_FIELD_DEF(uint32_t, isMsd);
  TILING_DATA_FIELD_DEF(uint32_t, isQuant2FP16);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionBaseParamsOp, PromptAttentionBaseParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSeqParams)
  // Temporary reuse
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, CoreHeadNumTail);       // coreNStart
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, actualS1);              // coreNEnd
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, actualCoreNums);        // coreSidStart
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, singleCoreHeadNumSize); // coreSidEnd
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSeqPosStart);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSeqPosEnd);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSeqParamsOp, PromptAttentionSeqParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSingleCoreParams)
  TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
  TILING_DATA_FIELD_DEF(uint32_t, singleProcessSOuterSize);
  TILING_DATA_FIELD_DEF(uint32_t, multiSmaxsInnerLoopTimes);
  TILING_DATA_FIELD_DEF(uint32_t, actualCoreNums);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftBatch);
  TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatch);
  TILING_DATA_FIELD_DEF(uint32_t, kvAntiquantSInnerSize);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSingleCoreParamsOp, PromptAttentionSingleCoreParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSingleCoreTensorSize)
  TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, attenMaskUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxSumSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxExpSize);
  TILING_DATA_FIELD_DEF(uint32_t, softmaxValueSize);
  TILING_DATA_FIELD_DEF(uint32_t, spmTmpSize);
  TILING_DATA_FIELD_DEF(uint32_t, scmTmpSize);
  TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpMMResBmm2PreUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpSoftmaxBmm2UbSize);
  TILING_DATA_FIELD_DEF(uint32_t, selectSpaceUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, tmpSoftMaxV2Size);
  TILING_DATA_FIELD_DEF(uint32_t, mm1TmpUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, mm2TmpUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, kvAntiquantUbSize);
  TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbMsdSize);
  TILING_DATA_FIELD_DEF(uint32_t, tempBmm2QueueMsdSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdInQueueSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdQRowSumBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdAMaxTmpBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdAMaxResBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxResAmaxBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxRowSumScaleBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdScaleBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdOffsetBuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm1BuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm2BuffSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdOutQueueSize);
  TILING_DATA_FIELD_DEF(uint32_t, msdComputeLines);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSingleCoreTensorSizeOp, PromptAttentionSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(PromptAttentionInitOutputParams)
  TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);
  TILING_DATA_FIELD_DEF(int64_t, totalOutputSize);
  TILING_DATA_FIELD_DEF(int64_t, totalSoftMaxLseOutputSize);
  TILING_DATA_FIELD_DEF(uint32_t, needInit);
  TILING_DATA_FIELD_DEF(uint32_t, isOneN);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionInitOutputParamsOp, PromptAttentionInitOutputParams)

BEGIN_TILING_DATA_DEF(PromptFlashAttentionTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingDataRect);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingDataRect);

  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionBaseParams, promptAttentionBaseParams);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSeqParams, promptAttentionSeqParams);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSingleCoreParams, promptAttentionSingleCoreParams);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionSingleCoreTensorSize, promptAttentionTensorSizeRect);
  TILING_DATA_FIELD_DEF_STRUCT(PromptAttentionInitOutputParams, promptAttentionInitOutputParams);

  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingDataRect);
  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingDataRect);
  TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataRect);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PromptFlashAttention, PromptFlashAttentionTilingData)


class PromptFlashAttentionTiling {
public:
    PromptFlashAttentionTiling(fe::PlatFormInfos* platFormInfo): ascendcPlatform(platFormInfo) {}
    ge::graphStatus RunBigKernelTilingWithParams(ContextParamsForPFATiling& contextKeyParams,
                                                uint64_t& tilingKey, uint32_t& blockDimToBeSet,
                                                PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus PromptFlashAttentionSetTilingData(gert::TilingContext* context,
                                                    PromptFlashAttentionTilingData& tilingData);
    bool CheckNonEmptyShapeExceptions(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* shape,
                                      const std::string &sName);
protected:
    ge::graphStatus ConvertContextToPFAParams(gert::TilingContext* context, ContextParamsForPFATiling& contextKeyParams);
    ge::graphStatus TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey, ContextParamsForPFATiling& contextKeyParams,
                                                       bool useNewTiling, PromptFlashAttentionTilingData& tilingData);
    void PromptFlashAttentionSplitNS(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths);
    void PromptFlashAttentionSplitNSNew(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths,
                                                    std::vector<int64_t>& actualSeqLengthsKV, int64_t actualSharedPrefixLen, bool useBalanceTiling);
    void GetPreNextTokensLeftUp(PromptFlashAttentionTilingData& tilingData, uint32_t actualSeqLength, uint32_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
    void SetSplitCoreMode(PromptFlashAttentionTilingData& tilingData, uint32_t sOuterFactor);
    void PromptFlashAttentionSplitSeqOneN(PromptFlashAttentionTilingData& tilingData, uint32_t curCoreNum, bool isVectorCore);
    bool EnableMTE2BmmPipe(PromptFlashAttentionTilingData& tilingData, matmul_tiling::MatmulApiTiling& bmm,
                           TCubeTiling& bmmTilingData, uint32_t sOuterFactor, uint32_t sInnerFactor);
    void EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData);
    void PromptFlashAttention310PSetBmm1(matmul_tiling::MatmulApiTiling& bmm1);
    void PromptFlashAttention310PSetBmm2(matmul_tiling::MatmulApiTiling& bmm2);
    bool PromptFlashAttentionCheckBmm1(PromptFlashAttentionTilingData& tilingData, TCubeTiling& bmm1TilingData,
                                       int64_t l1SizeRemain, int64_t l0CSize,
                                       uint32_t sOuterFactor, uint32_t sInnerFactor,
                                       bool allGM = false, bool autoBaseMNK = false);
    bool PromptFlashAttentionCheckBmm2(PromptFlashAttentionTilingData& tilingData, TCubeTiling& bmm1TilingData,
                                       int64_t l1SizeRemain, int64_t l0CSize,
                                       uint32_t sOuterFactor, uint32_t sInnerFactor,
                                       uint32_t dSplitFactor, bool allGM = false, bool autoBaseMNK = false);
    void PromptFlashAttentionSetTensorSize(PromptFlashAttentionTilingData& tilingData,
                        PromptAttentionSingleCoreTensorSize& tensorSize, uint32_t sOuterFactor, uint32_t sInnerFactor);
    bool PromptFlashAttentionCheckArgsLegal(PromptFlashAttentionTilingData& tilingData, int64_t ubSize, int64_t l1Size,
                                            int64_t l0CSize, uint32_t typeByteSize, uint32_t& sOuterFactor,
                                            uint32_t sInnerFactor, bool& updateDiv, uint32_t maskTypeSize, uint32_t dSplitFactor);
    ge::graphStatus AdjustBasicBlock(PromptFlashAttentionTilingData& tilingData, uint32_t& sOuterFactor);
    ge::graphStatus PromptFlashAttentionApiTiling(PromptFlashAttentionTilingData& tilingData, uint32_t typeSize,
                                                  uint32_t sOuterFactor, uint32_t softmaxSInnerFactor, uint32_t softmaxSOuterFactor);
    ge::graphStatus GetRectangleFactor(uint32_t seqSplit, std::queue<uint32_t>& sQueue, int32_t threshold = 16);
    ge::graphStatus SetInputLayout(const char* layout);
    bool GetApiTmpSize(const uint32_t sOuterFactor, const uint32_t sInnerFactor,
                        const uint32_t typeByteSize);
    uint32_t CalculateL1SizeUsed(PromptFlashAttentionTilingData& tilingData, const uint32_t typeByteSize);
    bool CheckInputDimAndHeadNum(ContextParamsForPFATiling& contextKeyParams, uint32_t nQAttr, uint32_t nKVAttr);
    bool SetTilingHeadNumRatio(ContextParamsForPFATiling& contextKeyParams, const int32_t* numQueryHeads,
                               const int32_t* numKeyValueHeads, PromptFlashAttentionTilingData& tilingData);
    void PromptFlashAttentionInitOutputSplit(uint64_t totalSize, PromptFlashAttentionTilingData &tilingData,
                                             uint32_t curCoreNum);
    void PromptFlashAttentionInitSoftmaxLseOutputSplit(uint64_t totalSize, PromptFlashAttentionTilingData &tilingData);
    void Align(uint32_t &num);
    ge::graphStatus GetBasicShape(uint32_t &b, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                const gert::StorageShape *queryShape, const gert::StorageShape *keyShape, const uint32_t n);
    ge::graphStatus GetBasicShape310P(uint32_t &b, uint32_t &bKV, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                      const gert::StorageShape *queryShape, const gert::StorageShape *keyShape, const uint32_t n,
                                      size_t actualLenDims, size_t actualLenDimsKV);
    ge::graphStatus GetBasicShape910B(uint32_t &b, uint32_t &s, uint32_t &h, uint32_t &seqInnerSize,
                                      const gert::StorageShape *queryShape, const gert::StorageShape *keyShape, const uint32_t n);

    size_t GetPFAWorkSpaceSize(PromptFlashAttentionTilingData& tilingData);
    void GetMatMulType(matmul_tiling::DataType &mmInputType, matmul_tiling::DataType &mmOutputType);
    ge::graphStatus CheckKeyValueParamsConsistency(const ContextParamsForPFATiling& contextKeyParams);
    bool CheckActualSeqLength(ContextParamsForPFATiling& contextKeyParams, uint32_t b, uint32_t sQ, uint32_t sKV,
                              const gert::Tensor* actualSeqLenQ, const gert::Tensor* actualSeqLenKV, InputLayout inLayout, PromptFlashAttentionTilingData& tilingData);
    bool CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *pseShiftShape,
                                   uint32_t b, uint32_t n, uint32_t s1, uint32_t s2);
    bool CheckPATypeAndShape(ContextParamsForPFATiling& contextKeyParams, const gert::Tensor* actualSeqLenKV,
                                   int32_t b, int32_t n, int32_t h, int32_t headNumRatio);
    bool CheckAttenMaskShape(ContextParamsForPFATiling& contextKeyParams, const int32_t* sparseMode, const gert::StorageShape* attenMaskShape,
                             uint32_t sQ, uint32_t sK, uint32_t batchSize);
    bool CheckAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* antiquantScaleShape,
                                   const gert::StorageShape* antiquantOffsetShape, const uint32_t n, const uint32_t d, const uint32_t h,
                                   PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus CheckPostQuantParams(const ContextParamsForPFATiling& contextKeyParams, uint32_t h, uint32_t n) const;
    ge::graphStatus PromptFlashAttentionCVDiffSetTensorSize(PromptFlashAttentionTilingData& tilingData,
        PromptAttentionSingleCoreTensorSize& tensorSize, uint32_t sOuterFactor,
        uint32_t sInnerFactor, uint32_t softmaxSOuterFactor);
    bool PromptFlashAttentionComputeCVDiffParams(PromptFlashAttentionTilingData& tilingData,
        int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t typeByteSize,
        uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t maskTypeSize, uint32_t &softmaxSOuterFactor);
    bool FindOptimalTilingBasicBLock(PromptFlashAttentionTilingData& tilingData,
        uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
        int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize);
    bool FindOptimalTilingSouter(PromptFlashAttentionTilingData& tilingData,
        uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
        int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize);
    void InferTilingMod(const ContextParamsForPFATiling& contextKeyParams, const std::vector<int64_t>& actualSeqLengths, const std::vector<int64_t>& actualSeqLengthsKV,
        uint32_t actualSeqArrayLen, uint32_t hDivN, uint32_t seqInnerSize, int32_t sparseModeVal);
    ge::graphStatus AdjustCVTiling(uint32_t hDivN, uint32_t n, int64_t middle_actualSeqLengths,
        int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t maskElemSize,
        uint32_t& sOuterFactor, uint32_t& sInnerFactor, PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus AdjustCVTilingCVDiff(int64_t ubSize, int64_t l1Size, int64_t l0CSize,
        uint32_t maskElemSize, uint32_t& sOuterFactor, uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor,
        PromptFlashAttentionTilingData& tilingData);
    bool CheckSparseModeRightDown(ContextParamsForPFATiling& contextKeyParams, const std::vector<int64_t>& actualSeqLengths,
                                  const std::vector<int64_t>& actualSeqLengthsKV, size_t lenDims);
    ge::graphStatus GetAndCheckEmptyQueryShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *queryShape) const;
    void UpdateTilingKeyFlag(ContextParamsForPFATiling& contextKeyParams, uint64_t& tilingKey);
    int64_t PromptFlashAttentionSetMsdUbSize(PromptFlashAttentionTilingData& tilingData, PromptAttentionSingleCoreTensorSize& tensorSize, int32_t sInnerFactorTmp) const;
  
    ge::graphStatus CheckIOType(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, int32_t& outputDataTypeSize);
    ge::graphStatus CheckMaskType(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, uint32_t& maskElemSize);
    void SetMaskSize(const gert::StorageShape* attenMaskShape, PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus CheckShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* queryShape, const gert::StorageShape* keyShape, 
                               const gert::StorageShape* valueShape, const gert::StorageShape* outShape, const gert::StorageShape* pseShiftShape,
                               const gert::StorageShape* attenMaskShape);

    protected:
        ContextParamsForPFATiling* contextKeyParamsPtr = nullptr;
        int64_t ubSizeRemain = 1;
        bool isSOuterNoTail = true;
        bool isSInnerNoTail = true;
        bool isDNoTail = true;
        bool enableKvAntiquant = false;
        bool enableMsd = false;
        bool enableQuantBF16 = false;
        bool enableMatmulNorm = false;
        bool enablePA = false;
        bool isKVHasPrefix = false;
        InputLayout inputLayout = InputLayout::BSH;
        ge::DataType inputType{ge::DT_FLOAT16};
        ge::DataType outputType{ge::DT_FLOAT16};
        ge::DataType pseShiftElemType{ge::DT_FLOAT16};
        uint32_t dataTypeSize = FLOAT32SIZE;
        uint32_t coreNum;
        uint32_t aivNum;
        uint32_t aicNum;
        uint32_t typeByteNum;
        uint32_t outputTypeByteNum;
        uint32_t softmaxTypeByteNum;
        uint32_t pseShiftTypeByteNum = 0;
        uint32_t pseShiftElemSize = 0;
        uint32_t pseMaskMaxSize = 0;
        uint32_t pseShiftBatch = 0;
        uint32_t pseShiftS1 = 0;
        uint32_t pseShiftS2 = 0;
        uint32_t usePseShift = 0;
        uint32_t tmpS2 = 0;  // In the PA scenario, there is no S2 axis. Use the change amount to normalize the S2 length in both PA and non PA scenarios
        int32_t blockTableDim2 = 1;
        int32_t PABlockNumSum = 1;
        uint32_t maskTypeByteNum;
        uint32_t maxQuerySeq = 0;
        int64_t apiTmpSize = 1;
        uint32_t softmaxDataTypeNZ_ = FLOAT32SIZE;
        uint32_t softmaxDataTypeSize = FLOAT32SIZE; // BF16 calculates through FP32
        platform_ascendc::SocVersion curShortSocName;
        uint32_t dataTypeSize_ = 4;
        uint32_t layoutType = 0;
        uint32_t PAlayoutType = 0;
        platform_ascendc::PlatformAscendC ascendcPlatform;
        TilingMod tilingMod = TilingMod::CVSAME;
        SplitCoreMode splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;
        uint32_t splitD = 0;
        uint32_t splitS2 = 1; // It can only be 0 when the D axis is split
        uint64_t innerPrecise = HIGH_PERFORMANCE;
        size_t defaultSysWorkspaceSize;
        matmul_tiling::PlatformInfo ascendPlatformInfo;
    };
    // end of class PromptFlashAttention
  PFA_EXTERN_C ge::graphStatus TilingPromptFlashAttention(gert::TilingContext* context);
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_H_
