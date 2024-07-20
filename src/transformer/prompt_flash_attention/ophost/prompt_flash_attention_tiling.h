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

namespace optiling { 

constexpr uint32_t INT8SIZE = 1;
constexpr uint32_t UINT8SIZE = 1;
constexpr uint32_t FLOAT16SIZE = 2;
constexpr uint32_t BFLOAT16SIZE = 2;
constexpr uint32_t FLOAT32SIZE = 4;
constexpr uint32_t BOOLSIZE = 1;

constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;

const uint32_t MAX_BATCH = 256U;
struct PromptFlashAttentionCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    size_t defaultSysWorkspaceSize;
    platform_ascendc::SocVersion socShortName;
};

/*
contextParams is a new structured defined for the use of FusedInferAttentionScore op.
It is meant to catch and organize all the necessary variables passed by FIAS tilling function.
It will be used as the input to the new 'runBigKernelWithParams' function in PFA tilling.
The old PFA tillingContext will also be transformed to this structure in the future.
*/
struct ContextParamsForPFATiling
{
    const gert::Tensor* pseShift;
    const gert::Tensor* attentionMask;
    const gert::Tensor* actualSeqenceLengthQ;
    const gert::Tensor* actualSeqenceLengthKV;
    const gert::Tensor* antiquantScale;
    const gert::Tensor* antiquantOffset;
    const gert::Tensor* queryPaddingSize;
    const gert::Tensor* kvPaddingSize;
    ge::DataType inputDataType;
    ge::DataType kDataType;
    ge::DataType vDataType;
    ge::DataType pseShiftDataType;
    ge::DataType maskDataType;
    ge::DataType outputDataType;
    const char* opName;
    const gert::StorageShape* queryInputShape;
    const gert::StorageShape* keyInputShape;
    const gert::StorageShape* valueInputShape;
    const gert::StorageShape* pseShiftShape;
    const gert::StorageShape* attentionMaskShape;
    const gert::StorageShape* deqScale1Shape;
    const gert::StorageShape* scale1Shape;
    const gert::StorageShape* deqScale2Shape;
    const gert::StorageShape* scale2Shape;
    const gert::StorageShape* offset2Shape;
    const gert::StorageShape* antiquantScaleShape;
    const gert::StorageShape* antiquantOffsetShape;
    const gert::StorageShape* outputShape;
    const gert::StorageShape* lseoutputShape;
    const int64_t* innerPrecisePtr;
    const int32_t* headsNumber;
    const int32_t* sparseMode;
    const int32_t* preToken;
    const int32_t* nextToken;
    const float* scaleValue;
    const char* layout;
    const int32_t* numKeyValueHeads;
    size_t* workspaceSize;
    const PromptFlashAttentionCompileInfo* compileInfoPtr;
    ge::DataType deqScaleType;
    ge::DataType deqScale2Type;
    ge::DataType quantScale2Type;
    ge::DataType quantOffset2Type;
    uint32_t isKvContinuous;
    std::vector<const gert::StorageShape*> kTensorList;
    std::vector <const gert::StorageShape*> vTensorList;
    uint32_t maxKVs;
    uint32_t fromFused;
    uint32_t emptyTensor;
    uint32_t isBSNDOut;
    const bool *softmaxLseFlag;
    bool isSoftMaxLseEnable;
    uint32_t fromTilingSink;    // 表明是否是“从tiling下沉中计算workspace的步骤进入”的flag
};

BEGIN_TILING_DATA_DEF(PromptAttentionBaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, headNumSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqSize);
  TILING_DATA_FIELD_DEF(uint32_t, headSize);
  TILING_DATA_FIELD_DEF(float, scaleValue);
  TILING_DATA_FIELD_DEF(int32_t, preTokens);
  TILING_DATA_FIELD_DEF(int32_t, nextTokens);
  TILING_DATA_FIELD_DEF(uint32_t, dimNumOfseq);
  TILING_DATA_FIELD_DEF(uint32_t, typeByteNum);
  TILING_DATA_FIELD_DEF(uint32_t, seqInnerSize);
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
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1Size);
  TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2Size);
  TILING_DATA_FIELD_DEF(uint32_t, maskKVsSize);
  TILING_DATA_FIELD_DEF(uint32_t, maskQsSize);
  TILING_DATA_FIELD_DEF(uint32_t, isLayoutSH);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsNull);
  TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsKVNull);
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
  TILING_DATA_FIELD_DEF(uint32_t, isQHasLeftPadding);
  TILING_DATA_FIELD_DEF(uint32_t, isKVHasLeftPadding);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionBaseParamsOp, PromptAttentionBaseParams)

BEGIN_TILING_DATA_DEF(PromptAttentionSeqParams)
  // 临时复用
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
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptAttentionSingleCoreTensorSizeOp, PromptAttentionSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(PromptAttentionInitOutputParams)
  TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);
  TILING_DATA_FIELD_DEF(int64_t, totalOutputSize);
  TILING_DATA_FIELD_DEF(int64_t, totalSoftMaxLseOutputSize);
  TILING_DATA_FIELD_DEF(uint32_t, needInit);
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

enum class InputLayout{ SH, BSH, BNSD, NSD, BSND, BNSD_BSND, NONE, };

enum class TilingMod {
    CVSAME = 0,
    CVDIFF,
};

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
                                                       uint32_t coreNum, bool useNewTiling, PromptFlashAttentionTilingData &tilingData);
    ge::graphStatus PromptFlashAttentionSplitNS(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, uint32_t coreNum, int64_t *actualSeqLengths);
    ge::graphStatus PromptFlashAttentionSplitNSNew(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, uint32_t curCoreNum, int64_t *actualSeqLengths,
                                                    int64_t *actualSeqLengthsKV, bool useBalanceTiling);
    void GetPreNextTokensLeftUp(PromptFlashAttentionTilingData& tilingData, uint32_t actualSeqLength, uint32_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp);
    bool EnableSplitSeqOneN(PromptFlashAttentionTilingData& tilingData, const ContextParamsForPFATiling& contextKeyParams);
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
    ge::graphStatus PromptFlashAttentionSetTensorSize(PromptFlashAttentionTilingData& tilingData,
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
                                             uint32_t coreNum);
    void PromptFlashAttentionInitSoftmaxLseOutputSplit(uint64_t totalSize, PromptFlashAttentionTilingData &tilingData,
                                             uint32_t coreNum);
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
                              const gert::Tensor* actualSeqLenQ, const gert::Tensor* actualSeqLenKV, InputLayout inLayout);
    bool CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *pseShiftShape,
                                   uint32_t b, uint32_t n, uint32_t s1, uint32_t s2);
    bool CheckAttenMaskShape(ContextParamsForPFATiling& contextKeyParams, const int32_t* sparseMode, const gert::StorageShape* attenMaskShape,
                             uint32_t sQ, uint32_t sK, uint32_t batchSize);
    bool CheckAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* antiquantScaleShape,
                                   const gert::StorageShape* antiquantOffsetShape, const uint32_t n, const uint32_t d, const uint32_t h,
                                   PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus CheckPostQuantParams(const ContextParamsForPFATiling& contextKeyParams, uint32_t h, uint32_t n);
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
    void InferTilingMod(const ContextParamsForPFATiling& contextKeyParams, const int64_t actualSeqLengths[], const int64_t actualSeqLengthsKV[],
        uint32_t actualSeqArrayLen, uint32_t hDivN, uint32_t seqInnerSize, int32_t sparseModeVal);
    ge::graphStatus AdjustCVTiling(uint32_t hDivN, uint32_t n, int64_t middle_actualSeqLengths,
        int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t maskElemSize,
        uint32_t& sOuterFactor, uint32_t& sInnerFactor, PromptFlashAttentionTilingData& tilingData);
    ge::graphStatus AdjustCVTilingCVDiff(int64_t ubSize, int64_t l1Size, int64_t l0CSize,
        uint32_t maskElemSize, uint32_t& sOuterFactor, uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor,
        PromptFlashAttentionTilingData& tilingData);
    bool CheckSparseModeRightDown(ContextParamsForPFATiling& contextKeyParams, const int64_t *actualSeqLengths,
                                  const int64_t *actualSeqLengthsKV, size_t lenDims);
    ge::graphStatus GetAndCheckEmptyQueryShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *queryShape) const;
    void UpdateTilingKeyFlag(ContextParamsForPFATiling& contextKeyParams, uint64_t& tilingKey);
    protected:
        ContextParamsForPFATiling* contextKeyParamsPtr = nullptr;
        int64_t ubSizeRemain = 1;
        bool isSOuterNoTail = true;
        bool isSInnerNoTail = true;
        bool isDNoTail = true;
        bool enableKvAntiquant = false;
        bool enableQuantBF16 = false;
        bool enableMatmulNorm = false;
        bool enableSplitSeqOneN = false;
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
        uint32_t maskTypeByteNum;
        uint32_t maxQuerySeq = 0;
        int64_t apiTmpSize = 1;
        uint32_t softmaxDataTypeNZ_ = FLOAT32SIZE;
        uint32_t softmaxDataTypeSize = FLOAT32SIZE; // bf16通过fp32来进行计算
        platform_ascendc::SocVersion curShortSocName;
        uint32_t dataTypeSize_ = 4;
        uint32_t layoutType = 0;
        platform_ascendc::PlatformAscendC ascendcPlatform;
        TilingMod tilingMod = TilingMod::CVSAME;
        uint32_t splitD = 0;
        uint32_t splitS2 = 1; // 在D轴切分时才可能为0
        uint64_t innerPrecise = HIGH_PERFORMANCE;
        size_t defaultSysWorkspaceSize;
        matmul_tiling::PlatformInfo ascendPlatformInfo;
    };
    // end of class PromptFlashAttention
  ge::graphStatus TilingPromptFlashAttention(gert::TilingContext* context);
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_PROMPTFLASHATTENTION_H_
