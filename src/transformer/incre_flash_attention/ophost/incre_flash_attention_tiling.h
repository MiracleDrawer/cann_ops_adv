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
 * \file incre_flash_attention_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_H_

#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"

const uint32_t MAX_CORE_NUM = 50;
const uint32_t MAX_SIZE_BATCH = 256U;

namespace optiling {

BEGIN_TILING_DATA_DEF(IncreFlashAttentionInitOutputParams)
TILING_DATA_FIELD_DEF(uint32_t, isPerChnOut)
TILING_DATA_FIELD_DEF(uint32_t, isOutQuantTypeBf16)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionInitOutputParamsOp, IncreFlashAttentionInitOutputParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, headSize)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerSeq)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, qHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, batchContinuousFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftB)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS)
TILING_DATA_FIELD_DEF(uint32_t, selectWithByteMaskTmpMinSize)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDims)
TILING_DATA_FIELD_DEF(uint32_t, kvPaddingFlag)
TILING_DATA_FIELD_DEF(uint32_t, msdIterNum)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
TILING_DATA_FIELD_DEF(uint32_t, antiquantPerTensorFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskSize)
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, totalBlockNum)
TILING_DATA_FIELD_DEF(uint32_t, paKvShapeType)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionBaseParamsOp, IncreFlashAttentionBaseParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionCoreParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSidxEnd);  // 50:MAX_CORE_NUM
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionCoreParamsOp, IncreFlashAttentionCoreParams);

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSingleCoreParamsOp, IncreFlashAttentionSingleCoreParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSingleCoreTensorSize)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSingleCoreTensorSizeOp, IncreFlashAttentionSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSplitKVParams)
TILING_DATA_FIELD_DEF(uint32_t, s2)
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopSize)
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSplitKVParamsOp, IncreFlashAttentionSplitKVParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSplitKVParams, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionCoreParams, increFlashAttentionCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSingleCoreParams, increFlashAttentionSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSingleCoreTensorSize, increFlashAttentionSingleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionInitOutputParams, outputParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionTilingDataOp, IncreFlashAttentionTilingData)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingDataPrefix)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTilingData, base);
TILING_DATA_FIELD_DEF(uint64_t, prefixAttenOutOffset);  // 临时输出偏移
TILING_DATA_FIELD_DEF(uint64_t, userPromptAttenOutOffset);
TILING_DATA_FIELD_DEF(uint64_t, tmpLseOffset);
TILING_DATA_FIELD_DEF(uint64_t, prefixLen);  // prefix 长度
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);  // combine 分核参数，参考普通bn分核流程，总数不超过blockdim
TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, batchSizeQ);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionTilingDataPrefixOp, IncreFlashAttentionTilingDataPrefix)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingDataV2)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTilingData, tilingBase);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTilingDataPrefix, tilingPrefix);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttention, IncreFlashAttentionTilingDataV2)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionEmptyInputTilingData)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionInitOutputParams, outputParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_13, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_14, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_27, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30, IncreFlashAttentionEmptyInputTilingData)

struct IncreFlashAttentionCompileInfo {
  int64_t core_num;
};

struct RequiredParaInfo {
  const gert::CompileTimeTensorDesc* desc;
  const gert::StorageShape* shape;
};

struct OptionalParaInfo {
  const gert::CompileTimeTensorDesc* desc;
  const gert::Tensor* tensor;
};

struct IncreFlashAttentionContext {
  const char* opName;
  fe::PlatFormInfos* platformInfo;
  RequiredParaInfo query;
  RequiredParaInfo key;
  RequiredParaInfo value;
  OptionalParaInfo pseShift;
  OptionalParaInfo attenMask;
  OptionalParaInfo actualSeqLengths;
  OptionalParaInfo deqScale1;
  OptionalParaInfo quantScale1;
  OptionalParaInfo deqScale2;
  OptionalParaInfo quantScale2;
  OptionalParaInfo quantOffset2;
  OptionalParaInfo antiquantScale;
  OptionalParaInfo antiquantOffset;
  OptionalParaInfo blockTable;
  OptionalParaInfo kvPaddingSize;
  OptionalParaInfo keyAntiquantScale;
  OptionalParaInfo keyAntiquantOffset;
  OptionalParaInfo valueAntiquantScale;
  OptionalParaInfo valueAntiquantOffset;
  OptionalParaInfo keySharedPrefix;
  OptionalParaInfo valueSharedPrefix;
  OptionalParaInfo actualSharedPrefixLen;
  RequiredParaInfo attenOut;
  const uint32_t* numHeads;
  const float* scaleValue;
  const uint32_t* kvHeadNums;
  const char* layOut;
  const uint32_t* blockSize;
  const uint32_t* innerPrecise;
  const uint32_t* antiquantMode;
  const bool* softmaxLseFlag;
  const uint32_t* keyAntiquantMode;
  const uint32_t* valueAntiquantMode;
  size_t* workSpaces;
  std::vector<gert::StorageShape*> kCache;
  std::vector<gert::StorageShape*> vCache;
  uint64_t tilingKey;
  uint32_t blockDim;
};

enum class TilingInOutMode : uint32_t {
  IO_INVALID = 0,
  INT8_INT8 = 1,
  FP16_INT8 = 2,
  INT8_FP16 = 3,
  FP16_FP16 = 4,
  BF16_BF16 = 5,
  FP32_FP32 = 6,
  FP16_FP16_SPLITKV = 7,
  BF16_INT8 = 8,
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},            // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                    // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},                // fp16 type
    {ge::DT_INT8, "DT_INT8"},                      // int8 type
    {ge::DT_INT16, "DT_INT16"},                    // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                  // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                    // uint8 type
    {ge::DT_INT32, "DT_INT32"},                    // uint32 type
    {ge::DT_INT64, "DT_INT64"},                    // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                  // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                  // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                      // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                  // double type
    {ge::DT_DUAL, "DT_DUAL"},                      // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},    // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},  // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},            // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},            // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},          // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                    // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                  // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                  // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                  // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},                // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},              // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},          // string ref type
    {ge::DT_STRING, "DT_STRING"},                  // string type
    {ge::DT_VARIANT, "DT_VARIANT"},                // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                  // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                      // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                    // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                      // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                     // dt_variant type
};

enum class IfaPerfMode : uint32_t { NORMAL = 0, BMM_ALL_BY_VEC, C1_V1 };

enum IfaSocVersion : uint32_t {
  SOC_ASCEND_910B = 0,
  SOC_ASCEND_310P = 1,
};

enum IfaLayout : uint32_t {
  BSH_BSND = 0,
  BNSD = 1,
};

enum kvCacheLayout : uint32_t {
  KV_CACHE_BSH = 0,
  KV_CACHE_BNSD = 1,
};

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t NUM_BYTES_BOOL = 1;
constexpr uint32_t NUM_BYTES_INT8 = 1;
constexpr uint32_t MAX_MATMUL_BASE = 512;
constexpr uint32_t MATMUL_BASE_N = 256;
constexpr uint32_t MAX_MATMUL_BASE_M = 128;
constexpr uint32_t MAX_SPLIT_SIZE = 8192;
constexpr uint32_t L0B_SIZE = 64 * 1024;
constexpr uint32_t L0C_SIZE = 128 * 1024;
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t PSE_SHIFT_INPUT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INPUT_INDEX = 4;
constexpr uint32_t ACT_SEQ_LEN_INPUT_INDEX = 5;
constexpr uint32_t DEQUANT_SCALE_1_INPUT_INDEX = 6;
constexpr uint32_t QUANT_SCALE_1_INPUT_INDEX = 7;
constexpr uint32_t DEQUANT_SCALE_2_INPUT_INDEX = 8;
constexpr uint32_t QUANT_SCALE_2_INPUT_INDEX = 9;
constexpr uint32_t QUANT_OFFSET_2_INPUT_INDEX = 10;
constexpr uint32_t ANTIQUANT_SCALE_INPUT_INDEX = 11;
constexpr uint32_t ANTIQUANT_OFFSET_INPUT_INDEX = 12;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 13;
constexpr uint32_t KV_PADDING_SIZE_INPUT_INDEX = 14;
constexpr uint32_t DIM_BNSD = 4;
constexpr uint32_t DIM_BNSD_OR_BNSD = 4;
constexpr uint32_t DIM_BSH = 3;
constexpr uint32_t BNSD_B_IDX = 0;
constexpr uint32_t BNSD_N_IDX = 1;
constexpr uint32_t BNSD_S_IDX = 2;
constexpr uint32_t BNSD_D_IDX = 3;
constexpr uint32_t BSND_B_IDX = 0;
constexpr uint32_t BSND_S_IDX = 1;
constexpr uint32_t BSND_N_IDX = 2;
constexpr uint32_t BSND_D_IDX = 3;
constexpr uint32_t BSH_B_IDX = 0;
constexpr uint32_t BSH_S_IDX = 1;
constexpr uint32_t BSH_H_IDX = 2;
constexpr uint32_t DIM_BH = 2;
constexpr uint32_t BH_B_IDX = 0;
constexpr uint32_t BH_H_IDX = 1;
constexpr uint32_t OUTPUT_INDEX = 0;
constexpr uint32_t NUM_HEADS_ATTR_INDEX = 0;
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 1;
constexpr uint32_t LAYOUT_ATTR_INDEX = 2;
constexpr uint32_t KV_NUM_HEADS_ATTR_INDEX = 3;
constexpr uint32_t BLOCK_SIZE_ATTR_INDEX = 4;
constexpr uint32_t INNER_PRECISE_ATTR_INDEX = 5;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t PSE_SHIFT_B = 0;
constexpr uint32_t PSE_SHIFT_N = 1;
constexpr uint32_t PSE_SHIFT_S0 = 2;
constexpr uint32_t PSE_SHIFT_S1 = 3;
constexpr uint32_t ITER_NUM = 2;
constexpr uint32_t HIGH_PRECISION_ITER_NUM = 3;  // 高精度场景的迭代次数
constexpr uint32_t IFA_HIGH_PRECISION = 0;
constexpr uint32_t IFA_HIGH_PERFORMANCE = 1;
constexpr int64_t MSD_VEC_LOAD = 1024;
constexpr uint32_t MAX_BLOCK_SIZE = 512;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;

class IFATiling {
 public:
  IFATiling() = default;
  ~IFATiling() = default;

  ge::graphStatus DoTiling(gert::TilingContext& context);
  ge::graphStatus RunBigKernelTiling(IncreFlashAttentionContext& context, IncreFlashAttentionTilingDataV2& tilingData,
                                     bool isWorkspace = false);
  ge::graphStatus IncreFlashAttentionSetTilingData(gert::TilingContext& context,
                                                   IncreFlashAttentionTilingDataV2& tilingData);
  static ge::graphStatus ConvertContext(gert::TilingContext& context, IncreFlashAttentionContext& ifaContext);
  bool NeedRollBack() {
    return passToOldTiling_;
  }

 private:
  ge::graphStatus GetNpuInfo();
  ge::graphStatus PreProcess();
  ge::graphStatus ProcessBaseTensors();
  ge::graphStatus ProcessOptionalTensors();
  ge::graphStatus ProcessPseShift();
  ge::graphStatus ProcessAttenMask();
  ge::graphStatus ProcessActualSeqLen();
  ge::graphStatus ProcessQuant1();
  ge::graphStatus ProcessQuant2();
  ge::graphStatus ProcessDequant1();
  ge::graphStatus ProcessDequant2();
  ge::graphStatus ProcessAntiQuant();
  ge::graphStatus ProcessBlockTable();
  ge::graphStatus ProcessKVPaddingSize();
  ge::graphStatus ProcessSharedPrefix();
  ge::graphStatus ProcessSharedPrefixLen();
  void SetupPerfMode();
  bool EnableAllVec();
  bool EnableC1V1();
  void UpdatePerfMode();

  ge::graphStatus InitInOutMode();
  ge::graphStatus KvShapePostProcess();
  ge::graphStatus CheckKVShape();
  ge::graphStatus CheckQKOutShape();
  ge::graphStatus CheckKeyShapeTensor(const gert::Shape& aShape);
  ge::graphStatus ZeroTensorProcess();
  ge::graphStatus SharedPrefixTiling();

  ge::graphStatus CheckUbSpace();
  ge::graphStatus CheckPABlockSize();
  ge::graphStatus SetL2CacheFlag();

  ge::graphStatus CheckQuant2Shape(const gert::Shape& inputParaShape);
  ge::graphStatus ProcessQuant2Dtype();
  ge::graphStatus CheckKVAntiQuantParaShapeLegal(const gert::Shape& inputParaShape);
  ge::graphStatus CheckAntiQuantParam(const gert::Tensor* antiquantScaleTensor,
                                      const gert::Tensor* antiquantOffsetTensor,
                                      const gert::CompileTimeTensorDesc* antiquantScaleDesc,
                                      const gert::CompileTimeTensorDesc* antiquantOffsetDesc);
  ge::graphStatus CheckSupportKVLeftPadding();
  ge::graphStatus CheckInputFormatAndLimits();
  bool CalcUbBmm();
  bool CalcUbSoftMax();
  bool CalcUbAttenMask();
  bool CalcUbQuant();
  bool CalcUbDeQuant();
  bool CalcUbAntiQuant();
  bool CalcUbPageAttention();
  bool CalcUbKvSplit();

  bool CheckIfRollBack();
  bool CanChangeToNew();
  void AdjustPABmm1Tiling(uint32_t& bmm1BaseN);
  void AdjustPABmm2Tiling() const;
  bool ShapeEqual(const gert::Shape& aShape, const gert::Shape& bShape);

  ge::graphStatus Split();
  ge::graphStatus CalcInnerSize(uint32_t seqSize);
  ge::graphStatus SplitBN();

  std::vector<int64_t> InitSparseValidArray(const int64_t* actualLens);
  bool BalanceLoad(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                   std::vector<int64_t>& localValue, std::vector<int64_t>& sparseStartIdx);
  void InitLoadValue(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                     const std::vector<int64_t>& sparseStartIdx, std::vector<int64_t>& localValue);
  void SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                         uint32_t* sparseStartIdx, int64_t splitFactorSize);

  bool IsFlashDecode() const;
  ge::graphStatus SplitBN_V0();
  ge::graphStatus SplitBNS();

  bool CheckWorkSpace();
  bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType* mmType);

  ge::graphStatus CalcWorkSpace();
  ge::graphStatus CalcBlockDim();
  ge::graphStatus GenTilingKey();

  ge::graphStatus FillTiling();
  void FillTilingBaseParams();
  void FillTilingSplitKV();
  void FillTilingCoreParams();
  void FillTilingSingleCoreParams();
  void FillTilingSingleCoreTensorSize();
  void FillTilingSoftmax();
  void FillTilingSoftmaxFlashTiling();
  void FillTilingTranspose();
  void FillTilingOutputParams();
  bool FillTilingBmm();  // may fail

  ge::graphStatus CalcSysPrefixWorkSpace();
  ge::graphStatus FillSysPrefixTiling();
  ge::graphStatus CalcSysPrefixBlockDim();
  ge::graphStatus SplitForLseCombine();

 private:
  bool passToOldTiling_ = false;
  uint32_t numHeads_ = 0;
  float scaleValue_ = 0;
  uint32_t numKvHeads_ = 0;
  uint32_t blockSize_ = 0;
  uint32_t innerPrecise_ = 0;
  uint32_t nNumOfQInOneGroup_ = 1;
  uint32_t msdIterNum_ = 1;
  uint32_t antiquantMode_ = 0;
  uint32_t antiquantPerTensorFlag_ = 0;

  uint32_t headDim_ = 0;
  uint32_t seqSize_ = 0;
  uint32_t batchSize_ = 0;
  IfaLayout inputLayout_ = IfaLayout::BSH_BSND;
  uint32_t sMax_ = 0;
  uint32_t blockTypeSize_ = 0;  // 计算中间量大小
  uint32_t kvSplitPart_ = 1;

  uint32_t sMaxPrefix_ = 0;
  uint32_t maxActualPrefixLen_ = 0;

  ge::DataType inputQType_ = ge::DT_FLOAT16;
  ge::DataType inputKvType_ = ge::DT_FLOAT16;
  ge::DataType outputType_ = ge::DT_FLOAT16;

  size_t ubSize_ = 0;
  size_t l1Size_ = 0;
  size_t l0cSize_ = 0;
  size_t l0bSize_ = 0;
  uint32_t coreNum_ = 0;
  uint32_t aicNum_ = 0;
  uint32_t aivNum_ = 0;
  IfaSocVersion socVersion_ = IfaSocVersion::SOC_ASCEND_910B;
  size_t libapiSize_ = 0;

  size_t mmResUbSize_ = 0;
  size_t bmm2ResUbSize_ = 0;

  size_t softmaxFlashTmpSize_ = 0;
  size_t softmaxTmpSize_ = 0;
  size_t softMaxSize_ = 0;

  size_t selectWithByteMaskTmpMinSize_ = 0;

  bool pseShiftFlag_ = false;
  uint32_t pseShiftTypeSize_ = NUM_BYTES_FLOAT16;
  uint32_t pseShiftBatch_ = 0U;
  uint32_t pseShiftS1_ = 0U;

  bool attenMaskFlag_ = false;
  uint32_t attenMaskSize_ = 0;
  uint32_t attenMaskTypeSize_ = 0;

  bool antiQuantFlag_ = false;
  size_t antiquantUb_ = 0;
  bool kvAntiParamSplitFlag_ = false;

  bool pageAttentionFlag_ = false;
  uint32_t pageAttentionKvLayoutType_ = kvCacheLayout::KV_CACHE_BSH;  // pa场景下kv的shape, 0:BSH 1:BNSD
  uint32_t maxBlockNumPerSeq_ = 0;
  size_t kvPageResUbSize_ = 0;
  uint32_t totalBlockNum_ = 0;

  bool batchContinuousFlag_ = true;
  std::vector<int64_t> kvListSeqLens_;

  bool actualSeqLenFlag_ = false;
  bool kvPaddingSizeFlag_ = false;

  bool quantFlag_ = false;
  size_t quantUbSize_ = 0;

  uint32_t actualLenDims_ = 0;
  uint32_t maxActualseq_ = 0;

  // flash config
  uint32_t sInnerLoopTimes_ = 0;
  uint32_t sInnerSize_ = 0;  // flash attention
  uint32_t sInnerSizeTail_ = 0;
  uint32_t sInnerSizeAlign_ = 0;
  uint32_t headDimAlign_ = 0;
  // uint32_t sOuterSize_;  // flash decode s2

  bool isSplitBPolicy_ = false;
  bool splitKVFlag_ = false;
  uint32_t kvSplit_ = 0;
  bool splitKVFlagPrefix_ = false;

  IfaPerfMode perfMode_ = IfaPerfMode::NORMAL;
  TilingInOutMode inOutMode_ = TilingInOutMode::FP16_FP16;
  size_t workspaceSize_ = 0;

  uint32_t taskRation_ = 0;
  uint32_t usedCoreNum_ = 0;

  uint32_t startIdxEachCore_[MAX_CORE_NUM] = {};
  IncreFlashAttentionContext* context_ = nullptr;
  IncreFlashAttentionTilingData* tilingData_ = nullptr;
  IncreFlashAttentionTilingDataPrefix* tilingDataPrefix_ = nullptr;
  bool isWorkspace_ = false;

  uint32_t formerCoreNum_ = 0;
  uint32_t blockSplitBn2Range_ = 0;
  uint32_t tailSplitedBatchRange_ = 0;

  uint32_t l2CacheOffFlag_ = 0;
  // softmaxLse
  bool softmaxLseFlag_ = false;

  bool sysPrefixFlag_ = false;
  bool isSysPrefixTiling_ = false;
  uint32_t batchSizeQ_ = 1;
  uint32_t actualLenDimsPrefix_ = 0;

  uint64_t prefixAttenOutOffset_ = 0;
  uint64_t userPromptAttenOutOffset_ = 0;
  uint64_t tmpLseOffset_ = 0;

  uint32_t formerCoreNumSp_ = 0;
  uint32_t blockSplitBn2RangeSp_ = 0;
  uint32_t tailSplitedBatchRangeSp_ = 0;
  uint32_t combinUsedCore_ = 0;
};

std::string DataTypeToSerialString(ge::DataType type);

ge::graphStatus TilingIncreFlashAttentionAdapter(gert::TilingContext* context, IncreFlashAttentionContext& ifaContext,
                                                 IncreFlashAttentionTilingDataV2& ifaTilingData);

ge::graphStatus TilingIncreFlashAttention(gert::TilingContext* context);

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_H_
