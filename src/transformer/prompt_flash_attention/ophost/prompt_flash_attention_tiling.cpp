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
 * \file prompt_flash_attention_tiling.cpp
 * \brief
 */
#include <queue>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/data_copy_transpose_tiling.h"
#include "error/ops_error.h"
#include "prompt_flash_attention_tiling.h"

#include <cstdint>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>
#include <stdio.h>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "register/tilingdata_base.h"

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32; // datacopy的block块大小，datacopy按block块粒度搬移数据
constexpr uint32_t SOFTMAX_BUFFER_NUM = 3;

constexpr uint32_t NUM_2 = 2;
constexpr uint32_t INDEX_2 = 2;
constexpr uint32_t INDEX_3 = 3;
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t PSE_SHIFT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint32_t DEQ_SCALE1_INDEX = 7;
constexpr uint32_t QUANT_SCALE1_INDEX = 8;
constexpr uint32_t DEQ_SCALE2_INDEX = 9;
constexpr uint32_t QUANT_SCALE2_INDEX = 10;
constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;

constexpr uint32_t INPUT_QKV_SHAPE_MIN_DIMS = 2;
constexpr uint32_t INPUT_QKV_SHAPE_MAX_DIMS = 4;

constexpr uint32_t ATTR_N_INDEX = 0;
constexpr uint32_t ATTR_SCALE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;

constexpr uint64_t EMPTY_KV_TILING_KEY = 20;
constexpr uint32_t LOOP_BEGIN_NUM = 0;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr uint32_t SPARSE_MODE_INT_MAX = 214748647;
constexpr uint32_t ATTR_SPARSE_MODE = 6;
constexpr uint32_t ATTR_INNER_PRECISE = 7;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;
constexpr uint32_t PSE_SHIFT_DIM = 4;
constexpr uint32_t ATTENTION_MASK_DIM2 = 2;
constexpr uint32_t ATTENTION_MASK_DIM3 = 3;
constexpr uint32_t ATTENTION_MASK_DIM4 = 4;
constexpr int32_t BLOCK_SIZE_BASE = 128;  // 当前需要是128倍数，同时为了防止跨block搬运，mm base也设置为128
constexpr int32_t BLOCK_SIZE_MAX = 512;

constexpr uint32_t CVDIFF_S2_THRESHOLDS = 1;
constexpr uint32_t CVDIFF_SMALL_QS_THRESHOLDS = 16;
constexpr uint32_t CVDIFF_MM1RES_UB_SIZE = 16384; // 128 * 128
constexpr uint32_t CVDIFF_SOUTER_FACTOR_DEFAULT = 128;
constexpr uint32_t CVDIFF_SMALL_KV_THRESHOLDS = 1024;
constexpr uint32_t CVDIFF_SINNER_FACTOR_SMALL_KVS = 512;   // kv_s <= 512 场景sinner切块大小
constexpr uint32_t CVDIFF_SINNER_FACTOR_DEFAULT = 1024;    // cv分离一般场景sinner切块大小
constexpr uint32_t CVDIFF_SINNER_FACTOR_SMALL_QS = 2048;   // q_s <= 16 场景sinner切块大小

constexpr uint32_t SPLIT_DOUBLE_UB = 2;
constexpr uint32_t DSPLIT_THRESHOLDS_512 = 512;
constexpr uint64_t DSPLIT_S2_D_TILING_KEY = 400;
constexpr uint64_t DSPLIT_S2_TILING_KEY = 300;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint64_t BENCHMARK_TILING_KEY = 1000000000000000000;
constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t MATMUL_NORM_MIN_SEQ = 128;
constexpr uint32_t MATMUL_NORM_MIN_HEADSIZE = 128;

constexpr uint32_t BLIMIT = 65536;
constexpr uint32_t NLIMIT = 256;  // n不大于256
constexpr uint32_t SLIMIT = 20971520;  // s、kvs不大于20M
constexpr uint32_t DLIMIT = 512; // D不大于512

uint32_t promptGcd(uint32_t a, uint32_t b)
{
    if (a % b == 0) {
        return b;
	}
    return promptGcd(b, a % b);
}

ge::graphStatus ConvertContextToPFAParams(gert::TilingContext* context, ContextParamsForPFATiling& contextKeyParams)
{
    contextKeyParams.opName = context->GetNodeName();
    bool inputOutputIsNullPtr = (context->GetInputDesc(QUERY_INDEX) == nullptr) || (context->GetInputDesc(KEY_INDEX) == nullptr) ||
                                (context->GetInputDesc(VALUE_INDEX) == nullptr) || (context->GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
                                (context->GetInputShape(QUERY_INDEX) == nullptr) || (context->GetInputShape(KEY_INDEX) == nullptr) ||
                                (context->GetInputShape(VALUE_INDEX) == nullptr) || (context->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OPS_ERR_IF(inputOutputIsNullPtr,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "q, k, v or attenOut is nullptr!"),
                return ge::GRAPH_FAILED);

    contextKeyParams.isKvContinuous = 1;
    contextKeyParams.emptyTensor = 0;
    contextKeyParams.fromTilingSink = 0;
    contextKeyParams.pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask = context->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    contextKeyParams.actualSeqenceLengthQ = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSeqenceLengthKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.antiquantScale = context->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffset = context->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.inputDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType = context->GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.blockTable = nullptr;
    contextKeyParams.keySharedPrefix = (nullptr);
    contextKeyParams.valueSharedPrefix = (nullptr);
    contextKeyParams.actualSharedPrefixLen = (nullptr);
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
    context->GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
    context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.outputDataType = context->GetOutputDesc(0)->GetDataType();
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context->GetInputShape(KEY_INDEX);
    contextKeyParams.valueInputShape = context->GetInputShape(VALUE_INDEX);
    contextKeyParams.pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.deqScale1Shape = context->GetOptionalInputShape(DEQ_SCALE1_INDEX);
    contextKeyParams.scale1Shape = context->GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape = context->GetOptionalInputShape(DEQ_SCALE2_INDEX);
    contextKeyParams.scale2Shape = context->GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape = context->GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.antiquantScaleShape = context->GetOptionalInputShape(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffsetShape = context->GetOptionalInputShape(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.outputShape = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE);
    contextKeyParams.preToken = attrs->GetAttrPointer<int32_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken = attrs->GetAttrPointer<int32_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);
    contextKeyParams.compileInfoPtr = reinterpret_cast<const PromptFlashAttentionCompileInfo *>(context->GetCompileInfo());
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1 : 0;

    contextKeyParams.deqScaleType = (context->GetOptionalInputDesc(DEQ_SCALE1_INDEX) != nullptr) ?
    context->GetOptionalInputDesc(DEQ_SCALE1_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.deqScale2Type = (context->GetOptionalInputDesc(DEQ_SCALE2_INDEX) != nullptr) ?
    context->GetOptionalInputDesc(DEQ_SCALE2_INDEX)->GetDataType() : contextKeyParams.inputDataType;

    contextKeyParams.quantScale2Type = (context->GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType() : ge::DT_FLOAT;

    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::UpdateTilingKeyFlag(ContextParamsForPFATiling& contextKeyParams, uint64_t& tilingKey)
{
    uint64_t binaryFlag = 0;
    auto queryDtype = contextKeyParams.inputDataType;
    auto kvDtype = contextKeyParams.kDataType;
    if ((queryDtype == ge::DT_FLOAT16) && (kvDtype == ge::DT_INT8)) {
        binaryFlag += 8;    // 4bit flag位，最左侧表示是否进行反量化操作，对应值2**3 = 8，剩下3bit预留
    }
    tilingKey += (binaryFlag * 100000000000);
    return;
}

bool PromptFlashAttentionTiling::GetApiTmpSize(const uint32_t sOuterFactor, const uint32_t sInnerFactor, const uint32_t typeByteSize)
{
    auto tmpShape = Shape({sOuterFactor, sInnerFactor});
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        apiTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, true, true);
        return true;
    }
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND910B) {
        uint32_t softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
        uint32_t softmaxFlashTmpSize = GetSoftMaxFlashMinTmpSize(tmpShape, typeByteSize, true, true);
        if ((softmaxTmpSize == 0) || (softmaxFlashTmpSize == 0)) {
            return false;
        }
        apiTmpSize = std::max(softmaxTmpSize, softmaxFlashTmpSize);
    }
    return false;
}

size_t PromptFlashAttentionTiling::GetPFAWorkSpaceSize(PromptFlashAttentionTilingData& tilingData)
{
    size_t sysWorkspaceSize, workspaceSize;
    const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        sysWorkspaceSize = defaultSysWorkspaceSize; // sys workspace size default value
        return sysWorkspaceSize;
    } else { // 910b
        uint64_t maxSpmSize = tilingData.promptAttentionTensorSizeRect.get_spmTmpSize();
        sysWorkspaceSize = defaultSysWorkspaceSize910B; // sys workspace size default value
        if (tilingMod == TilingMod::CVDIFF) {
            int64_t mm1ResSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() * \
                                 tilingData.promptAttentionSingleCoreParams.get_singleProcessSInnerSize();
            int64_t mm2ResSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() * \
                                 tilingData.promptAttentionBaseParams.get_headSize();
            workspaceSize = sysWorkspaceSize + coreNum * softmaxDataTypeSize * (maxSpmSize + mm1ResSize * NUM_2 + mm2ResSize * NUM_2); // 2:use 2mm ub
            if (enableKvAntiquant) {
                int32_t KvAntiquantSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSInnerSize() * \
                                 tilingData.promptAttentionBaseParams.get_alignedHeadSize();
                workspaceSize += coreNum * dataTypeSize * KvAntiquantSize * 2;  // key value
            }
            if (enablePA) {
                workspaceSize += coreNum * 2 * 2 * 64;  // 2 bmm, db, 保证每个结构体64B对齐，dcci cacheline需要
            }
        } else {
            if ((splitS2 == 1) && (splitD == 1)) {
                workspaceSize = sysWorkspaceSize + coreNum * softmaxDataTypeSize * (maxSpmSize + \
                    NUM_2 * tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * // 2 : 2 mm ub
                    tilingData.promptAttentionSingleCoreParams.get_multiSmaxsInnerLoopTimes());
            } else {
                workspaceSize = sysWorkspaceSize + coreNum * softmaxDataTypeSize * (maxSpmSize + \
                    tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() + \
                    tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * \
                    tilingData.promptAttentionSingleCoreParams.get_multiSmaxsInnerLoopTimes());
            }
        }
        return workspaceSize;
    }
}

ge::graphStatus PromptFlashAttentionTiling::TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey,
    ContextParamsForPFATiling& contextKeyParams, bool useNewTiling, PromptFlashAttentionTilingData &tilingData) {
    auto inputDataType = contextKeyParams.inputDataType; // input q
    auto attenMaskElemType = contextKeyParams.maskDataType;
    auto outputDataType = contextKeyParams.outputDataType; // output tensor
    tilingData.promptAttentionBaseParams.set_attenMaskElemType(attenMaskElemType);

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        tilingKey = 12288; // 12288: 310p tiling
        tilingKey += (inputLayout == InputLayout::BNSD) ? 0 : 10000; // 10000 : BSH/BSND 22288
        return ge::GRAPH_SUCCESS;
    }
    tilingKey = 0U;
    // 非 cv diff模板时，有tail需要加1
    tilingKey += (tilingMod == TilingMod::CVDIFF) || (isSOuterNoTail && isSInnerNoTail && isDNoTail) ? 0U : 1U;
    tilingKey += inputDataType == ge::DT_BF16 ? 100U : 0U; // 输入qkv为BF16 加100
    tilingKey += inputDataType == ge::DT_INT8 ? 200U : 0U; // 输入qkv为INT8 加200
    tilingKey += tilingMod == TilingMod::CVDIFF ? 1002U : 0U; // 1002：CV分离模板加1000；且不区分 tail/no tail，统一加2
    tilingKey += outputDataType == ge::DT_INT8 ? 20000U : 0U; // 输出output为INT8 加20000

    if (!useNewTiling) {
        return ge::GRAPH_SUCCESS; // 老模板不考虑NSD区别 只有0、1、100、101
    }

    tilingKey += 10U; // 新模板10、11、15、16、110、111、115、116
    tilingKey += (inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD) ? 5U : 0U;

    // 针对CV分离模板的KV cache反量化，当前只处理CV分离模板里Q是FP16的情况。
    if ((inputDataType == ge::DT_FLOAT16 || inputDataType == ge::DT_BF16) && (tilingMod == TilingMod::CVDIFF)) {
        tilingKey = 1012;   // 已经在CV分离分支，+1000；走new_tiling，+10；不区分tail/notail，+2
        tilingKey += ((inputDataType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) ? 600 : 0; // fp16高精度模式，视作一种type 600
        tilingKey += (inputDataType == ge::DT_BF16) ? 100 : 0; // 100: bf16
        tilingKey += (outputDataType == ge::DT_BF16) ? 10000 : 0;
        tilingKey += (outputDataType == ge::DT_INT8) ? 20000 : 0;
        tilingKey += ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::SH) || (inputLayout == InputLayout::BSND)) ? 100000 : 0;
        tilingKey += (enableMatmulNorm ? 1000000 : 0);
        tilingKey += (enableMatmulNorm && enableSplitSeqOneN ? 1000000 : 0);
        UpdateTilingKeyFlag(contextKeyParams, tilingKey);   // 判断是否进行反量化，并结合其余预留bit位生成二进制数，取其十进制表达数
    }

    if (enablePA) {
        tilingKey += 10000000;  // 10000000: PA场景
    }
    if (isKVHasPrefix) {
        tilingKey += 100000000;  // 100000000 prefix场景
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionSplitNS(ContextParamsForPFATiling& contextKeyParams,
                                            PromptFlashAttentionTilingData& tilingData,
                                            uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths) {
    if (contextKeyParams.fromTilingSink != 0) {
        return ge::GRAPH_SUCCESS;
    }
    PromptAttentionSingleCoreParams* singleCoreParams = &tilingData.promptAttentionSingleCoreParams;
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    PromptAttentionSeqParams* seqParams = &tilingData.promptAttentionSeqParams;

    uint32_t arrayLen = baseParams->get_dimNumOfseq();

    std::vector<uint32_t> CoreHeadNumTail(arrayLen, 0U);
    std::vector<uint32_t> actualS1(arrayLen, 0U);
    std::vector<uint32_t> singleCoreHeadNumSize(arrayLen, 0U);
    std::vector<uint32_t> actualCoreNums(arrayLen, 0U);

    uint32_t multiSmaxsInnerLoopTimes = 0;

    std::vector<uint32_t> sInnerLoopTimes(arrayLen, 0U);
    std::vector<uint32_t> sOuterBlockNums(arrayLen, 0U);

    for (uint32_t i = LOOP_BEGIN_NUM; i < arrayLen; i++) {
        int seqLen = actualSeqLengths[i];
        sOuterBlockNums[i] = (seqLen + singleCoreParams->get_singleProcessSOuterSize() - 1)
                                / (singleCoreParams->get_singleProcessSOuterSize());
        sInnerLoopTimes[i] = (seqLen + singleCoreParams->get_singleProcessSInnerSize() - 1)
                                / (singleCoreParams->get_singleProcessSInnerSize());

        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[i]);

        if ((seqLen % singleCoreParams->get_singleProcessSOuterSize()) != 0) {
            isSOuterNoTail = false;
        }
        if ((seqLen % singleCoreParams->get_singleProcessSInnerSize()) != 0) {
            isSInnerNoTail = false;
        }

        // 两种策略 1。gcd均分核心   2。舍弃部分核心
        uint32_t headNumSize = baseParams->get_headNumSize();
        uint32_t n1 = promptGcd(curCoreNum, headNumSize);
        if (headNumSize / n1 > sOuterBlockNums[i]) {
            //舍弃部分核心 or N维度分核
            if (headNumSize > curCoreNum) {
                singleCoreHeadNumSize[i] = headNumSize / curCoreNum;
                CoreHeadNumTail[i] = headNumSize % curCoreNum;
                actualS1[i] = 1;
                actualCoreNums[i] = curCoreNum;
            } else {
                singleCoreHeadNumSize[i] = 1;
                CoreHeadNumTail[i] = 0;
                actualS1[i] = curCoreNum / headNumSize;
                actualCoreNums[i] = actualS1[i] * headNumSize;
            }
        } else { //gcd均分核心
            uint32_t s1 = (curCoreNum / n1);
            singleCoreHeadNumSize[i] = (headNumSize / n1);
            CoreHeadNumTail[i] = 0;
            actualS1[i] = s1;
            actualCoreNums[i] = (n1 * actualS1[i]);
        }
    }
    seqParams->set_singleCoreHeadNumSize(singleCoreHeadNumSize.data());
    seqParams->set_actualS1(actualS1.data());
    seqParams->set_CoreHeadNumTail(CoreHeadNumTail.data());
    seqParams->set_actualCoreNums(actualCoreNums.data());

    singleCoreParams->set_multiSmaxsInnerLoopTimes(multiSmaxsInnerLoopTimes);

    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::PromptFlashAttentionInitOutputSplit(uint64_t totalSize,
    PromptFlashAttentionTilingData &tilingData, uint32_t curCoreNum)
{
    PromptAttentionInitOutputParams *initParams = &tilingData.promptAttentionInitOutputParams;

    uint32_t singleCoreSize = (totalSize + curCoreNum - 1) / (curCoreNum); // 向上取整, coreNum获取时已校验非0

    if (outputType == ge::DT_INT8) {
        singleCoreSize = (singleCoreSize + 1) / 2 * 2;        // int8场景，初始化填0时按照half类型填，要求每个核分到的点数必须是偶数
    }

    initParams->set_singleCoreSize(singleCoreSize);
    initParams->set_totalOutputSize(totalSize);
}

void PromptFlashAttentionTiling::PromptFlashAttentionInitSoftmaxLseOutputSplit(uint64_t totalSize,
    PromptFlashAttentionTilingData &tilingData)
{
    PromptAttentionInitOutputParams *initParams = &tilingData.promptAttentionInitOutputParams;
    initParams->set_totalSoftMaxLseOutputSize(totalSize);
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionSplitNSNew(
    ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData,
    uint32_t curCoreNum, std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, int64_t actualSharedPrefixLen, bool useBalanceTiling) {
    if (contextKeyParams.fromTilingSink != 0) {
        return ge::GRAPH_SUCCESS;
    }
    PromptAttentionSingleCoreParams* singleCoreParams = &tilingData.promptAttentionSingleCoreParams;
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    PromptAttentionSeqParams* seqParams = &tilingData.promptAttentionSeqParams;

    uint32_t arrayLen = baseParams->get_dimNumOfseq();

    std::vector<uint32_t> accumSOuterTilingNums(static_cast<size_t>(arrayLen), 0U);
    std::vector<uint32_t> sInnerLoopTimes(static_cast<size_t>(arrayLen), 0U);
    std::vector<uint32_t> sOuterBlockNums(static_cast<size_t>(arrayLen), 0U);

    // tiling结构体元素, 长度需要大于等于TILING_DATA_FIELD_DEF_ARR指定的长度
    // tiling结构体定义指定长度为50, 则vector定义需比较与curCoreNum的大小, 取较大值
    const size_t tilingElementArrayLen = (static_cast<size_t>(curCoreNum) > 50UL) ? \
        static_cast<size_t>(curCoreNum) : 50UL;
    std::vector<uint32_t> coreSposEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSposStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreSidStart(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidEnd(tilingElementArrayLen, 0U);
    std::vector<uint32_t> coreNidStart(tilingElementArrayLen, 0U);

    int64_t totalBlockWight = 0;
    int totalOuterBlockNum = 0;
    uint32_t preAccumSOuterNum = 0U;
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    int nextTokensPerBatch = 0;
    uint32_t sInnerPrefixLoopTimes = (actualSharedPrefixLen + singleCoreParams->get_singleProcessSInnerSize() - 1)
                                     / (singleCoreParams->get_singleProcessSInnerSize());
    for (uint32_t i = LOOP_BEGIN_NUM; i < arrayLen; i++) {
        int seqLen = actualSeqLengths[i];
        int subSeqInnerLen = actualSeqLengthsKV[i];
        sOuterBlockNums[i] = (seqLen + singleCoreParams->get_singleProcessSOuterSize() - 1)
                                / (singleCoreParams->get_singleProcessSOuterSize());
        sInnerLoopTimes[i] = (subSeqInnerLen + singleCoreParams->get_singleProcessSInnerSize() - 1)
                                / (singleCoreParams->get_singleProcessSInnerSize()) + sInnerPrefixLoopTimes;
        accumSOuterTilingNums[i] = (sOuterBlockNums[i] * baseParams->get_headNumSize()) + preAccumSOuterNum;
        preAccumSOuterNum = accumSOuterTilingNums[i];

        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[i]);

        if (baseParams->get_sparseMode() == SPARSE_MODE_RIGHT_DOWN) {
            nextTokensPerBatch = subSeqInnerLen + actualSharedPrefixLen - seqLen;
        } else {
            nextTokensPerBatch = baseParams->get_nextTokens();
        }

        if (seqLen % singleCoreParams->get_singleProcessSOuterSize() != 0) {
            isSOuterNoTail = false;
        }
        if (subSeqInnerLen % singleCoreParams->get_singleProcessSInnerSize() != 0) {
            isSInnerNoTail = false;
        }
        totalOuterBlockNum += sOuterBlockNums[i];
        if (nextTokensPerBatch == 0) {
            totalBlockWight += (static_cast<int64_t>(sOuterBlockNums[i]) + 1) * static_cast<int64_t>(sOuterBlockNums[i]) / NUM_2;  // div 2
        } else {
            totalBlockWight += static_cast<int64_t>(sOuterBlockNums[i]) * static_cast<int64_t>(sInnerLoopTimes[i]);
        }
    }
    if ((!useBalanceTiling)) {
        accumSOuterTilingNums[0] = 0;
    }

    float coreWightTarget = (float(totalBlockWight * baseParams->get_headNumSize()) / float(curCoreNum));

    // 临时算法 待优化
    int curWight = 0;
    int curCore = 0;
    coreSposStart[curCore] = 0;
    coreSidStart[curCore] = 0;
    coreNidStart[curCore] = 0;
    for (uint32_t i = LOOP_BEGIN_NUM; i < baseParams->get_headNumSize(); i++) {
        for (uint32_t j = 0; j < arrayLen; j++) {
            if (baseParams->get_sparseMode() == SPARSE_MODE_RIGHT_DOWN) {
                nextTokensPerBatch = actualSeqLengthsKV[j] + actualSharedPrefixLen - actualSeqLengths[j];
            } else {
                nextTokensPerBatch = baseParams->get_nextTokens();
            }
            for (uint32_t k = 0; k < sOuterBlockNums[j]; k++) {
                int64_t dif = int64_t(coreWightTarget * float(curCore + 1)) - curWight;
                int64_t curWightPlus;
                if (nextTokensPerBatch == 0) {
                    curWightPlus = k + 1;
                }else {
                    curWightPlus = sInnerLoopTimes[j];
                }
                if ((curWightPlus - dif) > dif) {
                    if (k == 0) {
                        if (j == 0) {
                            coreNidEnd[curCore] = i;
                            coreSidEnd[curCore] = arrayLen;
                            coreSposEnd[curCore] = sOuterBlockNums[arrayLen - 1];
                        } else {
                            coreNidEnd[curCore] = i + 1;
                            coreSidEnd[curCore] = j;
                            coreSposEnd[curCore] = sOuterBlockNums[j-1];
                        }
                    } else {
                        coreNidEnd[curCore] = i + 1;
                        coreSidEnd[curCore] = j + 1;
                        coreSposEnd[curCore] = k;
                    }
                    curCore += 1;
                    coreNidStart[curCore] = i;
                    coreSidStart[curCore] = j;
                    coreSposStart[curCore] = k;
                }
                curWight += curWightPlus;
            }
        }
    }
    coreNidEnd[curCore] = (baseParams->get_headNumSize());
    coreSidEnd[curCore] = arrayLen;
    coreSposEnd[curCore] = sOuterBlockNums[arrayLen-1];

    // 临时复用
    seqParams->set_CoreHeadNumTail(coreNidStart.data());
    seqParams->set_actualS1(coreNidEnd.data());
    seqParams->set_actualCoreNums(coreSidStart.data());
    seqParams->set_singleCoreHeadNumSize(coreSidEnd.data());
    seqParams->set_coreSeqPosStart(coreSposStart.data());
    seqParams->set_coreSeqPosEnd(coreSposEnd.data());

    singleCoreParams->set_multiSmaxsInnerLoopTimes(multiSmaxsInnerLoopTimes);
    singleCoreParams->set_actualCoreNums(curCore + 1);

    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::GetPreNextTokensLeftUp(PromptFlashAttentionTilingData& tilingData,
    uint32_t actualSeqLength, uint32_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) {
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    int64_t sparsePreTokens = baseParams->get_preTokens();
    int64_t sparseNextTokens = baseParams->get_nextTokens();
    if (baseParams->get_sparseMode() == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        nextTokensLeftUp = (int64_t)actualSeqLengthKV - (int64_t)actualSeqLength;
    } else if (baseParams->get_sparseMode() == SPARSE_MODE_BAND) {
        preTokensLeftUp = sparsePreTokens - (int64_t)actualSeqLengthKV + (int64_t)actualSeqLength;
        nextTokensLeftUp = sparseNextTokens + (int64_t)actualSeqLengthKV - (int64_t)actualSeqLength;
    } else {
        preTokensLeftUp = sparsePreTokens;
        nextTokensLeftUp = sparseNextTokens;
    }
}

bool PromptFlashAttentionTiling::EnableSplitSeqOneN(PromptFlashAttentionTilingData& tilingData, const ContextParamsForPFATiling& contextKeyParams) {
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    uint32_t actualSeqLength = baseParams->get_seqSize();
    uint32_t actualSeqLengthKV = baseParams->get_seqInnerSize();
    uint32_t b = baseParams->get_dimNumOfseq();
    uint32_t n = baseParams->get_headNumSize();
    const int64_t seq16K = 16 * 1024;
    const int64_t seq32K = 32 * 1024;
    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;

    bool enableLeftPadding = ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr));
    bool enableRingAttention = (contextKeyParams.isSoftMaxLseEnable == true);

    GetPreNextTokensLeftUp(tilingData, actualSeqLength, actualSeqLengthKV, preTokensLeftUp, nextTokensLeftUp);

    if (enableMatmulNorm && !enableLeftPadding && !enableRingAttention && (b == 1) && ((n == 12) || (n == 40)) &&  // 12 : headNumSize, 40 : headNumSize
        (baseParams->get_isActualSeqLengthsNull() == 1) && (baseParams->get_isActualSeqLengthsKVNull() == 1) &&
        (baseParams->get_isKvContinuous() == 1) && (actualSeqLength == actualSeqLengthKV) && (actualSeqLength >= seq16K) &&
        (actualSeqLength % 256 == 0) && (baseParams->get_sparseMode() == SPARSE_MODE_BAND) &&
        ((preTokensLeftUp == seq16K) || (preTokensLeftUp == seq32K)) && (nextTokensLeftUp == 0)) {
        enableSplitSeqOneN = true;
    }

    return enableSplitSeqOneN;
}

void PromptFlashAttentionTiling::PromptFlashAttentionSplitSeqOneN(PromptFlashAttentionTilingData& tilingData,
                                                                  uint32_t curCoreNum, bool isVectorCore) {
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    PromptAttentionSingleCoreParams* singleCoreParams = &tilingData.promptAttentionSingleCoreParams;
    PromptAttentionSeqParams* seqParams = &tilingData.promptAttentionSeqParams;

    uint32_t actualSeqLength = baseParams->get_seqSize();
    uint32_t actualSeqLengthKV = baseParams->get_seqInnerSize();
    int64_t preTokensLeftUp;
    int64_t nextTokensLeftUp;
    GetPreNextTokensLeftUp(tilingData, actualSeqLength, actualSeqLengthKV, preTokensLeftUp, nextTokensLeftUp);

    uint32_t sOuterSize = singleCoreParams->get_singleProcessSOuterSize();
    if (!isVectorCore) {        // 以cube为视角时，sOuter*2去分核，每个cube核内，2个vector核拿到的仍然是sOuter
        sOuterSize = sOuterSize * 2;  // 2 : sOuter*2去分核
        curCoreNum = curCoreNum / 2;  // 2 : 每个cube核内，2个vector核拿到的仍然是sOuter
    }

    int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
    int64_t outerBlockFirstColNums = (preTokensLeftUp < static_cast<int32_t>(actualSeqLength)) ?
                                     ((preTokensLeftUp + sOuterSize - 1) / sOuterSize + 1) : outerBlockNums;
    int64_t outerBlockLeftDownFirstColNums = outerBlockNums - outerBlockFirstColNums;
    int64_t leftDownBlockNums = (outerBlockLeftDownFirstColNums + 1) * outerBlockLeftDownFirstColNums / 2;

    int64_t innerBlockNums = (actualSeqLengthKV + sOuterSize - 1) / sOuterSize;
    int64_t innerBlockFirstRowNums = (nextTokensLeftUp < static_cast<int32_t>(actualSeqLengthKV)) ?
                                     ((nextTokensLeftUp + sOuterSize - 1) / sOuterSize - 1) : innerBlockNums;
    int64_t innerBlockRightUpFirstRowNums = innerBlockNums - innerBlockFirstRowNums;
    int64_t rightUpBlockNums = (innerBlockRightUpFirstRowNums + 1) * innerBlockRightUpFirstRowNums / 2;

    int64_t toCalcBlockNums = innerBlockNums * outerBlockNums - rightUpBlockNums - leftDownBlockNums;
    double perWeight = static_cast<double>(toCalcBlockNums) / static_cast<double>(curCoreNum);

    uint32_t coreSOuterIndexStart[50] = {0};
    uint32_t coreSOuterIndexEnd[50] = {0};
    int64_t curWeight = 0;
    uint32_t coreIndex = 0;
    int64_t sInnerBlockNums = 0;
    int64_t sInnerIndexStart = 0;
    int64_t sInnerIndexEnd = innerBlockFirstRowNums;

    for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
        sInnerIndexStart = (sOuterIndex < outerBlockFirstColNums) ? 0 : (sInnerIndexStart + 1);
        sInnerBlockNums = sInnerIndexEnd - sInnerIndexStart;
        curWeight += sInnerBlockNums;
        if (curWeight >= (perWeight * (coreIndex + 1))) {
            coreSOuterIndexEnd[coreIndex] = sOuterIndex;
            coreIndex++;
            coreSOuterIndexStart[coreIndex] = sOuterIndex;
            if (coreIndex >= curCoreNum - 1) {
                coreSOuterIndexEnd[coreIndex] = outerBlockNums;
                break;
            }
        }
        sInnerIndexEnd = std::min(innerBlockNums, sInnerIndexEnd + 1);
    }

    // 核没有分配满的情况
    coreSOuterIndexEnd[coreIndex] = outerBlockNums;
    seqParams->set_coreSeqPosStart(coreSOuterIndexStart);
    seqParams->set_coreSeqPosEnd(coreSOuterIndexEnd);
    uint32_t actualCoreNums = coreIndex + 1;
    if (!isVectorCore) {
        actualCoreNums = actualCoreNums * 2;  // 2 : 分核
    }
    singleCoreParams->set_actualCoreNums(actualCoreNums);
}

bool PromptFlashAttentionTiling::EnableMTE2BmmPipe(PromptFlashAttentionTilingData& tilingData,
                                                   matmul_tiling::MatmulApiTiling& bmm, TCubeTiling& bmmTilingData,
                                                   uint32_t sOuterFactor, uint32_t sInnerFactor) {
    if (tilingData.promptAttentionBaseParams.get_seqSize() > 16) { // 小艺投机推理
        return true;
    }
    uint32_t baseK = 32U;
    uint32_t head_size = tilingData.promptAttentionBaseParams.get_headSize();
    if(head_size%baseK != 0) {
        return true;
    }

    uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
    uint32_t baseN = std::min(uint32_t(512), sInnerFactor);
    if (enablePA) {
        baseN = BLOCK_SIZE_BASE;
    }
    bmm.SetFixSplit(baseM, baseN, baseK);
    bool res = bmm.GetTiling(bmmTilingData) != -1;
    return res;
}

void PromptFlashAttentionTiling::EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData) {
    if ((bmmTilingData.get_depthA1() == 1) && (bmmTilingData.get_depthB1() == 1)) {
        bmmTilingData.set_depthA1(2); // 2 : depthA1
        bmmTilingData.set_depthB1(2); // 2 : depthB1
    }
}

void PromptFlashAttentionTiling::PromptFlashAttention310PSetBmm1(matmul_tiling::MatmulApiTiling& bmm1)
{
    // 310p mm1: A gm ND, B gm ND, C vec NZ
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
                  matmul_tiling::DataType::DT_FLOAT16);
}

void PromptFlashAttentionTiling::PromptFlashAttention310PSetBmm2(matmul_tiling::MatmulApiTiling& bmm2)
{
    // 310p mm2: A vec NZ, B gm ND, C vec ND
    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
                  matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT16);
}

ge::graphStatus PromptFlashAttentionTiling::CheckKeyValueParamsConsistency(const ContextParamsForPFATiling& contextKeyParams) {
    if (!contextKeyParams.isKvContinuous) {
        return GRAPH_SUCCESS;
    }

    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    const uint32_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    uint32_t tmpKeyDim, tmpValueDim;

    OPS_ERR_IF(contextKeyParams.kDataType != contextKeyParams.vDataType,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "tensor key dtype(%d) must be consistent with tensor value dtype(%d)!", contextKeyParams.kDataType, contextKeyParams.vDataType),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyDimNum != valueDimNum,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "tensor key shape dimNum(%u) must be consistent with tensor value shape dimNum(%u)!", keyDimNum, valueDimNum),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((keyDimNum < INPUT_QKV_SHAPE_MIN_DIMS) || (keyDimNum > INPUT_QKV_SHAPE_MAX_DIMS),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "tensor key shape dimNum(%u) is invalid! Only support range [%u, %u]", keyDimNum, INPUT_QKV_SHAPE_MIN_DIMS, INPUT_QKV_SHAPE_MAX_DIMS),
                    return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < keyDimNum; ++i) {
        tmpKeyDim = keyShape->GetStorageShape().GetDim(i);
        tmpValueDim = valueShape->GetStorageShape().GetDim(i);
        OPS_ERR_IF(tmpKeyDim != tmpValueDim,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "tensor key shape (%u) do not equal to tensor value shape(%u) in dim %u", tmpKeyDim, tmpValueDim, i),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::PromptFlashAttentionCheckBmm1(PromptFlashAttentionTilingData& tilingData,
    TCubeTiling& bmm1TilingData,  int64_t l1SizeRemain, int64_t l0CSize,
    uint32_t sOuterFactor, uint32_t sInnerFactor, bool allGM, bool autoBaseMNK) {
    int32_t ret = 0;
    matmul_tiling::MatmulApiTiling bmm1(ascendPlatformInfo);
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        PromptFlashAttention310PSetBmm1(bmm1);
    } else { // 910b
        matmul_tiling::DataType bmm1InputType = matmul_tiling::DataType::DT_FLOAT16;
        matmul_tiling::DataType bmm1OutputType = matmul_tiling::DataType::DT_FLOAT16;
        GetMatMulType(bmm1InputType, bmm1OutputType);
        matmul_tiling::TPosition cPosition = allGM ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::VECCALC;
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, true);
        bmm1.SetCType(cPosition, matmul_tiling::CubeFormat::ND, bmm1OutputType);
    }
    ret = bmm1.SetShape(sOuterFactor, sInnerFactor, tilingData.promptAttentionBaseParams.get_headSize());
    OPS_ERR_IF(ret != 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetShape failed, ret = %d!", ret),
                    return false);
    int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
    int32_t strideQ = tilingData.promptAttentionBaseParams.get_headSize() *
                        tilingData.promptAttentionBaseParams.get_headNumSize();
    int32_t strideK = strideQ / ratio;
    if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::SH) ||
        (inputLayout == InputLayout::BSND)) {
        bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                    tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                    strideQ, strideK);

        if (enableKvAntiquant) {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                             tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                             strideQ, tilingData.promptAttentionBaseParams.get_headSize());
        }
    } else if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
        if (enablePA && PAlayoutType == 1) {  // PA左矩阵为BNSD，右矩阵为BSH
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                     tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                     tilingData.promptAttentionBaseParams.get_headSize(), strideK);
        } else {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                     tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                     tilingData.promptAttentionBaseParams.get_headSize());
        }
    }

    bmm1.SetBias(false);
    ret = bmm1.SetBufferSpace(l1SizeRemain, l0CSize);
    OPS_ERR_IF(ret != 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetBufferSpace failed, l1SizeRemain = %ld, l0CSize = %ld, ret = %d!",
                    l1SizeRemain, l0CSize, ret),
                    return false);
    if (enablePA) {
        ret = bmm1.SetFixSplit(sOuterFactor, BLOCK_SIZE_BASE);
    } else {
        ret = bmm1.SetFixSplit(sOuterFactor, sInnerFactor);
    }
    OPS_ERR_IF(ret != 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetFixSplit failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
                    l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, ret),
                    return false);
    if (inputType == ge::DT_INT8) {
        bmm1.SetDequantType(matmul_tiling::DequantType::SCALAR);
    }

    ret = bmm1.GetTiling(bmm1TilingData);
    if (autoBaseMNK) {
        if (enableMatmulNorm) {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(128), sInnerFactor);
            uint32_t baseK = 128U;
            bmm1.SetFixSplit(baseM, baseN, baseK);
            ret = bmm1.GetTiling(bmm1TilingData);
        } else {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(256), sInnerFactor);
            uint32_t baseK = 64U;
            if (enablePA) {
                baseN = BLOCK_SIZE_BASE;
            }
            if (ret != 0) {
                bmm1.SetFixSplit(baseM, baseN, baseK);
                ret = bmm1.GetTiling(bmm1TilingData);
            }
        }
    }

    OPS_ERR_IF(ret != 0, // GetTiling失败
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 GetTiling failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d, ret = %d!",
                    l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
                    return false);

    bmm1TilingData.set_shareMode(0);
    bmm1TilingData.set_shareL1Size(l1SizeRemain);
    bmm1TilingData.set_shareL0CSize(l0CSize);
    if (curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
        bmm1TilingData.set_shareUbSize(0);
        EnableBmmDoubleBuffer(bmm1TilingData); // 开启bmm1计算的double buffer，bmm1的mte2能bound
    }

    bool res = EnableMTE2BmmPipe(tilingData, bmm1, bmm1TilingData, sOuterFactor, sInnerFactor); // 开启MTE2 Matmul 流水pipe

    OPS_ERR_IF(res == false,      // EnableMTE2BmmPipe执行失败
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "EnableMTE2BmmPipe failed!"),
                    return false);

    return true;
}

void PromptFlashAttentionTiling::GetMatMulType(matmul_tiling::DataType &mmInputType,
    matmul_tiling::DataType &mmOutputType) {
    if (inputType == ge::DT_FLOAT16 && innerPrecise == HIGH_PRECISION) {
        mmInputType = matmul_tiling::DataType::DT_FLOAT16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_BF16) {
        mmInputType = matmul_tiling::DataType::DT_BF16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_INT8) {
        mmInputType = matmul_tiling::DataType::DT_INT8;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT16;
    }
}

bool PromptFlashAttentionTiling::PromptFlashAttentionCheckBmm2(PromptFlashAttentionTilingData& tilingData,
    TCubeTiling& bmm2TilingData,  int64_t l1SizeRemain, int64_t l0CSize,
    uint32_t sOuterFactor, uint32_t sInnerFactor, uint32_t dSplitFactor, bool allGM, bool autoBaseMNK) {
    int32_t ret = 0;
    matmul_tiling::MatmulApiTiling bmm2(ascendPlatformInfo);
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        PromptFlashAttention310PSetBmm2(bmm2);
        ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(), sInnerFactor);
        if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) {
            int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
            int32_t strideQ = tilingData.promptAttentionBaseParams.get_headSize() *
                            tilingData.promptAttentionBaseParams.get_headNumSize();
            int32_t strideV = strideQ / ratio;
            bmm2.SetOrgShape(sOuterFactor, strideV, sInnerFactor,
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
        } else if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) { // M, N, KA, KB
            bmm2.SetOrgShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(), sInnerFactor,
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
        }
    } else { // 910b
        matmul_tiling::DataType bmm2InputType = matmul_tiling::DataType::DT_FLOAT16;
        matmul_tiling::DataType bmm2OutputType = matmul_tiling::DataType::DT_FLOAT16;
        GetMatMulType(bmm2InputType, bmm2OutputType);
        if ((splitS2 == 1) && (splitD == 1)) {
            bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
            bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
            bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutputType);
            ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(),
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
        } else {
            matmul_tiling::TPosition aPosition = allGM ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::TSCM;
            matmul_tiling::TPosition cPosition = allGM ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::VECCALC;
            bmm2.SetAType(aPosition, matmul_tiling::CubeFormat::NZ, bmm2InputType, false);
            bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
            bmm2.SetCType(cPosition, matmul_tiling::CubeFormat::ND_ALIGN, bmm2OutputType);
            ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(), sInnerFactor);
        }
    OPS_ERR_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set SetShape failed, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
                sOuterFactor, sInnerFactor, ret),
                return false);
    int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
    int32_t strideQ = tilingData.promptAttentionBaseParams.get_headSize() *
                    tilingData.promptAttentionBaseParams.get_headNumSize();
    int32_t strideV = strideQ / ratio;
        if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) ||
            (inputLayout == InputLayout::SH)) {
            bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(), strideV,
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
            if (enableKvAntiquant) {
                bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                                 tilingData.promptAttentionBaseParams.get_headSize(),
                                 tilingData.promptAttentionBaseParams.get_seqInnerSize());
            }
        } else if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
            if (enablePA && PAlayoutType == 1) {  // PA左矩阵为BNSD，右矩阵为BSH
                bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(), strideV,
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
            } else {
                bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                            tilingData.promptAttentionBaseParams.get_headSize(),
                            tilingData.promptAttentionBaseParams.get_seqInnerSize());
            }
        }
    }

    bmm2.SetBias(false);
    ret = bmm2.SetBufferSpace(l1SizeRemain, l0CSize);
    OPS_ERR_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set SetBufferSpace failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
                l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, ret),
                return false);
    if (inputType == ge::DT_INT8) {
        bmm2.SetDequantType(matmul_tiling::DequantType::SCALAR);
    }

    if (autoBaseMNK) {
        if (enableMatmulNorm) {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(128), tilingData.promptAttentionBaseParams.get_headSize());
            uint32_t baseK = 128U;
            bmm2.SetFixSplit(baseM, baseN, baseK);
        }
        ret = bmm2.GetTiling(bmm2TilingData);
    } else {
        if ((isDNoTail) || (splitS2 == 0) || (splitD == 1)) {
            bmm2.SetFixSplit(sOuterFactor, dSplitFactor);
        } else {
            bmm2.SetFixSplit(sOuterFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize());
        }
        ret = bmm2.GetTiling(bmm2TilingData);
    }
    OPS_ERR_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set GetTiling failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d, ret = %d!",
                l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
                return false);
    bmm2TilingData.set_shareMode(0);
    bmm2TilingData.set_shareL1Size(l1SizeRemain);
    bmm2TilingData.set_shareL0CSize(l0CSize);
    OPS_ERR_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set shareL0CSize failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d, ret = %d!",
                l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
                return false);
    if (curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
         bmm2TilingData.set_shareUbSize(0);
    }
    return true;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionSetTensorSize(
    PromptFlashAttentionTilingData& tilingData,
    PromptAttentionSingleCoreTensorSize& tensorSize,
    uint32_t sOuterFactor, uint32_t sInnerFactor) {
    if (tilingData.promptAttentionBaseParams.get_useMask() == 0U && usePseShift == 0U) {
        // 在没有配置attentionMask且没有pse的场景中，可以节省掉attentionMask的UB内存
        // 但是需要预留2分BYTE_BLOCK(32BYTE)的UB内存用于Bmm2UpdateDiv
        tensorSize.set_attenMaskUbSize(sOuterFactor * BYTE_BLOCK * NUM_2 / softmaxDataTypeSize);
    } else {
        tensorSize.set_attenMaskUbSize(sOuterFactor * sInnerFactor);
    }

    if (usePseShift == 0U) {
        tensorSize.set_pseShiftUbSize(0);
    } else {
        tensorSize.set_pseShiftUbSize(sOuterFactor * sInnerFactor);
    }

    tensorSize.set_mmResUbSize(sOuterFactor * sInnerFactor);
    tensorSize.set_maskSize(tensorSize.get_mmResUbSize());
    tensorSize.set_softmaxSumSize(tensorSize.get_softmaxMaxSize());
    tensorSize.set_softmaxExpSize(sOuterFactor * tilingData.promptAttentionBaseParams.get_softmaxTypeByteNum());
    tensorSize.set_softmaxValueSize(sOuterFactor * sInnerFactor);
    tensorSize.set_bmm2ResUbSize(sOuterFactor * tilingData.promptAttentionBaseParams.get_alignedHeadSize());
    tensorSize.set_tmpMMResBmm2PreUbSize(std::max(tensorSize.get_mmResUbSize(), tensorSize.get_bmm2ResUbSize()));
    tensorSize.set_tmpSoftmaxBmm2UbSize(SOFTMAX_BUFFER_NUM * tensorSize.get_softmaxMaxSize());
    if ((splitS2 == 1) && (splitD == 1)) {
        tensorSize.set_spmTmpSize(tensorSize.get_bmm2ResUbSize() + tensorSize.get_softmaxExpSize() * SPLIT_DOUBLE_UB);
    } else {
        tensorSize.set_spmTmpSize(tensorSize.get_bmm2ResUbSize());
    }
    // 310P需要tscm buf
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        tensorSize.set_scmTmpSize(tilingData.promptAttentionBaseParams.get_headSize() * std::max(sOuterFactor, sInnerFactor));
        tensorSize.set_softmaxMaxSize(sOuterFactor * (BYTE_BLOCK / softmaxDataTypeNZ_));
    } else {
        tensorSize.set_softmaxMaxSize(sOuterFactor * (BYTE_BLOCK / sizeof(float)));
    }
    if (tilingData.promptAttentionBaseParams.get_maskTypeByteNum() == (BYTE_BLOCK / BOOLSIZE)) {
        tensorSize.set_selectSpaceUbSize(GetSelectWithBytesMaskMinTmpSize(Shape({sOuterFactor, sInnerFactor}), Shape({1}), 1,
            Shape({sOuterFactor, sInnerFactor}), 1, false));
    } else {
        tensorSize.set_selectSpaceUbSize(0);
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t PromptFlashAttentionTiling::CalculateL1SizeUsed(PromptFlashAttentionTilingData& tilingData, const uint32_t typeByteSize)
{
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        return (typeByteSize * tilingData.promptAttentionTensorSizeRect.get_scmTmpSize() * 3); // 需要两块tscm buf给a1、b1 or b1、b2
    }
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND910B) {
        return (typeByteSize * tilingData.promptAttentionTensorSizeRect.get_scmTmpSize());
    }
    return 0;
}

bool PromptFlashAttentionTiling::PromptFlashAttentionCheckArgsLegal(PromptFlashAttentionTilingData& tilingData,
    int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t typeByteSize,
    uint32_t& sOuterFactor, uint32_t sInnerFactor,
    bool& updateDiv, uint32_t maskTypeSize, uint32_t dSplitFactor) {
    // 调整基本块
    bool res = true;
    if (AdjustBasicBlock(tilingData, sOuterFactor) != ge::GRAPH_SUCCESS) {
            return false;
    }
    auto tmpShape = Shape({sOuterFactor, sInnerFactor});  // [S,s]
    int64_t softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
    int64_t softmaxFlashTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, softmaxDataTypeNZ_, true, true);
    if ((softmaxTmpSize == 0) || (softmaxFlashTmpSize == 0)) {
        return false;
    }

    int64_t queueBufferSize = 0;
    int64_t pseShiftBufferSize = 0;

    PromptFlashAttentionSetTensorSize(tilingData, tilingData.promptAttentionTensorSizeRect,
                                        sOuterFactor, sInnerFactor);
    int32_t l1SizeRemain = l1Size - CalculateL1SizeUsed(tilingData, typeByteSize);
    if (l1SizeRemain < 0) {
        updateDiv = true;
        res = false;
        return res;
    }

    res = (PromptFlashAttentionCheckBmm1(tilingData, tilingData.bmm1TilingDataRect,
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor)) &&
        (PromptFlashAttentionCheckBmm2(tilingData, tilingData.bmm2TilingDataRect,
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, dSplitFactor));

    queueBufferSize = tilingData.promptAttentionTensorSizeRect.get_attenMaskUbSize();

    pseShiftBufferSize = tilingData.promptAttentionTensorSizeRect.get_pseShiftUbSize();

    pseMaskMaxSize = std::max(maskTypeSize, pseShiftElemSize);

    uint32_t pseShiftCastSize = 0U;
    if ((usePseShift == 1) && (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || pseShiftElemType == ge::DT_BF16)) {
        pseShiftCastSize = FLOAT32SIZE;   // 在高精度生效或者bf16情况，pse需要做cast，申请ub
    }

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        matmul_tiling::SysTilingTempBufSize mm1bufSize, mm2bufSize;
        int32_t getBufRes;
        apiTmpSize = GetApiTmpSize(sOuterFactor, sInnerFactor, typeByteSize);
        getBufRes = MatmulGetTmpBufSize(tilingData.bmm1TilingDataRect, mm1bufSize);
        getBufRes += MatmulGetTmpBufSize(tilingData.bmm2TilingDataRect, mm2bufSize);
        if (getBufRes < 0) {
            updateDiv = true;
            res = false;
            return res;
        }
        ubSizeRemain = ubSize - (apiTmpSize +
                    tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() +
                    tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() * 2 + // 2:2 mm2 ub
                    SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                    typeByteSize - tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize() * 4 - queueBufferSize * maskTypeSize * 2;
        tilingData.promptAttentionTensorSizeRect.set_tmpSoftMaxV2Size((ubSizeRemain + apiTmpSize) / UB_ALIGN * UB_ALIGN);
        tilingData.promptAttentionTensorSizeRect.set_mm1TmpUbSize(mm1bufSize.ubSize);
        tilingData.promptAttentionTensorSizeRect.set_mm2TmpUbSize(mm2bufSize.ubSize);
    } else {
        apiTmpSize = std::max(softmaxTmpSize, softmaxFlashTmpSize);
        if ((splitS2 == 1) && (splitD == 1)) {
            ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * SPLIT_DOUBLE_UB +
                                    tilingData.promptAttentionTensorSizeRect.get_softmaxValueSize() +
                                    SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                                    typeByteSize - (queueBufferSize * pseMaskMaxSize) -
                                    tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                                    pseShiftBufferSize * pseShiftCastSize;
        } else if ((splitS2 == 1) && (splitD == 0)) {
            ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() +
                                    tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() * NUM_2 + // 2:2 mm2 ub
                                    SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                                    typeByteSize - (queueBufferSize * pseMaskMaxSize) -
                                    tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                                    pseShiftBufferSize * pseShiftCastSize;
        } else {
            ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() +
                                    SPLIT_DOUBLE_UB * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                                    typeByteSize - (queueBufferSize * pseMaskMaxSize) -
                                    tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                                    pseShiftBufferSize * pseShiftCastSize;
        }
    }

    if (ubSizeRemain < 0) {
        updateDiv = true;
        res = false;
        return res;
    }
    updateDiv = (!res);
    return res;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionApiTiling(PromptFlashAttentionTilingData& tilingData,
    uint32_t typeSize,  uint32_t sOuterFactor, uint32_t softmaxSInnerFactor, uint32_t softmaxSOuterFactor) {
    auto softmaxShapeRect = Shape({softmaxSOuterFactor, softmaxSInnerFactor});

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        uint32_t sftV2Size = GetSoftMaxFlashV2MinTmpSize(softmaxShapeRect, softmaxDataTypeNZ_, softmaxDataTypeNZ_, true);
        if ((ubSizeRemain + apiTmpSize) < sftV2Size) {
            return ge::GRAPH_FAILED;
        }
        SoftMaxFlashV2TilingFunc(softmaxShapeRect, softmaxDataTypeNZ_, softmaxDataTypeNZ_, (ubSizeRemain + apiTmpSize) / UB_ALIGN * UB_ALIGN,
            tilingData.softmaxTilingDataRect, true);
    } else {
        SoftMaxTilingFunc(softmaxShapeRect, sizeof(float), ubSizeRemain + apiTmpSize, tilingData.softmaxTilingDataRect);
        SoftMaxFlashV2TilingFunc(softmaxShapeRect, softmaxDataTypeSize, sizeof(float), ubSizeRemain + apiTmpSize,
            tilingData.softmaxFlashTilingDataRect, true, true);
    }

    auto transposeSrcShapeRect = Shape({1, 1, sOuterFactor,
                                      tilingData.promptAttentionBaseParams.get_headSize()});
    auto transposeDstShape = Shape({tilingData.promptAttentionBaseParams.get_batchSize(),
                                      tilingData.promptAttentionBaseParams.get_headNumSize(),
                                      tilingData.promptAttentionBaseParams.get_seqSize(),
                                      tilingData.promptAttentionBaseParams.get_headSize() *
                                      tilingData.promptAttentionBaseParams.get_headNumSize()});

    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShapeRect, typeSize, tilingData.transposeTilingDataRect);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionSetTilingData(gert::TilingContext* context,
    PromptFlashAttentionTilingData& tilingData) {
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::GetRectangleFactor(uint32_t seqFactorThreshold,
    std::queue<uint32_t>& sQueue, int32_t threshold) {
    for (int i = seqFactorThreshold; i >= threshold ; i = (i - threshold)) { // threshold 16
        sQueue.push(i);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::SetInputLayout(const char* layout){
    if (layout == nullptr){
        inputLayout = InputLayout::BSH;
        return ge::GRAPH_SUCCESS;
    }

    std::string layoutStr(layout);
    if (layoutStr == "") {
        inputLayout = InputLayout::BSH;
    } else if (layoutStr == "SH") {
        inputLayout = InputLayout::SH;
    } else if (layoutStr == "BSH") {
        inputLayout = InputLayout::BSH;
    } else if (layoutStr == "NSD") {
        inputLayout = InputLayout::NSD;
    } else if (layoutStr == "BSND") {
        inputLayout = InputLayout::BSND;
    } else if (layoutStr == "BNSD") {
        inputLayout = InputLayout::BNSD;
    } else if (layoutStr == "BNSD_BSND") { // BNSD_BSND 复用 BNSD 流程
        inputLayout = InputLayout::BNSD;
    } else {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::CheckInputDimAndHeadNum(ContextParamsForPFATiling& contextKeyParams, const uint32_t nQAttr, const uint32_t nKVAttr) {
    uint32_t nQ = nQAttr;
    uint32_t nKV = nKVAttr;
    if (nKVAttr == 0U) { // 检测到默认值，即客户未传入
        nKV = nQAttr;
    }

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    uint32_t queryShapeHeadNum = nQ;
    uint32_t keyShapeHeadNum = nKV;
    uint32_t valueShapeHeadNum = nKV;
    const uint32_t queryDim = queryShape->GetStorageShape().GetDimNum();
    const uint32_t keyDim = keyShape->GetStorageShape().GetDimNum();
    const uint32_t valueDim = valueShape->GetStorageShape().GetDimNum();
    const uint32_t nIdx = inputLayout == InputLayout::BNSD ? 1U : 2U; // BNSD: 1; BSND:2

    if (((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::BSND)) && (!enablePA)) {
        if ((queryDim == 4) && (keyDim == 4) && (valueDim == 4)) { // dim num: 4
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(nIdx);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(nIdx);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(nIdx);
        } else {
            OPS_LOG_E(contextKeyParams.opName, "input dim of q(%u), k(%u), v(%u) must be 4 for BNSD or BSND format!", queryDim, keyDim, valueDim);
            return false;
        }
    } else if ((inputLayout == InputLayout::NSD) && (!enablePA)) {
        if ((queryDim == 3) && (keyDim == 3) && (valueDim == 3)) { // dim num: 3
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(0);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(0);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(0);
        } else {
            OPS_LOG_E(contextKeyParams.opName, "input dim of q(%u), k(%u), v(%u) must be 3 for NSD format!", queryDim, keyDim, valueDim);
            return false;
        }
    }

    OPS_ERR_IF(nQ > 256U,  // head最大限制到256
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) should not be more than 256!", nQ),
                    return false);

    OPS_ERR_IF(queryShapeHeadNum != nQ,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) in query shape must be equal to numHeads(%u) in attr!", queryShapeHeadNum, nQ),
                    return false);
    OPS_ERR_IF(keyShapeHeadNum != nKV,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) in key shape do not match numKeyValueHeads(%u) in attr!", keyShapeHeadNum, nKV),
                    return false);
    OPS_ERR_IF(valueShapeHeadNum != nKV,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) in value shape do not match numKeyValueHeads(%u) in attr!", valueShapeHeadNum, nKV),
                    return false);
    return true;
}

bool PromptFlashAttentionTiling::SetTilingHeadNumRatio(ContextParamsForPFATiling& contextKeyParams,
                                                       const int32_t* numQueryHeads, const int32_t* numKeyValueHeads,
                                                       PromptFlashAttentionTilingData& tilingData) {
    const int32_t nQ = *numQueryHeads;
    const int32_t nKV = *numKeyValueHeads;

    if ((nQ < 0) || (nKV < 0)) {
        OPS_LOG_E(contextKeyParams.opName, "numHeads(%d) or numKeyValueHeads(%d) is negative!", nQ, nKV);
        return false;
    }

    if (!CheckInputDimAndHeadNum(contextKeyParams, nQ, nKV)) {
        return false;
    }

    if (nKV == 0) { // 检测到默认值，即客户未传入
        tilingData.promptAttentionBaseParams.set_headNumRatio(1);
        return true;
    }

    if (nQ % nKV != 0) {
        OPS_LOG_E(contextKeyParams.opName, "numHeads(%d) must be divisible by numKeyValueHeads(%d)!", nQ, nKV);
        return false;
    } else {
        if (nQ / nKV > 64) { // G不能大于64
            OPS_LOG_E(contextKeyParams.opName, "numHeads / numKeyValueHeads = %d, cannot be larger than 64", nQ / nKV);
            return false;
        }
        tilingData.promptAttentionBaseParams.set_headNumRatio(nQ / nKV);
        return true;
    }
}

bool PromptFlashAttentionTiling::CheckNonEmptyShapeExceptions(ContextParamsForPFATiling& contextKeyParams,
                                                              const gert::StorageShape* shape,
                                                              const std::string &sName) {
    OPS_ERR_IF(shape == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "%s shape is null.", sName.c_str()),
                    return true);
    OPS_ERR_IF(shape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Shape size of %s is overflow.", sName.c_str()),
                    return true);
    return false;
}

bool PromptFlashAttentionTiling::CheckActualSeqLength(ContextParamsForPFATiling& contextKeyParams, uint32_t b, uint32_t sQ, uint32_t sKV,
                                                      const gert::Tensor* actualSeqLenQ, const gert::Tensor* actualSeqLenKV,
                                                      InputLayout inLayout) {
    if (contextKeyParams.fromTilingSink != 0) {
        return true;
    }
    uint64_t actualLenDimsQ  = actualSeqLenQ  != nullptr ? actualSeqLenQ->GetShapeSize()  : 0;
    uint64_t actualLenDimsKV = actualSeqLenKV != nullptr ? actualSeqLenKV->GetShapeSize() : 0;
    bool inputActualSeqQ  = !((actualLenDimsQ  == 0) || (actualSeqLenQ  == nullptr) || (actualSeqLenQ->GetData<int64_t>()  == nullptr));
    bool inputActualSeqKV = !((actualLenDimsKV == 0) || (actualSeqLenKV == nullptr) || (actualSeqLenKV->GetData<int64_t>() == nullptr));
    int64_t actualSeqQSum = 0;
    int64_t actualSeqTmp = 0;

    // SH格式单独校验
    if (inLayout == InputLayout::SH) {
        if (inputActualSeqQ) {
            for (uint32_t i = LOOP_BEGIN_NUM; i < b; ++i) {
                actualSeqQSum = actualSeqQSum + static_cast<uint32_t>(actualSeqLenQ->GetData<int64_t>()[i]);
            }
            OPS_ERR_IF(actualSeqQSum != sQ,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "SH format sum of actual_seq_q(%ld) do not match s_q(%u)!", actualSeqQSum, sQ),
                            return false);
        }
        return true;
    }

    if (inputActualSeqQ) {
        OPS_ERR_IF(b != actualLenDimsQ,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dim(%lu) of actual_seq_lengths must equal to batch size(%u)!", actualLenDimsQ, b),
                        return false);
        for (uint32_t i = LOOP_BEGIN_NUM; i < b; ++i) {
            actualSeqTmp = static_cast<int64_t>(actualSeqLenQ->GetData<int64_t>()[i]);
            OPS_ERR_IF(actualSeqTmp < 0 || actualSeqTmp > sQ,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths[%u](%ld) must be in range[0, %u]!", i, actualSeqTmp, sQ),
                            return false);
        }
    }
    if (inputActualSeqKV) {
        OPS_ERR_IF(b != actualLenDimsKV,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dim(%lu) of actual_seq_lengths_kv must equal to batch size(%u)!", actualLenDimsKV, b),
                        return false);
        for (uint32_t i = LOOP_BEGIN_NUM; i < b; ++i) {
            actualSeqTmp = static_cast<int64_t>(actualSeqLenKV->GetData<int64_t>()[i]);
            if (contextKeyParams.isKvContinuous == 1) {
                if (!enablePA) {
                    OPS_ERR_IF(actualSeqTmp < 0 || actualSeqTmp > sKV,
                                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths_kv[%u](%ld) must be in range[0, %u]!", i, actualSeqTmp, sKV),
                                return false);
                } else {
                    OPS_ERR_IF(actualSeqTmp < 0,
                                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths_kv[%u](%ld) must >= 0", i, actualSeqTmp),
                                return false);
                }
            } else {
                if ((inLayout == InputLayout::BSND) || (inLayout == InputLayout::BSH)) {
                    OPS_ERR_IF(actualSeqTmp < 0 || actualSeqTmp > contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1),
                                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths_kv[%u](%ld) must be in range[0, %li]!", i, actualSeqTmp,
                                                                    contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1)),
                                    return false);
                } else {
                    OPS_ERR_IF(actualSeqTmp < 0 || actualSeqTmp > contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2),
                                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Actual_seq_lengths_kv[%u](%ld) must be in range[0, %li]!", i, actualSeqTmp,
                                                                    contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2)),
                                    return false);
                }
            }
        }
    }

    return true;
}

bool PromptFlashAttentionTiling::CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape *pseShiftShape, uint32_t b, uint32_t n, uint32_t s1, uint32_t s2) {
    if (contextKeyParams.fromTilingSink != 0) {
        return true;
    }
    pseShiftElemType = contextKeyParams.pseShiftDataType;

    OPS_ERR_IF((curShortSocName == platform_ascendc::SocVersion::ASCEND310P),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "not support 310P when pse is not null"),
                    return false);

    OPS_ERR_IF((inputType == ge::DT_FLOAT16 && pseShiftElemType != ge::DT_FLOAT16),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "q type is fp16, but pse shift type is not fp16, pse shift type = %ld", (int64_t)pseShiftElemType),
                    return false);

    OPS_ERR_IF((inputType == ge::DT_BF16 && pseShiftElemType != ge::DT_BF16),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "q type is bf16, but pse shift type is not bf16, pse shift type = %ld", (int64_t)pseShiftElemType),
                    return false);

    OPS_ERR_IF((inputType == ge::DT_INT8 && pseShiftElemType != ge::DT_FLOAT16),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "q type is int8, but pse shift type is not fp16, pse shift type = %ld", (int64_t)pseShiftElemType),
                    return false);

    // 暂不支持D超大
     OPS_ERR_IF((n == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "num head is zero"),
                    return false);

    // pse 为空，则不需要做pse的动作
    if (((pseShiftShape != nullptr) && (pseShiftShape->GetStorageShape().GetShapeSize() == 0)) ||
        (pseShiftShape == nullptr)) {
            usePseShift = 0;
            return true;
    }

    if (pseShiftElemType == ge::DT_FLOAT16) {
        pseShiftElemSize = FLOAT16SIZE;
    } else if (pseShiftElemType == ge::DT_BF16) {
        pseShiftElemSize = BFLOAT16SIZE;
    }
    pseShiftTypeByteNum = BYTE_BLOCK / pseShiftElemSize;

    uint32_t pseShiftDim = pseShiftShape->GetStorageShape().GetDimNum();
    OPS_ERR_IF((pseShiftDim != PSE_SHIFT_DIM),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "pse shift shape must be 4 dimension, rather than %u dimension", pseShiftDim),
                    return false);

    pseShiftBatch = pseShiftShape->GetStorageShape().GetDim(0);
    uint32_t pseShiftN = pseShiftShape->GetStorageShape().GetDim(1);  // 1: N的维度
    pseShiftS1 = pseShiftShape->GetStorageShape().GetDim(2);          // 2: S1的维度
    pseShiftS2 = pseShiftShape->GetStorageShape().GetDim(3);          // 3: S2的维度
    OPS_ERR_IF(((pseShiftBatch != 1 && pseShiftBatch != b) || (pseShiftN != n) ||
                    (pseShiftS1 < s1) || (pseShiftS2 < s2)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "pse shift shape must be [1 or %u, %u, >=%u, >=%u], but now it is [%u, %u, %u, %u]",
                    b, n ,s1, s2, pseShiftBatch, pseShiftN, pseShiftS1, pseShiftS2),
                    return false);

    return true;
}

bool PromptFlashAttentionTiling::CheckPATypeAndShape(ContextParamsForPFATiling& contextKeyParams,
    const gert::Tensor* actualSeqLenKV, int32_t b, int32_t n, int32_t h, int32_t headNumRatio) {
    const int32_t* blockSize = contextKeyParams.blockSize;
    OPS_ERR_IF((*blockSize % BLOCK_SIZE_BASE != 0 || *blockSize < BLOCK_SIZE_BASE || *blockSize > BLOCK_SIZE_MAX),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "block size(%d) should be a multiple of %d, and can't greater than %d when PA enable",
                    *blockSize, BLOCK_SIZE_BASE, BLOCK_SIZE_MAX),
                    return false);

    const gert::StorageShape* blockTableShape = contextKeyParams.blockTableShape;
    OPS_ERR_IF((((blockTableShape != nullptr) && (blockTableShape->GetStorageShape().GetShapeSize() == 0)) ||
                (blockTableShape == nullptr)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "blockTable can't be empty when PA enable"),
                    return false);
    int32_t blockTableDim1 = (int32_t)(blockTableShape->GetStorageShape().GetDim(0));
    blockTableDim2 = (int32_t)(blockTableShape->GetStorageShape().GetDim(1));
    // 可能blockTableDim2 > maxBlockNumPerBatch，kernel侧在blockTable中索引block id时，应该以blockTableDim2作为第二维
    // 但用于mask S2轴的校验，应该仍然用 maxBlockNumPerBatch * tempBlockSize 作为校验基准

    if (contextKeyParams.fromTilingSink != 0) {
        tmpS2 = blockTableDim2 * (*blockSize); // tiling下沉场景，需要计算workspace，此时以blockTableDim2 * blockSize作为S2
        return true;
    }
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    int32_t keyDim = keyShape->GetStorageShape().GetDimNum();
    int32_t valueDim = valueShape->GetStorageShape().GetDimNum();
    OPS_ERR_IF(keyDim != valueDim,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim num of key(%d) and value(%d) are inconsistent when PA enable", keyDim, valueDim),
                    return false);
    OPS_ERR_IF(((keyDim != 3) && (keyDim != 4)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of key and value must be 3 or 4 when PA enable"),
                    return false);

    int32_t keyDim1 = keyShape->GetStorageShape().GetDim(0);  // block_num_sum
    int32_t keyDim2 = keyShape->GetStorageShape().GetDim(1);
    int32_t keyDim3 = keyShape->GetStorageShape().GetDim(2);
    int32_t keyDim4 = 0;
    int32_t valueDim1 = valueShape->GetStorageShape().GetDim(0);
    int32_t valueDim2 = valueShape->GetStorageShape().GetDim(1);
    int32_t valueDim3 = valueShape->GetStorageShape().GetDim(2);
    int32_t valueDim4 = 0;
    int32_t tempBlockSize = keyDim2;
    int32_t tempH = keyDim3;
    int32_t tempN = 0;
    int32_t tempD = 0;

    if (keyDim == 4) {  // dim num: 4
        keyDim4 = keyShape->GetStorageShape().GetDim(3);  // 3: 第三维
        valueDim4 = valueShape->GetStorageShape().GetDim(3);  // 3: 第三维
        tempN = keyDim2;
        tempBlockSize = keyDim3;
        tempD = keyDim4;
    }

    OPS_ERR_IF(((keyDim1 != valueDim1) || (keyDim2 != valueDim2) || (keyDim3 != valueDim3) || (keyDim4 != valueDim4)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of key and value are inconsistent when PA enable"),
                    return false);

    if (keyDim == 3) {  // dim num: 3
        PAlayoutType = 1;  // 三维则PAlayoutType = 1
        OPS_ERR_IF(((tempBlockSize != *blockSize) || (tempH * headNumRatio != h)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of key [%d, %d, %d] is wrong, which should be [*, %d, %d] when PA enable", keyDim1,
                    keyDim2, keyDim3, *blockSize, h / headNumRatio),  // headNumRatio 赋值时已保证不会为0
                    return false);
        // pa场景BSH输入, 要求KV矩阵的h不超过65535, 前面已校验K/V的dim和dim3相等, 所以这里只校验K矩阵
        OPS_ERR_IF(keyDim3 > 65535,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "layout of key/value is BSH, the h of key/value %d should not > 65535 when PA enable",
                    keyDim3),
                    return false);
    } else {
        PAlayoutType = 0;  // 四维则PAlayoutType = 0
        OPS_ERR_IF(((tempN * headNumRatio != n) || (tempBlockSize != *blockSize) || (tempD != (h / n))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of key [%d, %d, %d, %d] is wrong, which should be [*, %d, %d, %d] when PA enable",
                    keyDim1, keyDim2, keyDim3, keyDim4, n / headNumRatio, *blockSize, (h / n)),
                    return false);
    }

    std::string layoutStr(contextKeyParams.layout);
    if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND" || layoutStr == "NSD") {
        OPS_ERR_IF(((keyDim != 3) && (keyDim != 4)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the layout of query is %s, key and value layout should be [*, %d, %d] or [*, %d, %d, %d] when PA enable",
                    layoutStr.c_str(), *blockSize, h, n, *blockSize, (h / n)),
                    return false);
    } else if (layoutStr == "BSH" || layoutStr == "BSND") {
        OPS_ERR_IF(keyDim != 3,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the layout of query is %s, key and value layout should be [*, %d, %d] when PA enable",
                    layoutStr.c_str(), *blockSize, h),
                    return false);
    } else {
        OPS_LOG_E(contextKeyParams.opName, "unsupported input data layout when PA enable");
        return false;
    }

    ge::DataType blockTableType = contextKeyParams.blockTableType;
    OPS_ERR_IF((blockTableType != ge::DT_INT32),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "blockTable only support int32 when PA enable"),
                    return false);

    int32_t blockTableDim = (int32_t)(blockTableShape->GetStorageShape().GetDimNum());
    OPS_ERR_IF(blockTableDim != 2,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the dim of block table must be 2 when PA enable"),
                    return false);

    int32_t actualSeqKVPerBatch = 0;
    int32_t blockNumPerBatch = 0;
    int32_t blockNumValid = 0;
    int32_t maxBlockNumPerBatch = 0;
    for (int32_t i = 0; i < b; i++) {
        actualSeqKVPerBatch = static_cast<int32_t>(actualSeqLenKV->GetData<int64_t>()[i]);
        blockNumPerBatch = (actualSeqKVPerBatch + *blockSize - 1) / *blockSize;
        blockNumValid += blockNumPerBatch;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }

    OPS_ERR_IF(((blockTableDim1 != b) || (blockTableDim2 < maxBlockNumPerBatch)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "block table shape should be [%d, >=%d], now is [%d, %d] when PA enable",
                    b, maxBlockNumPerBatch, blockTableDim1, blockTableDim2),
                    return false);

    OPS_ERR_IF((keyDim1 < blockNumValid),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "the first dim of key(%d) should not less than valid block num(%d) when PA enable",
                    keyDim1, blockNumValid),
                    return false);

    PABlockNumSum = keyDim1;
    tmpS2 = maxBlockNumPerBatch * tempBlockSize;
    return true;
}

bool PromptFlashAttentionTiling::CheckAttenMaskShape(ContextParamsForPFATiling& contextKeyParams,
                                                     const int32_t* sparseMode,
                                                     const gert::StorageShape* attenMaskShape,
                                                     const uint32_t sQ, const uint32_t sK, const uint32_t batchSize) {
    if (contextKeyParams.fromTilingSink != 0) {
        return true;
    }
    // attention mask 空Tensor场景，不需要根据 sparse mode 值校验 attention mask shape
    if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0)) ||
        (attenMaskShape == nullptr)) {
        return true;
    }
    uint32_t attenMaskDim = attenMaskShape->GetStorageShape().GetDimNum();
    uint32_t attenMaskBatch = 1U;
    uint32_t attenMaskS1, attenMaskS2;
    int32_t checkShapeRet = 0;
    if (attenMaskDim == ATTENTION_MASK_DIM2) {
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(1);
        if ((sparseMode == nullptr) || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_NO_MASK) ||
            (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
            checkShapeRet = (attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
                            (attenMaskBatch == 1 || attenMaskBatch == batchSize);
        }

        if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN || *sparseMode == SPARSE_MODE_BAND)) {
            checkShapeRet = attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE &&
                            attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE;
        }
    } else if (attenMaskDim == ATTENTION_MASK_DIM3) {
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(2);  // dim 为 3 时，第 2 维是 S2
        if ((sparseMode == nullptr) || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_NO_MASK) ||
            (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
            checkShapeRet = (attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
                            (attenMaskBatch == 1 || attenMaskBatch == batchSize);
        }
        if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN || *sparseMode == SPARSE_MODE_BAND)) {
            checkShapeRet = attenMaskBatch == 1 &&
                            attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE &&
                            attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE;
        }
    } else if (attenMaskDim == ATTENTION_MASK_DIM4) {
        uint32_t attenMaskN = 1U;
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskN = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(2);  // dim 为 4 时，第 2 维是 S1
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(3);  // dim 为 4 时，第 3 维是 S2
        if ((sparseMode == nullptr) || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_NO_MASK) ||
            (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
            checkShapeRet = (attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
                            (attenMaskBatch == 1 || attenMaskBatch == batchSize);
        }
        if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN || *sparseMode == SPARSE_MODE_BAND)) {
            checkShapeRet = attenMaskBatch == 1 && attenMaskN == 1 &&
                            attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE &&
                            attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE;
        }
    } else {
        OPS_LOG_E(contextKeyParams.opName, "attenMask dim(%u) must be 2 or 3 or 4!", attenMaskDim);
        return false;
    }
    if ((sparseMode == nullptr) || ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK)) ||
        ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_ALL_MASK))) {
        OPS_ERR_IF(checkShapeRet != 1, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "attenMask batch(%u) must be 1 or %u, attenMask Q_S(%u) must be larger than sQ(%u), attenMask KV_S(%u) must be larger than sK(%u), please check",
            attenMaskBatch, batchSize, attenMaskS1, sQ, attenMaskS2, sK), return false);
    }
    if ((sparseMode != nullptr) && ((*sparseMode == SPARSE_MODE_LEFT_UP) || (*sparseMode == SPARSE_MODE_RIGHT_DOWN) || (*sparseMode == SPARSE_MODE_BAND))) {
        OPS_ERR_IF(checkShapeRet != 1, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "attenMask shape must be (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) when sparse mode = %d",
            *sparseMode), return false);
    }
    return true;
}

bool PromptFlashAttentionTiling::CheckAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* antiquantScaleShape,
                                                           const gert::StorageShape* antiquantOffsetShape, const uint32_t n, const uint32_t d, const uint32_t h,
                                                           PromptFlashAttentionTilingData& tilingData) {
    OPS_ERR_IF(contextKeyParams.antiquantScale == nullptr || antiquantScaleShape == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale is nullptr"),
                    return false);
    tilingData.promptAttentionBaseParams.set_isAntiPerchannel(1);
    if (antiquantScaleShape->GetStorageShape().GetDimNum() == 1) {
        tilingData.promptAttentionBaseParams.set_isAntiPerchannel(0);
        OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != 2,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim[0] = %ld, but it should be 2 under Per-Tensor mode!", antiquantScaleShape->GetStorageShape().GetDim(0)),
                        return false);
        OPS_ERR_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDim(0) != 2,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim[0] = %ld, but it should be 2 under Per-Tensor mode!", antiquantOffsetShape->GetStorageShape().GetDim(0)),
                        return false);
    } else {
        if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDimNum() != 4,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim num[%zu] should be 4 if layout is BNSD or NSD!", antiquantScaleShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != 2 || antiquantScaleShape->GetStorageShape().GetDim(1) != n ||
                            antiquantScaleShape->GetStorageShape().GetDim(2) != 1 || antiquantScaleShape->GetStorageShape().GetDim(3) != d,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim [%ld, %ld, %ld, %ld] is wrong!", antiquantScaleShape->GetStorageShape().GetDim(0),
                            antiquantScaleShape->GetStorageShape().GetDim(1), antiquantScaleShape->GetStorageShape().GetDim(2), antiquantScaleShape->GetStorageShape().GetDim(3)),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDimNum() != 4,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim num[%zu] should be 4 if layout is BNSD or NSD!", antiquantOffsetShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && (antiquantOffsetShape->GetStorageShape().GetDim(0) != 2 || antiquantOffsetShape->GetStorageShape().GetDim(1) != n ||
                            antiquantOffsetShape->GetStorageShape().GetDim(2) != 1 || antiquantOffsetShape->GetStorageShape().GetDim(3) != d),
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim [%ld, %ld, %ld, %ld] is wrong!", antiquantOffsetShape->GetStorageShape().GetDim(0),
                            antiquantOffsetShape->GetStorageShape().GetDim(1), antiquantOffsetShape->GetStorageShape().GetDim(2), antiquantOffsetShape->GetStorageShape().GetDim(3)),
                            return false);
        } else if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::SH)) {
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDimNum() != 2,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim num[%zu] should be 2 if layout is BSH or SH!", antiquantScaleShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != 2 || antiquantScaleShape->GetStorageShape().GetDim(1) != h,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim [%ld, %ld] is wrong!", antiquantScaleShape->GetStorageShape().GetDim(0),
                            antiquantScaleShape->GetStorageShape().GetDim(1)),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDimNum() != 2,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim num[%zu] should be 2 if layout is BSH or SH!", antiquantOffsetShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && (antiquantOffsetShape->GetStorageShape().GetDim(0) != 2 || antiquantOffsetShape->GetStorageShape().GetDim(1) != h),
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim [%ld, %ld] is wrong!", antiquantOffsetShape->GetStorageShape().GetDim(0),
                            antiquantOffsetShape->GetStorageShape().GetDim(1)),
                            return false);
        } else if (inputLayout == InputLayout::BSND) {
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDimNum() != 3,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim num[%zu] should be 3 if layout is BSND!", antiquantScaleShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != 2 || antiquantScaleShape->GetStorageShape().GetDim(1) != n ||
                            antiquantScaleShape->GetStorageShape().GetDim(2) != d,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim [%ld, %ld, %ld] is wrong!", antiquantScaleShape->GetStorageShape().GetDim(0),
                            antiquantScaleShape->GetStorageShape().GetDim(1), antiquantScaleShape->GetStorageShape().GetDim(2)),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDimNum() != 3,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim num[%zu] should be 3 if layout is BSND!", antiquantOffsetShape->GetStorageShape().GetDimNum()),
                            return false);
            OPS_ERR_IF(antiquantOffsetShape != nullptr && (antiquantOffsetShape->GetStorageShape().GetDim(0) != 2 || antiquantOffsetShape->GetStorageShape().GetDim(1) != n ||
                            antiquantOffsetShape->GetStorageShape().GetDim(2) != d),
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim [%ld, %ld, %ld] is wrong!", antiquantOffsetShape->GetStorageShape().GetDim(0),
                            antiquantOffsetShape->GetStorageShape().GetDim(1), antiquantOffsetShape->GetStorageShape().GetDim(2)),
                            return false);
        }
    }

    return true;
}

ge::graphStatus PromptFlashAttentionTiling::CheckPostQuantParams(const ContextParamsForPFATiling& contextKeyParams, uint32_t h, uint32_t n) {
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    const gert::StorageShape* quantOffset2Shape = contextKeyParams.offset2Shape;
    const ge::DataType quantScale2Type = contextKeyParams.quantScale2Type;
    const ge::DataType quantOffset2Type = contextKeyParams.quantOffset2Type;
    int64_t quantScale2ShapeSize = 0;
    int64_t quantOffset2ShapeSize = 0;
    uint32_t quantD = 0;

    if (outputType == ge::DT_INT8) {
        // 基础校验：quantScale2必须输入且非空tensor
        OPS_ERR_IF(quantScale2Shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale is nullptr when output type is int8."),
                return ge::GRAPH_FAILED);
        quantScale2ShapeSize = quantScale2Shape->GetStorageShape().GetShapeSize();
        quantD = quantScale2ShapeSize / n;
        OPS_ERR_IF(quantScale2ShapeSize == 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "quant_scale2 is empty tensor when output type is int8."),
                return ge::GRAPH_FAILED);

        // 交叉特性校验：后quant per-channel暂不支持左padding、ring attention、D非32B对齐
        if (quantScale2ShapeSize != 1) {
            OPS_ERR_IF((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant per-channel do not support left padding."),
                    return ge::GRAPH_FAILED);
            OPS_ERR_IF(contextKeyParams.isSoftMaxLseEnable == true,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant per-channel do not support ring attention."),
                    return ge::GRAPH_FAILED);
            OPS_ERR_IF(quantD % BYTE_BLOCK != 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant per-channel do not support D(%u) non-32-byte aligned.", quantD),
                    return ge::GRAPH_FAILED);
        }

        // dtype校验
        OPS_ERR_IF((quantScale2Type != ge::DT_BF16) && (quantScale2Type != ge::DT_FLOAT),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale dtype(%d) only support bf16 and fp32.", quantScale2Type),
                return ge::GRAPH_FAILED);
        OPS_ERR_IF((quantOffset2Shape != nullptr) && (quantScale2Type != quantOffset2Type),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale dtype(%d) and offset dtype(%d) must be consistent.", quantScale2Type, quantOffset2Type),
                return ge::GRAPH_FAILED);
        OPS_ERR_IF((inputType != ge::DT_BF16) && (quantScale2Type == ge::DT_BF16),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale and offset support bf16 only if input dtype(%d) is bf16.", inputType),
                return ge::GRAPH_FAILED);

        // shape校验
        if (quantOffset2Shape != nullptr) {
            quantOffset2ShapeSize = quantOffset2Shape->GetStorageShape().GetShapeSize();
            OPS_ERR_IF(quantScale2ShapeSize != quantOffset2ShapeSize,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "quant_scale2 dimension multiply result(%ld) do not equal quant_offset2 dimension multiply result(%ld).",
                    quantScale2ShapeSize, quantOffset2ShapeSize), return ge::GRAPH_FAILED);
        }
        OPS_ERR_IF((quantScale2ShapeSize != 1) && (quantScale2ShapeSize != h),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "post quant scale2/offset2 dimension multiply result only support 1 and H(%u), now is (%ld). "
                "Maybe the shape of scale2/offset2 do not match that of query, or D is not 32 Byte aligned, "
                "which post quant per-channel do not support.", h, quantScale2ShapeSize), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::AdjustBasicBlock(PromptFlashAttentionTilingData& tilingData,
                                                             uint32_t& sOuterFactor) {
    PromptAttentionBaseParams* baseParams = &tilingData.promptAttentionBaseParams;
    uint32_t headNumSize = baseParams->get_headNumSize();
    uint32_t sCoreNum = (coreNum / headNumSize);
    uint32_t sOuterBlockNum = (maxQuerySeq + sOuterFactor - 1U) / sOuterFactor;
    if ((coreNum % headNumSize == 0) && (sCoreNum > 1) && (sOuterBlockNum % sCoreNum == 0) &&
        sOuterBlockNum / sCoreNum == 1) {
      // n方向全部开核; s方向开多核且每个核处理仅处理一个Souter，此时，Souter切分成两块，供负载均衡优化使用。
      // 为了保证基本块是typeByteNum的整数倍
      sOuterFactor = (sOuterFactor / 2 + typeByteNum - 1) / typeByteNum * typeByteNum;  // split outer: 2
    }
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::Align(uint32_t &num) {
    num = (num + typeByteNum - 1) / typeByteNum * typeByteNum;
}

// Code for ut, no pratical use
ge::graphStatus PromptFlashAttentionTiling::GetBasicShape310P(uint32_t &b,
                                                              uint32_t &bKV,
                                                              uint32_t &s,
                                                              uint32_t &h,
                                                              uint32_t &seqInnerSize,
                                                              const gert::StorageShape *queryShape,
                                                              const gert::StorageShape *keyShape,
                                                              const uint32_t n,
                                                              size_t actualLenDims,
                                                              size_t actualLenDimsKV) {
    OPS_ERR_IF(queryShape == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "queryShape is null."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(keyShape == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "keyShape is null."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(n == 0,
                    OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "n is 0."),
                    return ge::GRAPH_FAILED);
    if (inputLayout == InputLayout::NSD) {
      uint32_t d;
      b = 1;
      bKV = 1;
      s = queryShape->GetStorageShape().GetDim(1);
      seqInnerSize = keyShape->GetStorageShape().GetDim(1);
      d = queryShape->GetStorageShape().GetDim(2); // dim num: 2
      Align(d);
      h = (d * n);
      return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::BNSD) {
      uint32_t d;
      b = queryShape->GetStorageShape().GetDim(0);
      bKV = keyShape->GetStorageShape().GetDim(0);
      s = queryShape->GetStorageShape().GetDim(2); // dim num: 2
      seqInnerSize = keyShape->GetStorageShape().GetDim(2); // dim num: 2
      d = queryShape->GetStorageShape().GetDim(3); // dim num: 3
      Align(d);
      h = (queryShape->GetStorageShape().GetDim(1) * d);
      return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::SH) {
      b = ((actualLenDims == 0) ? 1 : actualLenDims); // SH格式未输入actual_seq时，按照batch=1处理
      bKV = ((actualLenDimsKV == 0) ? 1 : actualLenDimsKV); // SH格式未输入actual_seqkv时，按照batch=1处理
      uint32_t d;
      s = queryShape->GetStorageShape().GetDim(0);
      h = queryShape->GetStorageShape().GetDim(1);
      seqInnerSize = keyShape->GetStorageShape().GetDim(0);

      Align(s);
      Align(seqInnerSize);
      d = (h / n);
      Align(d);
      h = (d * n);
      return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::BSH) {
      uint32_t d;
      b = queryShape->GetStorageShape().GetDim(0);
      bKV = keyShape->GetStorageShape().GetDim(0);
      s = queryShape->GetStorageShape().GetDim(1);
      h = queryShape->GetStorageShape().GetDim(2); // dim num: 2
      seqInnerSize = keyShape->GetStorageShape().GetDim(1);
      d = (h / n);
      Align(d);
      h = (d * n);
      return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::BSND) {
      uint32_t d;
      b = (queryShape->GetStorageShape().GetDim(0));
      bKV = (keyShape->GetStorageShape().GetDim(0));
      s = (queryShape->GetStorageShape().GetDim(1));
      d = (queryShape->GetStorageShape().GetDim(INDEX_3));
      seqInnerSize = (keyShape->GetStorageShape().GetDim(1));
      Align(d);
      h = (d * n);
      return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus PromptFlashAttentionTiling::GetAndCheckEmptyQueryShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *queryShape) const {
    OPS_ERR_IF(queryShape == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryShape is null."),
               return ge::GRAPH_FAILED);
    uint32_t b;
    uint32_t n;
    uint32_t s;
    uint32_t d;
    uint32_t h = 0;
    if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
        if (queryShape->GetStorageShape().GetDimNum() == 3) { // dim num: 3
            b = 1;
            n = queryShape->GetStorageShape().GetDim(0);
            s = queryShape->GetStorageShape().GetDim(1);
            d = queryShape->GetStorageShape().GetDim(2); // dim num: 2
        } else {
            b = (queryShape->GetStorageShape().GetDim(0));
            n = (queryShape->GetStorageShape().GetDim(1));
            s = (queryShape->GetStorageShape().GetDim(2)); // dim num: 2
            d = (queryShape->GetStorageShape().GetDim(3)); // dim num: 3
        }
    } else if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) ||
        (inputLayout == InputLayout::SH)) {
        if (queryShape->GetStorageShape().GetDimNum() == NUM_2) { // dim num: 2
            b = 1; // 按照batch=1处理
            s = (queryShape->GetStorageShape().GetDim(0));
            h = (queryShape->GetStorageShape().GetDim(1));
        } else if (queryShape->GetStorageShape().GetDimNum() == 3) { // 3 : BSH
            b = (queryShape->GetStorageShape().GetDim(0));
            s = (queryShape->GetStorageShape().GetDim(1));
            h = (queryShape->GetStorageShape().GetDim(2)); // dim num: 2
        } else { // BSND
            b = (queryShape->GetStorageShape().GetDim(0));
            s = (queryShape->GetStorageShape().GetDim(1));
            n = (queryShape->GetStorageShape().GetDim(2)); // dim num: 2
            d = (queryShape->GetStorageShape().GetDim(3)); // dim num: 3
        }
    } else {
        return ge::GRAPH_FAILED;
    }
    OPS_ERR_IF(b > BLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
               "batch size should <= 65536, but batch size = %u", b), return ge::GRAPH_FAILED);
    if (s > SLIMIT) {
        OPS_LOG_W(contextKeyParams.opName,
                   "seq should <= 20m, but seq = %u", s);
    }
    if (inputLayout == InputLayout::BSH || inputLayout == InputLayout::SH) {
        OPS_ERR_IF(h > DLIMIT * NLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                   "h should <= 512 * 256, but h = %u", h), return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(n > NLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "n should <= 256, but n = %u", n), return ge::GRAPH_FAILED);
        OPS_ERR_IF(d > DLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "D should <= 512, but d = %u", d), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::RunBigKernelTilingWithParams(ContextParamsForPFATiling& contextKeyParams,
                                            uint64_t& tilingKey,
                                            uint32_t& blockDimToBeSet,
                                            PromptFlashAttentionTilingData& tilingData) {
    uint64_t l0CSize;
    uint64_t l1Size;
    uint64_t ubSize;
    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    contextKeyParamsPtr = &contextKeyParams;        // 后续整改，将contextKeyParams写成类的成员变量

    OPS_ERR_IF(compileInfoPtr == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);

    ubSize = compileInfoPtr->ubSize;
    l1Size = compileInfoPtr->l1Size;
    l0CSize = compileInfoPtr->l0CSize;

    coreNum = compileInfoPtr->aivNum;
    OPS_ERR_IF(coreNum == 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "coreNum is 0"),
                    return ge::GRAPH_FAILED);
    aivNum = compileInfoPtr->aivNum;
    OPS_ERR_IF(aivNum == 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "aivNum is 0"),
                    return ge::GRAPH_FAILED);
    aicNum = compileInfoPtr->aicNum;
    OPS_ERR_IF(aicNum == 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "aicNum is 0"),
                    return ge::GRAPH_FAILED);
    curShortSocName = compileInfoPtr->socShortName;
    defaultSysWorkspaceSize = compileInfoPtr->defaultSysWorkspaceSize;

    ascendPlatformInfo.socVersion = compileInfoPtr->socShortName;
    ascendPlatformInfo.l1Size = compileInfoPtr->l1Size;
    ascendPlatformInfo.l0CSize = compileInfoPtr->l0CSize;
    ascendPlatformInfo.l0ASize = compileInfoPtr->l0ASize;
    ascendPlatformInfo.l0BSize = compileInfoPtr->l0BSize;
    ascendPlatformInfo.ubSize = compileInfoPtr->ubSize;
    OPS_LOG_I(contextKeyParams.opName,
            "ascendPlatformInfo:aivNum = %d, aicNum = %d, l1Size = %lu, l0CSize = %lu, l0ASize = %lu, l0BSize = %lu, ubSize = %lu!",
            aivNum, aicNum, ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize, ascendPlatformInfo.l0ASize, ascendPlatformInfo.l0BSize, ascendPlatformInfo.ubSize);
    int32_t outputDataTypeSize = FLOAT32SIZE;
    outputType = contextKeyParams.outputDataType;
    inputType = contextKeyParams.inputDataType;
    if (inputType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_INT8) {
        enableKvAntiquant = true;
    }
    if (inputType == ge::DT_FLOAT16) {
        dataTypeSize = FLOAT16SIZE;
    } else if (inputType == ge::DT_BF16) {
        dataTypeSize = BFLOAT16SIZE;
    } else if (inputType == ge::DT_INT8) {
        dataTypeSize = INT8SIZE;
    }
    if (outputType == ge::DT_FLOAT16) {
        outputDataTypeSize = FLOAT16SIZE;
    } else if (outputType == ge::DT_BF16) {
        outputDataTypeSize = BFLOAT16SIZE;
    } else if (outputType == ge::DT_INT8) {
        outputDataTypeSize = INT8SIZE;
    }

    OPS_ERR_IF(((inputType == ge::DT_FLOAT) || (outputType == ge::DT_FLOAT)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "inputType(%d) and outputType(%d) can not be DT_FLOAT", inputType, outputType),
                    return ge::GRAPH_FAILED);

    const int64_t* innerPrecisePtr = contextKeyParams.innerPrecisePtr;

    innerPrecise = innerPrecisePtr ? *innerPrecisePtr : HIGH_PERFORMANCE; // 910B默认高性能, 310P的高性能指的是高精度(不走近似计算)

    if (innerPrecise >= 4) { //0 : 复数无效 4 : 大于等于4无效 0,1,2,3是innerPrecise有效值
        OPS_LOG_W(contextKeyParams.opName, "innerPrecise [%d] should be 0,1,2,3, please check.", innerPrecise);
    }
    // 判断innerPrecise的bit1位，是否需要行无效修正
    if ((innerPrecise >> 1) & 1) {
        tilingData.promptAttentionBaseParams.set_isRowInvalid(1U);
    } else {
        tilingData.promptAttentionBaseParams.set_isRowInvalid(0U);
    }
    // 判断innerPrecise的bit0位，高性能或高精度模式
    innerPrecise = ((innerPrecise >> 0) & 1) ? HIGH_PERFORMANCE : HIGH_PRECISION;
    OPS_ERR_IF(((innerPrecise != HIGH_PERFORMANCE) && (innerPrecise != HIGH_PRECISION)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "precision mode[%lu] should be 0 or 1", innerPrecise),
                    return ge::GRAPH_FAILED); // 当前只支持高精度0和高性能1
    if (inputType != ge::DT_FLOAT16) {
        OPS_LOG_W(contextKeyParams.opName,
            "innerPrecise will not take effect when input type is %d!", inputType);
    }

    // fp16 pse强制走高精度模式
    if ((contextKeyParams.pseShift != nullptr) && (inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) {
        innerPrecise = HIGH_PRECISION;
        OPS_LOG_W(contextKeyParams.opName, "when the input is fp16, the mode is forcibly switched to high-precision!");
    }

    // blockTable非空即认为是PA场景
    if (contextKeyParams.blockTable != nullptr) {
        enablePA = true;
        OPS_ERR_IF(contextKeyParams.blockSize == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "blockSize can't be null when PA enable"),
                    return ge::GRAPH_FAILED);
    }

    if (enablePA) {
        OPS_ERR_IF(curShortSocName == platform_ascendc::SocVersion::ASCEND310P,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "not support 310P when blockTable is not null"),
                    return ge::GRAPH_FAILED);

        OPS_ERR_IF(contextKeyParams.isKvContinuous == 0,  // 与左padding互斥的拦截已在FIA中实现
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "not support tensorlist when blockTable is not null"),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF(enableKvAntiquant,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "not support antiquant when blockTable is not null"),
                    return ge::GRAPH_FAILED);
    }

    if ((curShortSocName == platform_ascendc::SocVersion::ASCEND310P) && innerPrecise == HIGH_PRECISION) {
        softmaxDataTypeNZ_ = FLOAT16SIZE;
        innerPrecise = HIGH_PERFORMANCE;
    }
    if (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) ||
        (inputType == ge::DT_INT8)) {
        softmaxDataTypeSize = FLOAT16SIZE; // 默认为fp32的size
    }

    uint32_t maskElemSize = dataTypeSize;
    if (contextKeyParams.attentionMask != nullptr) {
        auto maskDataType = contextKeyParams.maskDataType;
        if (maskDataType == ge::DT_FLOAT16) {
            maskElemSize = FLOAT16SIZE;
        }
        else if (maskDataType == ge::DT_FLOAT) {
            maskElemSize = FLOAT32SIZE;
        }
        else if (maskDataType == ge::DT_BOOL) {
            maskElemSize = BOOLSIZE;
        }
        else if (maskDataType == ge::DT_INT8) { // 适配静态图模式，bool型 attentionmask 被转为int8
            maskElemSize = INT8SIZE;
        }
        else if (maskDataType == ge::DT_UINT8) {
            maskElemSize = UINT8SIZE;
        }
        // fp32 mask type 不支持
        OPS_ERR_IF(maskDataType == ge::DT_FLOAT,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "maskType[%d] should not be float[%d]", maskDataType, ge::DT_FLOAT),
                        return ge::GRAPH_FAILED);
        // 当fp16高精度模式时，mask类型只支持bool或int8
        OPS_ERR_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) &&
                        (maskDataType != ge::DT_BOOL) && (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "maskType[%d] should be bool, int8 or uint8 when precision mode", maskDataType),
                        return ge::GRAPH_FAILED);
        // 当bf16时，mask类型只支持bool或int8
        OPS_ERR_IF((inputType == ge::DT_BF16) &&
                        (maskDataType != ge::DT_BOOL) && (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "maskType[%d] should be bool, int8 or uint8 when input type is bfloat16", maskDataType),
                        return ge::GRAPH_FAILED);
        // fp16 mask type 不支持行无效修正
        OPS_ERR_IF((maskDataType == ge::DT_FLOAT16 && tilingData.promptAttentionBaseParams.get_isRowInvalid()),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "maskType[%d] should not be float16[%d] when innerPrecise = 2 or 3", maskDataType, ge::DT_FLOAT16),
                        return ge::GRAPH_FAILED);
        if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
            OPS_ERR_IF(maskDataType != ge::DT_BOOL, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "maskType[%d] should be bool when socVersion is 310p", maskDataType), return ge::GRAPH_FAILED);
        }
    }
    typeByteNum = BYTE_BLOCK / dataTypeSize;
    outputTypeByteNum = BYTE_BLOCK / outputDataTypeSize;
    softmaxTypeByteNum = BYTE_BLOCK / softmaxDataTypeSize;
    maskTypeByteNum = BYTE_BLOCK / maskElemSize;

    tilingData.promptAttentionBaseParams.set_maskTypeByteNum(maskTypeByteNum);
    tilingData.promptAttentionBaseParams.set_softmaxTypeByteNum(softmaxTypeByteNum);
    tilingData.promptAttentionBaseParams.set_outputTypeByteNum(outputTypeByteNum);
    tilingData.promptAttentionBaseParams.set_typeByteNum(typeByteNum);
    // get shape
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    const gert::StorageShape* quantOffset2Shape = contextKeyParams.offset2Shape;
    const gert::StorageShape* antiquantScaleShape = contextKeyParams.antiquantScaleShape;
    const gert::StorageShape* antiquantOffsetShape = contextKeyParams.antiquantOffsetShape;
    const gert::StorageShape* outShape = contextKeyParams.outputShape;
    const gert::StorageShape* SoftmaxLseOutShape = contextKeyParams.lseoutputShape;

    uint32_t deqScaleTypeFlag = (contextKeyParams.deqScaleType == DT_UINT64) ? 0U : 1U;
    uint32_t deqScale2TypeFlag = (contextKeyParams.deqScale2Type == DT_UINT64) ? 0U : 1U;

    tilingData.promptAttentionBaseParams.set_deqScaleFlag(deqScaleTypeFlag);
    tilingData.promptAttentionBaseParams.set_deqScale2Flag(deqScale2TypeFlag);

    OPS_ERR_IF(((contextKeyParams.inputDataType == ge::DT_INT8) && (contextKeyParams.outputDataType == ge::DT_FLOAT16) && ((contextKeyParams.scale2Shape != nullptr) || (contextKeyParams.offset2Shape != nullptr))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When query dtype is int8 and output dtype is fp16, quantScale2 and quantOffset2 should be null."),
                    return ge::GRAPH_FAILED);

    // prefix
    isKVHasPrefix = contextKeyParams.keySharedPrefix != nullptr && contextKeyParams.valueSharedPrefix != nullptr ? true : false;
    OPS_ERR_IF((!isKVHasPrefix && (contextKeyParams.keySharedPrefix != nullptr || contextKeyParams.valueSharedPrefix != nullptr)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when system prefix is used, key_shared_prefix and value_shared_prefix are required!"),
                return ge::GRAPH_FAILED);
    if (isKVHasPrefix) {
        // prefix不支持tensorlist、PA、左padding
        OPS_ERR_IF((contextKeyParams.isKvContinuous == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when tensorlist is used, system prefix is not supported!"),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF((enablePA),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when system prefix is used, page attention is not supported!"),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF(((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when system prefix is used, leftpadding is not supported!"),
                    return ge::GRAPH_FAILED);

        uint32_t prefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDimNum();
        uint32_t prefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDimNum();
        OPS_ERR_IF(((prefixKeyDim != keyShape->GetStorageShape().GetDimNum()) || (prefixKeyDim != prefixValueDim)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "prefix dim num is invalid, key_shared_prefix dim:%u, value_shared_prefix dim:%u!", prefixKeyDim, prefixValueDim),
                    return ge::GRAPH_FAILED);
        for (uint32_t i = 0; i < prefixKeyDim; i++) {
            uint32_t tmpPrefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(i);
            uint32_t tmpPrefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDim(i);
            OPS_ERR_IF(((tmpPrefixKeyDim != tmpPrefixValueDim) || (tmpPrefixKeyDim == 0)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "invalid prefix dim, key_shared_prefix[%u]:%u, value_shared_prefix[%u]:%u!", i, tmpPrefixKeyDim, i, tmpPrefixValueDim),
                        return ge::GRAPH_FAILED);
        }
    }

    // set mask last dim size
    auto maskKVsSize = 2048; // 2048 : default last frist dim
    auto maskQsSize = 2048; // 2048 : default last second dim
    if (attenMaskShape != nullptr) {
        maskKVsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() - 1); // 1: last frist dim
        maskQsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() - 2); // 2: last second dim
    }

    tilingData.promptAttentionBaseParams.set_maskKVsSize(maskKVsSize);
    tilingData.promptAttentionBaseParams.set_maskQsSize(maskQsSize);

    // 内部log打印，此处无需打印，下同
    if (CheckNonEmptyShapeExceptions(contextKeyParams, queryShape, "query")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, keyShape, "key")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, valueShape, "value")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, outShape, "out")) {
        return ge::GRAPH_FAILED;
    }
    // 可选输入可以为空
    OPS_ERR_IF((pseShiftShape != nullptr) &&
                    (pseShiftShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Shape size of pseShift is overflow."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((attenMaskShape != nullptr) &&
                    (attenMaskShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Shape size of attenMask is overflow."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((outShape->GetStorageShape().GetShapeSize() != 0) &&
                    (queryShape->GetStorageShape().GetShapeSize() == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "query is empty tensor."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((queryShape->GetStorageShape().GetDimNum() < NUM_2) || (queryShape->GetStorageShape().GetDimNum() > 4), // 2:min dimnum,4:max
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryShape dim num is error, queryShape dim num = %lu", queryShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);

    OPS_ERR_IF(SetInputLayout(contextKeyParams.layout) == GRAPH_FAILED,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "invalid input layout:%s.", contextKeyParams.layout),
                return ge::GRAPH_FAILED);

    // 入图场景可能有out为空tensor的情况出现，这里out为空对size 0处理，相当于什么都没做直接返回
    if ((keyShape->GetStorageShape().GetShapeSize() == 0) || (valueShape->GetStorageShape().GetShapeSize() == 0) ||
        (outShape->GetStorageShape().GetShapeSize() == 0) || (contextKeyParams.emptyTensor == 1)) {
        tilingKey = EMPTY_KV_TILING_KEY;
        OPS_ERR_IF(GetAndCheckEmptyQueryShape(contextKeyParams, queryShape) == ge::GRAPH_FAILED,
                   OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "GetAndCheckEmptyQueryShape failed."),
                   return ge::GRAPH_FAILED);
        PromptFlashAttentionInitOutputSplit(outShape->GetStorageShape().GetShapeSize(), tilingData, coreNum);
        tilingData.promptAttentionInitOutputParams.set_needInit(1);

        blockDimToBeSet = ascendcPlatform.CalcTschBlockDim(coreNum, aicNum, coreNum);

        size_t* workspace = contextKeyParams.workspaceSize;
        const size_t sysWorkspaceSize = 16 * 1024 * 1024;  // workspace至少需要这么多
        workspace[0] = sysWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }
    tilingData.promptAttentionBaseParams.set_useMask(1);
    if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0))
        || (attenMaskShape == nullptr)) {
        tilingData.promptAttentionBaseParams.set_useMask(0);
    }

    if (inputType == ge::DT_INT8) {
        OPS_ERR_IF((deqScale1Shape == nullptr) || (quantScale1Shape == nullptr) || (deqScale2Shape == nullptr),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "dequant scale or fisrt quant scale is nullptr when input type is int8."),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF((deqScale1Shape != nullptr && deqScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
                        (quantScale1Shape != nullptr && quantScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
                        (deqScale2Shape != nullptr && deqScale2Shape->GetStorageShape().GetShapeSize() == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "dequant scale or fisrt quant scale is empty tensor when input type is int8."),
                    return ge::GRAPH_FAILED);
    }

    const int32_t* n = contextKeyParams.headsNumber; // q的num_heads
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    const int32_t* nextTokens = contextKeyParams.nextToken;
    const int32_t* preTokens = contextKeyParams.preToken;
    const float* scaleValue = contextKeyParams.scaleValue;
    const int32_t* blockSize = contextKeyParams.blockSize;

    int32_t sparsePreTokens;
    int32_t sparseNextTokens;
    int32_t sparseModeVal = 0;
    // KV一致性检查
    OPS_ERR_IF(CheckKeyValueParamsConsistency(contextKeyParams) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "key value consistency check failed!"),
                    return ge::GRAPH_FAILED);

    const int32_t* numKeyValueHeads = contextKeyParams.numKeyValueHeads;
    if (!SetTilingHeadNumRatio(contextKeyParams, n, numKeyValueHeads, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // get dims
    uint32_t seqInnerSize = 0U; // kv s
    uint32_t h = 0U;
    uint32_t s = 0U;
    uint32_t b = 0U;
    uint32_t bKV = 0U;
    uint32_t prefixSeqInnerSize = 0;
    uint32_t bPreifx = 0U;
    uint32_t nPreifx = 0U;
    uint32_t hPreifx = 0U;
    uint32_t dPreifx = 0U;

    const gert::Tensor* tempData = contextKeyParams.actualSeqenceLengthQ;
    const gert::Tensor* tempDataKV = contextKeyParams.actualSeqenceLengthKV;
    size_t actualLenDims = (tempData != nullptr) ? tempData->GetShapeSize() : 0;
    size_t actualLenDimsKV = (tempDataKV != nullptr) ? tempDataKV->GetShapeSize() : 0;
    uint32_t isActualSeqLengthsNull = contextKeyParams.fromTilingSink == 0 ? (actualLenDims == 0 || tempData == nullptr || tempData->GetData<int64_t>() == nullptr) : 1;
    uint32_t isActualSeqLengthsKVNull = contextKeyParams.fromTilingSink == 0 ? (actualLenDimsKV == 0 || tempDataKV == nullptr || tempDataKV->GetData<int64_t>() == nullptr) : 1;
    OPS_ERR_IF(enablePA && (isActualSeqLengthsKVNull == 1) && (contextKeyParams.fromTilingSink == 0),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "actual seq length kv can't be null when blockTable is not null"),
                        return ge::GRAPH_FAILED);
    if (inputLayout == (InputLayout::SH) && (actualLenDimsKV != 0)) {
        OPS_LOG_W(contextKeyParams.opName, "actual_seq_lengths_kv is useless for SH format!");
    }
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        unsigned int ret;
        ret = GetBasicShape310P(b, bKV, s, h, seqInnerSize, queryShape, keyShape, *n, actualLenDims, actualLenDimsKV);
        OPS_ERR_IF(ret == GRAPH_FAILED,
                        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "execute is failed."),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF((s > 65536) || (seqInnerSize > 65536),
                        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "310P not support Qs or KVs lager than 65536,Qs = %u, Kvs = %u", s, seqInnerSize),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF((tilingData.promptAttentionBaseParams.get_useMask()!= 0 && (s % 16 != 0 || seqInnerSize % 16 != 0 || s != seqInnerSize)),
                        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "attention mask must be NULL，when Qs,Kvs is unAlign or Qs is not equal to Kvs, Qs = %u, Kvs = %u", s, seqInnerSize),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF(((*preTokens < static_cast<int32_t>(s)) || (*nextTokens < static_cast<int32_t>(seqInnerSize) && *nextTokens != 0)),
                        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "pretokens should lager than Qs, nexttokens should be 0 or larger than Kvs, Qs = %u, Kvs = %u, preTokens = %d, nextTokens = %d", s, seqInnerSize, *preTokens, *nextTokens),
                        return ge::GRAPH_FAILED);
    } else {
        if (inputLayout == InputLayout::BNSD || inputLayout == InputLayout::NSD) {
            if (queryShape->GetStorageShape().GetDimNum() == 3) { // dim num: 3
                b = 1;
                bKV = 1;
                s = queryShape->GetStorageShape().GetDim(1);
                seqInnerSize = keyShape->GetStorageShape().GetDim(1);
                h = (*n) * queryShape->GetStorageShape().GetDim(2); // dim num: 2
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1) : 0;
            } else {
                b = queryShape->GetStorageShape().GetDim(0);
                bKV = keyShape->GetStorageShape().GetDim(0);
                s = queryShape->GetStorageShape().GetDim(2); // dim num: 2
                seqInnerSize = keyShape->GetStorageShape().GetDim(2); // dim num: 2
                h = queryShape->GetStorageShape().GetDim(1) * queryShape->GetStorageShape().GetDim(3);  // dim num: 3
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_2) : 0;
                bPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0) : 0;
                nPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1) : 0;
                dPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_3) : 0;
            }
        } else if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) ||
            (inputLayout == InputLayout::SH)) {
            if (queryShape->GetStorageShape().GetDimNum() == NUM_2) { // dim num: 2
                b = actualLenDims == 0 ? 1 : actualLenDims; // SH格式未输入actual_seq时，按照batch=1处理
                bKV = actualLenDimsKV == 0 ? 1 : actualLenDimsKV; // SH格式未输入actual_seqkv时，按照batch=1处理
                s = queryShape->GetStorageShape().GetDim(0);
                h = queryShape->GetStorageShape().GetDim(1);
                seqInnerSize = keyShape->GetStorageShape().GetDim(0);
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0) : 0;
            } else if (queryShape->GetStorageShape().GetDimNum() == 3) { // 3 : BSH
                b = queryShape->GetStorageShape().GetDim(0);
                bKV = keyShape->GetStorageShape().GetDim(0);
                s = queryShape->GetStorageShape().GetDim(1);
                h = queryShape->GetStorageShape().GetDim(2); // dim num: 2
                seqInnerSize = keyShape->GetStorageShape().GetDim(1);
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1) : 0;
                bPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0) : 0;
                hPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_2) : 0;
            } else { // BSND
                b = queryShape->GetStorageShape().GetDim(0);
                bKV = keyShape->GetStorageShape().GetDim(0);
                s = queryShape->GetStorageShape().GetDim(1);
                h = queryShape->GetStorageShape().GetDim(INDEX_2) *
                    queryShape->GetStorageShape().GetDim(INDEX_3);
                seqInnerSize = keyShape->GetStorageShape().GetDim(1);
                prefixSeqInnerSize = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(1) : 0;
                bPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(0) : 0;
                nPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_2) : 0;
                dPreifx = isKVHasPrefix ? contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(INDEX_3) : 0;
            }
        } else {
            return ge::GRAPH_FAILED;
        }
        if (contextKeyParams.isKvContinuous == 0) {
            seqInnerSize = contextKeyParams.maxKVs;
        }
    }

    uint32_t actualSharedPrefixLen = 0U;
    if (isKVHasPrefix && contextKeyParams.actualSharedPrefixLen != nullptr) {
        OPS_ERR_IF(((contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetShapeSize() != 1) ||
                    (contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetDimNum() != 1)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "actualSharedPrefixLen input is invalid!"),
                    return ge::GRAPH_FAILED);
        actualSharedPrefixLen = static_cast<uint32_t>(contextKeyParams.actualSharedPrefixLen->GetData<int64_t>()[0]);
        OPS_ERR_IF((actualSharedPrefixLen > prefixSeqInnerSize),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "actualSharedPrefixLen[%u] should be smaller than prefixSeqInnerSize[%u]!", actualSharedPrefixLen, prefixSeqInnerSize),
                    return ge::GRAPH_FAILED);
        tilingData.promptAttentionBaseParams.set_isActualSharedPrefixLenNull(0);
    } else {
        tilingData.promptAttentionBaseParams.set_isActualSharedPrefixLenNull(1);
        actualSharedPrefixLen = prefixSeqInnerSize;
    }
    tilingData.promptAttentionBaseParams.set_prefixSeqInnerSize(prefixSeqInnerSize);

    if (isKVHasPrefix) {
        OPS_ERR_IF((bPreifx != 1),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "prefix batch %u only support 1!", bPreifx),
                    return ge::GRAPH_FAILED);
        if (inputLayout == InputLayout::BSH) {
            OPS_ERR_IF((hPreifx != h / tilingData.promptAttentionBaseParams.get_headNumRatio()),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "prefix H:%u shoueld be same wath KV!", hPreifx),
                        return ge::GRAPH_FAILED);
        } else {
            OPS_ERR_IF((nPreifx != (*n) / tilingData.promptAttentionBaseParams.get_headNumRatio()) || (dPreifx != h / (*n)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "prefix N:%u and D:%u shoueld be same with KV!", nPreifx, dPreifx),
                        return ge::GRAPH_FAILED);
        }
    }

    if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::SH)) && (h > 65535)) {  // 搬入stride不能超过65535
        OPS_LOG_W(contextKeyParams.opName, "h(%u) is larger than 65535, which may cause precision problem! Please use BNSD or BNSD_BSND instead.", h);
    }

    // PA场景没有B轴，不校验
    OPS_ERR_IF((b != bKV) && (contextKeyParams.isKvContinuous == 1) && (!enablePA) && (contextKeyParams.fromTilingSink == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "query batch must be equal to key/value batch, query batch = %u , key/value batch = %u .", b, bKV),
                    return ge::GRAPH_FAILED);

    OPS_ERR_IF((b > BLIMIT || (b > 128 && (inputLayout == InputLayout::SH))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "batch size = %u, more than 65536 or more than 128(when layout is SH) or less than 0, is error.", b),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF((curShortSocName == platform_ascendc::SocVersion::ASCEND310P && b > 128U),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "ascend310p platform do not support batch size(%u) more than 128.", b),
                    return ge::GRAPH_FAILED);

    bool iskvdiff = (seqInnerSize != s);
    OPS_ERR_IF((iskvdiff) && (inputLayout == InputLayout::SH) && (!enablePA) && (contextKeyParams.fromTilingSink == 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "SH format not support q kv diff, length of q = %u , length of kv = %u.", s, seqInnerSize),
                    return ge::GRAPH_FAILED);

    // actualLenDims & actualLenDimsKV 维度和长度检查
    if (!CheckActualSeqLength(contextKeyParams, b, s, seqInnerSize, tempData, tempDataKV, inputLayout)) {
        return ge::GRAPH_FAILED;
    }

    if (enablePA) {  // PA场景对mask的shape进行校验时，以maxBlockNumPerBatch * tempBlockSize作为tmpS2
        if (!CheckPATypeAndShape(contextKeyParams, tempDataKV, (int32_t)b, (int32_t)(*n), (int32_t)h, (int32_t)tilingData.promptAttentionBaseParams.get_headNumRatio())) {
            return ge::GRAPH_FAILED;
        }
    } else {
        tmpS2 = seqInnerSize;
    }
    tilingData.promptAttentionBaseParams.set_PAlayoutType(PAlayoutType);

    if (!CheckAttenMaskShape(contextKeyParams, sparseMode, attenMaskShape, s, tmpS2 + actualSharedPrefixLen, b)) {
        return ge::GRAPH_FAILED;
    }
    // 防护pse的数据类型和shape
    if (contextKeyParams.pseShift != nullptr) {
        usePseShift = 1;
        if (!CheckPseShiftTypeAndShape(contextKeyParams, pseShiftShape, b, *n, s, tmpS2)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        usePseShift = 0;
    }

    // sparse check
    int32_t sparseRet = 0;
    if (sparseMode != nullptr) {
        sparseRet = (*sparseMode != SPARSE_MODE_NO_MASK && *sparseMode != SPARSE_MODE_LEFT_UP &&
                     *sparseMode != SPARSE_MODE_RIGHT_DOWN && *sparseMode != SPARSE_MODE_ALL_MASK && *sparseMode != SPARSE_MODE_BAND);
        OPS_ERR_IF((sparseRet == 1),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "sparse_mode = %d is out of range.", *sparseMode),
                    return ge::GRAPH_FAILED);

        if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0))
            || (attenMaskShape == nullptr)) {
            tilingData.promptAttentionBaseParams.set_useMask(0); // for sparse check rule 5
        }
    }
    sparsePreTokens = *preTokens;
    sparseNextTokens = *nextTokens;

    uint32_t attenMaskBatch = 1U;
    bool isBandMode = false;
    bool isDefaultMode = (sparseMode == nullptr) || ((sparseMode != nullptr) && *sparseMode == SPARSE_MODE_NO_MASK);
    if (attenMaskShape != nullptr) {
        uint32_t attenMaskDim = attenMaskShape->GetStorageShape().GetDimNum();
        if (attenMaskDim != NUM_2) { // 2: target dimension of attenMask
            attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        }
        if (sparseMode != nullptr) {
            if (*sparseMode == SPARSE_MODE_LEFT_UP) {
                sparsePreTokens = SPARSE_MODE_INT_MAX;
                sparseNextTokens = 0;
                sparseModeVal = *sparseMode;
            } else if (*sparseMode == SPARSE_MODE_RIGHT_DOWN) { // right down 的tokens计算在kernel侧
                sparsePreTokens = SPARSE_MODE_INT_MAX;
                sparseModeVal = *sparseMode;
            } else if (*sparseMode == SPARSE_MODE_ALL_MASK) {
                sparsePreTokens = SPARSE_MODE_INT_MAX;
                sparseNextTokens = SPARSE_MODE_INT_MAX;
                sparseModeVal = *sparseMode;
            } else if (*sparseMode == SPARSE_MODE_BAND) {
                sparseModeVal = *sparseMode;
                isBandMode = true;
                OPS_ERR_IF(*preTokens < 0 && *nextTokens < 0,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "preTokens and nextTokens must not be negative number in band mode, preTokens = %d , nextTokens = %d .", *preTokens, *nextTokens),
		            return ge::GRAPH_FAILED);
            }
            OPS_LOG_I(contextKeyParams.opName, "sparseMode is %d", *sparseMode);
        }
    }
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN ||
        *sparseMode == SPARSE_MODE_ALL_MASK || *sparseMode == SPARSE_MODE_BAND)) {
        sparseRet = (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0))
                    || (attenMaskShape == nullptr));

        OPS_ERR_IF((sparseRet == 1),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "attenMask should not be null when sparse_mode is %d.", *sparseMode),
                    return ge::GRAPH_FAILED);

        auto maskDataType = contextKeyParams.maskDataType;
        // 当sparse = 2、3、4时，mask类型只支持bool、int8、uint8
        OPS_ERR_IF((*sparseMode != SPARSE_MODE_ALL_MASK) && (maskDataType != ge::DT_BOOL) &&
                        (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "maskType[%d] should be bool, int8 or uint8 when sparse mode is %d.", maskDataType, *sparseMode),
                        return ge::GRAPH_FAILED);
    }
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK)) {
        // sparse mode，需要对 attention mask 空tensor 的2种场景做相同处理
        if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0))
            || (attenMaskShape == nullptr)) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
            sparseModeVal = *sparseMode;
        }
    }

    if (isDefaultMode && ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr))) {
        // sparse mode = 0 且有左padding的场景，对 attention mask 部分全算
        sparsePreTokens = SPARSE_MODE_INT_MAX;
        sparseNextTokens = SPARSE_MODE_INT_MAX;
    }
    OPS_ERR_IF((sparsePreTokens < 0) && (sparseNextTokens < 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "preTokens and nextokens cannot neither be negative number, preTokens = %d, nextTokens = %d.", sparsePreTokens, sparseNextTokens),
		            return ge::GRAPH_FAILED);

    OPS_ERR_IF((sparseNextTokens * (-1)) > sparsePreTokens,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "nexttoken line should be higher than pretoken line."),
		            return ge::GRAPH_FAILED);

    OPS_ERR_IF(isDefaultMode && (sparseNextTokens < 0) && (sparseNextTokens * (-1)) >= (int32_t)s,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "nextTokens absolute value should be smaller than length of q, nextTokens = %d, length of q = %u.", sparseNextTokens, s),
                    return ge::GRAPH_FAILED);

    OPS_ERR_IF(isDefaultMode && (sparsePreTokens < 0) && (sparsePreTokens * (-1) >= ((int32_t)tmpS2 + (int32_t)actualSharedPrefixLen)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "preToken absolute value should be smaller than length of k and v (length of k and v + length of prefix when enable prefix)."),
                    return ge::GRAPH_FAILED);

    size_t lenDims = b; // 当前actual_seq_length数组长度等于b
    uint32_t isLayoutSH = (inputLayout == InputLayout::SH) ? 1U : 0U;

    std::vector<int64_t> actualSeqLengths(lenDims);
    int64_t middleActualSeqLengths = 0;
    std::vector<int64_t> actualSeqLengthsKV(lenDims);

    OPS_ERR_IF(((*n <= 0) || (*n > static_cast<int32_t>(h))),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "num heads is error."),
                    return ge::GRAPH_FAILED);
    uint32_t needInit = 0U;
    int64_t preTokensPerbatch = 0;
    int64_t nextTokensPerbatch = 0;
    bool checkQuantValue = (outputType == ge::DT_INT8) &&
                           (quantOffset2Shape != nullptr) &&
                           (quantOffset2Shape->GetStorageShape().GetShapeSize() != 0);

    OPS_ERR_IF((outputType == ge::DT_INT8 && isBandMode && ((sparsePreTokens < 0) || sparseNextTokens < 0)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "When output type is int8, sparse mode = 4, preTokens (%d) or nextTokens (%d) cannot be negative.",  sparsePreTokens, sparseNextTokens),
                        return ge::GRAPH_FAILED);
    if (contextKeyParams.fromTilingSink == 0) {
        for (size_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
            if ((actualLenDims == 0) || (tempData == nullptr) || (tempData->GetData<int64_t>() == nullptr)) {
                actualSeqLengths[i] = s;
                middleActualSeqLengths += actualSeqLengths[i];
            } else {
                actualSeqLengths[i] = static_cast<uint32_t>(tempData->GetData<int64_t>()[i]);
                if (actualSeqLengths[i] != s) {
                    needInit = 1;
                    OPS_ERR_IF(isDefaultMode && sparseNextTokens < 0 && sparseNextTokens * (-1) >= (int32_t)actualSeqLengths[i],
                                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                                    "nexttoken absolute value should be smaller than actual length of q."),
                                    return ge::GRAPH_FAILED);
                }
                middleActualSeqLengths += actualSeqLengths[i];
            }
            if ((actualLenDimsKV == 0) || (tempDataKV == nullptr) || (tempDataKV->GetData<int64_t>() == nullptr)) {       // 用户没输入act_seq_kv
                if (contextKeyParams.isKvContinuous == 1){
                    actualSeqLengthsKV[i] = tmpS2;
                } else {
                    if ((inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::BSH)) {
                        actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1);
                    } else {
                        actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2);
                    }
                }
            } else {
                actualSeqLengthsKV[i] = static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[i]);
                if (actualSeqLengthsKV[i] != tmpS2) {
                    needInit = 1;
                    OPS_ERR_IF(isDefaultMode && sparsePreTokens < 0 && \
                        (sparsePreTokens * (-1) >= ((int32_t)actualSeqLengthsKV[i] + (int32_t)actualSharedPrefixLen)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "preToken absolute value should be smaller than actual length of k and v "
                        "(actual length of k and v + length of prefix when enable prefix), preToken = %d, actual length of k and v = %ld, actual prefix len = %u.",
                        sparsePreTokens, actualSeqLengthsKV[i], actualSharedPrefixLen),
                        return ge::GRAPH_FAILED);
                }
            }
            if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
                preTokensPerbatch = SPARSE_MODE_INT_MAX;
                nextTokensPerbatch = actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen - actualSeqLengths[i];
            } else if (sparseModeVal == SPARSE_MODE_BAND) {
                preTokensPerbatch = sparsePreTokens - actualSeqLengthsKV[i] - (int64_t)actualSharedPrefixLen + actualSeqLengths[i];
                nextTokensPerbatch = sparseNextTokens + actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen - actualSeqLengths[i];
            } else {
                preTokensPerbatch = sparsePreTokens;
                nextTokensPerbatch = sparseNextTokens;
            }
            if ((nextTokensPerbatch < 0) ||
                (actualSeqLengths[i] > (actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + preTokensPerbatch))) {
                needInit = 1;
            }
            // if preTokensPerbatch + actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i] < 0 or nextTokensPerbatch < 0,
            // the last few lines or the first few lines of the QKt matrix are not computed.
            OPS_ERR_IF((checkQuantValue && \
                ((preTokensPerbatch + actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen - actualSeqLengths[i] < 0) || (nextTokensPerbatch < 0))),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When sparse mode = %d, output dtype is int8, quantOffset2 is not null or empty tensor, "
                "preTokens = %d and nextTokens = %d, some rows of the matrix do not participate in the calculation, "
                "the accuracy of the final result will be incorrect. Please see the documentation for more details.",
                sparseModeVal, *preTokens, *nextTokens),
                return ge::GRAPH_FAILED);
            OPS_LOG_I(contextKeyParams.opName, "preTokensPerbatch[%d] is %d, nextTokensPerbatch[%d] is %d",
                    i, preTokensPerbatch, i, nextTokensPerbatch);
            if (!isBandMode && actualSeqLengths[i] > actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + (int64_t)sparsePreTokens) {
                actualSeqLengths[i] = actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + (int64_t)sparsePreTokens;
            }

            OPS_ERR_IF((isBandMode && (sparseNextTokens < 0) && (sparseNextTokens * (-1) >= actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen)),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "nextTokens absolute value should be smaller than actual length of k and v in band mode (actual length of k and v + length of "
                        "prefix when enable prefix), nextTokens = %d, actual length of k and v = %ld, prefix length = %u",
                        sparseNextTokens, actualSeqLengthsKV[i], actualSharedPrefixLen),
                        return ge::GRAPH_FAILED);

            OPS_ERR_IF((isBandMode && (sparsePreTokens < 0) && (sparsePreTokens * (-1) >= actualSeqLengths[i])),
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                            "preTokens absolute value should be smaller than actual length of q in band mode, preTokens = %d, actual length of q = %ld", sparsePreTokens, actualSeqLengths[i]),
                            return ge::GRAPH_FAILED);

            if(isBandMode && actualSeqLengths[i] > actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + preTokensPerbatch){
                actualSeqLengths[i] = actualSeqLengthsKV[i] + (int64_t)actualSharedPrefixLen + preTokensPerbatch;
            }

            OPS_LOG_I(contextKeyParams.opName, "actualSeqLengths[%d] is %ld, actualSeqLengthsKV[%d] is %ld, actualSharedPrefixLen is %u, needInit is %u",
                    i, actualSeqLengths[i], i, actualSeqLengthsKV[i], actualSharedPrefixLen, needInit);
        }
    }

    uint32_t hDivN = h / *n; // d = h / n
    // 拦截高精度模式目前不支持shape
    const uint32_t precisionBlockEleCut = BYTE_BLOCK / FLOAT16SIZE; // 高精度目前只支持FP16，按32/2=16对齐
    OPS_ERR_IF((hDivN > DLIMIT),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "d = %u after it is aligned with 32B, but d should <= 512. When layout is BNSD,"
                    "d is query shape in dim 3, and layout is BSH, d = h / n", hDivN),
                    return ge::GRAPH_FAILED); // 高精度和高性能的d均不能大于512
    if ((s > SLIMIT) || (tmpS2 > SLIMIT)) {
        OPS_LOG_W(contextKeyParams.opName,
                   "seq should <= 20M, qs = %u, kvs = %u", s, tmpS2);
    }
    OPS_ERR_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION) &&
                    (inputLayout == InputLayout::SH)),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "do not support SH input format when high precision!"),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION) &&
                    (hDivN % precisionBlockEleCut) != 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "d should be align when high precision, d = %u", hDivN),
                    return ge::GRAPH_FAILED); // d到这里会被pad，无法获取原始值，故不打印
    if ((inputType == ge::DT_FLOAT16) && (outputType == ge::DT_INT8)) {
        OPS_ERR_IF((inputLayout == InputLayout::SH),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "When input dtype is fp16 and output dtype is int8, SH layout is not supported."),
                        return ge::GRAPH_FAILED);
        OPS_ERR_IF((deqScale1Shape != nullptr) || (quantScale1Shape != nullptr) || (deqScale2Shape != nullptr),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "When input dtype is fp16 and output dtype is int8, PFA inputs "
                        "dequantScale1, quantScale1 and dequantScale2 should be null."),
                        return ge::GRAPH_FAILED);
    }

    // 后quant参数检查
    OPS_ERR_IF(CheckPostQuantParams(contextKeyParams, h, *n) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "post quant params check failed!"),
                    return ge::GRAPH_FAILED);

    // Perchannel判定待适配，先保持现有逻辑
    tilingData.promptAttentionBaseParams.set_isQuant2Perchannel(0);
    tilingData.promptAttentionBaseParams.set_isQuant2BF16(0);
    if (outputType == ge::DT_INT8) {
        if (quantScale2Shape->GetStorageShape().GetShapeSize() > 1) {
            tilingData.promptAttentionBaseParams.set_isQuant2Perchannel(1);
        }
        if (contextKeyParams.quantScale2Type == ge::DT_BF16) {
            tilingData.promptAttentionBaseParams.set_isQuant2BF16(1);
        }
    }

    if ((curShortSocName == platform_ascendc::SocVersion::ASCEND310P )&& softmaxDataTypeNZ_ == FLOAT16SIZE) {
        sparseModeVal = 99; // 310p临时用sparse字段表示是否走近似计算方案
    }
    tilingData.promptAttentionBaseParams.set_dimNumOfseq(lenDims);
    tilingData.promptAttentionBaseParams.set_scaleValue(*scaleValue);
    tilingData.promptAttentionBaseParams.set_headSize(hDivN);
    if (enablePA) {
        tilingData.promptAttentionBaseParams.set_blockSize(*blockSize);
    } else {
        tilingData.promptAttentionBaseParams.set_blockSize(BLOCK_SIZE_BASE);
    }
    tilingData.promptAttentionBaseParams.set_blockTableDim2(blockTableDim2);
    tilingData.promptAttentionBaseParams.set_PABlockNumSum(PABlockNumSum);
    tilingData.promptAttentionBaseParams.set_seqInnerSize(tmpS2);
    tilingData.promptAttentionBaseParams.set_seqSize(s);
    tilingData.promptAttentionBaseParams.set_headNumSize(*n);
    tilingData.promptAttentionBaseParams.set_batchSize(lenDims);

    tilingData.promptAttentionBaseParams.set_preTokens(static_cast<uint32_t>(sparsePreTokens));
    tilingData.promptAttentionBaseParams.set_nextTokens(static_cast<uint32_t>(sparseNextTokens));
    tilingData.promptAttentionBaseParams.set_sparseMode(static_cast<uint32_t>(sparseModeVal));
    tilingData.promptAttentionBaseParams.set_isLayoutSH(isLayoutSH);
    tilingData.promptAttentionBaseParams.set_isActualSeqLengthsNull(isActualSeqLengthsNull);
    tilingData.promptAttentionBaseParams.set_isActualSeqLengthsKVNull(isActualSeqLengthsKVNull);
    tilingData.promptAttentionSingleCoreParams.set_attenMaskBatch(attenMaskBatch);
    tilingData.promptAttentionInitOutputParams.set_needInit(needInit);

    uint32_t originHeadSize = tilingData.promptAttentionBaseParams.get_headSize();
    uint32_t blockElementCnt = BYTE_BLOCK / dataTypeSize;
    if (originHeadSize % blockElementCnt != 0) { // 判断D是否为32B对齐，以fp16类型，元素个数为16
        tilingData.promptAttentionBaseParams.set_alignedHeadSize(((
            originHeadSize + blockElementCnt - 1) / blockElementCnt) * blockElementCnt);
        isDNoTail = false;
    } else {
        tilingData.promptAttentionBaseParams.set_alignedHeadSize(originHeadSize);
    }

    // 检查kv antiquant参数 - scale和offset的shape
    uint32_t nKV = *n / tilingData.promptAttentionBaseParams.get_headNumRatio();
    uint32_t hKV = h / tilingData.promptAttentionBaseParams.get_headNumRatio();
    if (enableKvAntiquant && !CheckAntiquantParamsShape(contextKeyParams, antiquantScaleShape, antiquantOffsetShape, nKV, hDivN, hKV, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // 判断是否进 new tiling
    bool useNewTiling = true;
    bool useBalanceTiling = true;
    bool noInputActualSeqKV = contextKeyParams.fromTilingSink == 0 ? ((actualLenDimsKV == 0) || (tempDataKV == nullptr) || (tempDataKV->GetData<int64_t>() == nullptr)) : true;
    if ((inputLayout != InputLayout::BNSD) && (inputLayout != InputLayout::NSD)
        && (tilingData.promptAttentionBaseParams.get_headNumRatio() == 1)
        && (lenDims == 1)
        && (!iskvdiff)
        && ((*n % coreNum == 0) && (tmpS2 < CVDIFF_S2_THRESHOLDS))
        && noInputActualSeqKV) {
        useNewTiling = false;
    }
    if (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || (enablePA)) {
        useNewTiling = true; // 高精度模式不走老模板
    }

    // 目前只针对bs=1的场景，待优化
    if ((needInit == 1) || (lenDims != 1)) {
        useBalanceTiling = false;
    }
    if (tilingData.promptAttentionBaseParams.get_headNumRatio() != 1) {
        useBalanceTiling = false;
    }
    OPS_LOG_I(contextKeyParams.opName,
              "Tiling Info: b is %u, bKV is %d, n is %d, numKeyValueHeads is %d, s1 is %u, s2 is %u, h is %u, d is %u, headNumRatio = %u",
              b, bKV, *n, *numKeyValueHeads, s, tmpS2, h, hDivN, tilingData.promptAttentionBaseParams.get_headNumRatio());
    OPS_LOG_I(contextKeyParams.opName,
              "inputLayout is %d, innerPrecise is %lu, "
              "scaleValue is %f, preTokens is %d, nextTokens is %d",
              inputLayout, innerPrecise, *scaleValue, *preTokens, *nextTokens);
    // 推理出tiling模式 是否D轴切分，是否S2全载，是否CV分离，是否使用matmul norm模板
    InferTilingMod(contextKeyParams, actualSeqLengths, actualSeqLengthsKV, lenDims, hDivN, tmpS2, sparseModeVal);

    uint32_t sOuterFactor;
    uint32_t sInnerFactor;
    uint32_t softmaxSInnerFactor;
    uint32_t softmaxSOuterFactor;

    // 当前不会出现D拆分场景, 分块时默认splitD=0
    if (tilingMod == TilingMod::CVSAME) {
        OPS_ERR_IF(lenDims > 128,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "when D axis size(%u) is unaligend with 32 bytes, batch size(%zu) can not larger than 128.", hDivN, lenDims),
                        return ge::GRAPH_FAILED);
        auto ret = AdjustCVTiling(hDivN, *n, middleActualSeqLengths, ubSize, l1Size, l0CSize, maskElemSize,
                                    sOuterFactor, sInnerFactor, tilingData);
        OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "adjust tiling fail"),
                        return ret);
        softmaxSOuterFactor = sOuterFactor;
        softmaxSInnerFactor = sInnerFactor;
    } else {
        auto ret = AdjustCVTilingCVDiff(ubSize, l1Size, l0CSize, maskElemSize, sOuterFactor,
                                        sInnerFactor, softmaxSOuterFactor, tilingData);
        OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "adjust tiling cv diff fail"),
                        return ret);
        softmaxSInnerFactor = sInnerFactor;
    }

    uint32_t isKvContinuous = contextKeyParams.isKvContinuous;
    uint32_t fromFused = contextKeyParams.fromFused;
    tilingData.promptAttentionSingleCoreParams.set_singleProcessSOuterSize(sOuterFactor);
    tilingData.promptAttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerFactor);
    tilingData.promptAttentionBaseParams.set_splitS2(splitS2);
    tilingData.promptAttentionBaseParams.set_splitD(splitD);
    tilingData.promptAttentionBaseParams.set_softmaxOuterSize(softmaxSOuterFactor);
    tilingData.promptAttentionBaseParams.set_usePseShift(usePseShift);
    tilingData.promptAttentionBaseParams.set_pseShiftTypeByteNum(pseShiftTypeByteNum);
    tilingData.promptAttentionBaseParams.set_pseMaskMaxSize(pseMaskMaxSize);
    tilingData.promptAttentionSingleCoreParams.set_pseShiftBatch(pseShiftBatch);
    tilingData.promptAttentionBaseParams.set_pseShiftS1Size(pseShiftS1);
    tilingData.promptAttentionBaseParams.set_pseShiftS2Size(pseShiftS2);
    tilingData.promptAttentionBaseParams.set_isKvContinuous(isKvContinuous);
    tilingData.promptAttentionBaseParams.set_isQHasLeftPadding(contextKeyParams.queryPaddingSize != nullptr ? 1 : 0);
    tilingData.promptAttentionBaseParams.set_isKVHasLeftPadding(contextKeyParams.kvPaddingSize != nullptr ? 1 : 0);
    tilingData.promptAttentionBaseParams.set_fromFused((fromFused == FROM_FUSED_FLAG) ? 1 : 0);
    tilingData.promptAttentionBaseParams.set_isBSNDOut(contextKeyParams.isBSNDOut);
    tilingData.promptAttentionBaseParams.set_isSoftMaxLseEnable(contextKeyParams.isSoftMaxLseEnable);

    // compute tilingdata
    if (EnableSplitSeqOneN(tilingData, contextKeyParams) && (tilingMod == TilingMod::CVDIFF)) {
        PromptFlashAttentionSplitSeqOneN(tilingData, coreNum, false);
    } else {
        if (useNewTiling) {
            PromptFlashAttentionSplitNSNew(contextKeyParams, tilingData, coreNum, actualSeqLengths, actualSeqLengthsKV, actualSharedPrefixLen, useBalanceTiling);
        } else {
            PromptFlashAttentionSplitNS(contextKeyParams, tilingData, coreNum, actualSeqLengths);
        }
    }

    if (needInit == 1) {
        PromptFlashAttentionInitOutputSplit(outShape->GetStorageShape().GetShapeSize(), tilingData, coreNum);
    }

    if (contextKeyParams.isSoftMaxLseEnable) {
        PromptFlashAttentionInitSoftmaxLseOutputSplit(SoftmaxLseOutShape->GetStorageShape().GetShapeSize(), tilingData);
    }

    ge::graphStatus tilingRet = TilingGetTilingKeyAttentionAscendC(tilingKey, contextKeyParams, useNewTiling, tilingData);
    OPS_ERR_IF(tilingRet != ge::GRAPH_SUCCESS,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get tilingKey fail"),
                            return tilingRet);

    if ((splitS2 == 1) && (splitD == 1)) {
        tilingKey = DSPLIT_S2_D_TILING_KEY;
    }

    if ((splitS2 == 0) && (splitD == 1)) {
        tilingKey = DSPLIT_S2_TILING_KEY;
    }
    tilingRet = PromptFlashAttentionApiTiling(tilingData, outputDataTypeSize, sOuterFactor, softmaxSInnerFactor, softmaxSOuterFactor);
    OPS_ERR_IF(tilingRet != ge::GRAPH_SUCCESS,
                            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Get apiTiling fail"),
                            return tilingRet);

    blockDimToBeSet = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    size_t* workspaces = contextKeyParams.workspaceSize;;
    workspaces[0] = GetPFAWorkSpaceSize(tilingData);

    OPS_LOG_I(contextKeyParams.opName, "The Tiling key is %lu", tilingKey);

    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::InferTilingMod(const ContextParamsForPFATiling& contextKeyParams, const std::vector<int64_t>& actualSeqLengths, const std::vector<int64_t>& actualSeqLengthsKV,
                                                uint32_t actualSeqArrayLen, uint32_t hDivN, uint32_t seqInnerSize, int32_t sparseModeVal)
{
    if (hDivN > DSPLIT_THRESHOLDS_512) {   // D切分阈值 // S1S2D切分fp16和int8类型
        splitD = 1;
    }

    if ((seqInnerSize <= DSPLIT_THRESHOLDS_512) && (splitD == 1)) {
        splitS2 = 0;
    }

    if ((curShortSocName != platform_ascendc::SocVersion::ASCEND310P) &&
        (splitD != 1) && (isDNoTail == true)) {
        tilingMod = TilingMod::CVDIFF;
    }

    // 判断是否使用norm模板
    if (curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
        int64_t minActualSeqLengths = INT64_MAX;
        int64_t minActualSeqLengthsKV = INT64_MAX;
        for (uint32_t i = 0; i < actualSeqArrayLen; ++i) {
            minActualSeqLengths = std::min(minActualSeqLengths, actualSeqLengths[i]);
            minActualSeqLengthsKV = std::min(minActualSeqLengthsKV, actualSeqLengthsKV[i]);
        }
        if (minActualSeqLengths >= MATMUL_NORM_MIN_SEQ && minActualSeqLengthsKV >= MATMUL_NORM_MIN_SEQ && hDivN == MATMUL_NORM_MIN_HEADSIZE &&
            inputType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_FLOAT16 &&
            contextKeyParams.maskDataType == ge::DT_BOOL && outputType == ge::DT_FLOAT16 && usePseShift == 0 &&
            inputLayout == InputLayout::BNSD && sparseModeVal == SPARSE_MODE_BAND && (!enablePA)) {     // 当前仅针对X1场景开放matmul norm模板
            enableMatmulNorm = true;
        }
    }
}

ge::graphStatus PromptFlashAttentionTiling::AdjustCVTiling(uint32_t hDivN, uint32_t n, int64_t middleActualSeqLengths,
                                                           int64_t ubSize, int64_t l1Size, int64_t l0CSize,
                                                           uint32_t maskElemSize, uint32_t& sOuterFactor,
                                                           uint32_t& sInnerFactor, PromptFlashAttentionTilingData& tilingData)
{
    // D不切分，S2固定切成128大小，S1调整大小做切分
    uint32_t minFactor = 128U;  // Souter
    uint32_t rectangleFactor = 128U; // Sinner
    uint32_t seqFactorThreshold = 128U;
    uint32_t dSplitFactor = hDivN;
    // 310P涉及nz2nd转换 当前不能随意提升基本块大小
    if (curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
        const uint32_t littleDLimit = 64;
        if ((tilingData.promptAttentionBaseParams.get_useMask() == 0) && (hDivN <= littleDLimit)) {
            // 如果没有配置attentionMask， 可以节省attentionMask的UB空间用于softmax计算
            // 此场景中，在d比较小的情况下，可以调整基本块Sinner的大小到256，提升计算性能
            rectangleFactor = 256;
        }
        // 策略 当分核心不够时 对 souter 初始值减半，最低到32
        while (n * middleActualSeqLengths / seqFactorThreshold <= coreNum) {
            seqFactorThreshold = seqFactorThreshold / 2;  // div 2
            if (seqFactorThreshold <= 32) { // 最低到32
                break;
            }
        }
    }

    std::queue<uint32_t> rectangleQueue;
    if (GetRectangleFactor(seqFactorThreshold, rectangleQueue) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    minFactor = rectangleQueue.front();
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        minFactor = std::min(minFactor, (tilingData.promptAttentionBaseParams.get_seqSize() + 16 - 1) / 16 * 16);
        rectangleFactor = std::min(rectangleFactor, (tilingData.promptAttentionBaseParams.get_seqInnerSize() + 16 - 1) / 16 * 16);
    }

    while (true) {
        bool updateDivRect = false;
        if (PromptFlashAttentionCheckArgsLegal(tilingData, ubSize, l1Size, l0CSize,
            softmaxDataTypeSize, minFactor, rectangleFactor, updateDivRect, maskElemSize, dSplitFactor)) {
            break;
        }
        if (updateDivRect) {
            rectangleQueue.pop();
            if (rectangleQueue.size() == 0) {
                return ge::GRAPH_FAILED;
            }
            minFactor = (rectangleQueue.front());
        }
    }
    sOuterFactor = minFactor;
    sInnerFactor = rectangleFactor;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionCVDiffSetTensorSize(
    PromptFlashAttentionTilingData& tilingData,
    PromptAttentionSingleCoreTensorSize& tensorSize, uint32_t sOuterFactor,
    uint32_t sInnerFactor, uint32_t softmaxSOuterFactor)
{
    if (usePseShift == 0) {
        tensorSize.set_pseShiftUbSize(0);
    } else {
        tensorSize.set_pseShiftUbSize(softmaxSOuterFactor * sInnerFactor);
    }

    tensorSize.set_attenMaskUbSize(softmaxSOuterFactor * sInnerFactor);
    tensorSize.set_mmResUbSize(tensorSize.get_attenMaskUbSize());
    tensorSize.set_softmaxMaxSize(sOuterFactor * (BYTE_BLOCK / sizeof(float)));
    tensorSize.set_maskSize(tensorSize.get_mmResUbSize());
    tensorSize.set_softmaxSumSize(tensorSize.get_softmaxMaxSize());
    tensorSize.set_softmaxExpSize(sOuterFactor * tilingData.promptAttentionBaseParams.get_softmaxTypeByteNum());
    tensorSize.set_softmaxValueSize(sOuterFactor * sInnerFactor);
    tensorSize.set_bmm2ResUbSize(sOuterFactor * tilingData.promptAttentionBaseParams.get_alignedHeadSize());
    tensorSize.set_tmpMMResBmm2PreUbSize(std::max(tensorSize.get_mmResUbSize(), tensorSize.get_bmm2ResUbSize()));
    tensorSize.set_tmpSoftmaxBmm2UbSize(SOFTMAX_BUFFER_NUM * tensorSize.get_softmaxMaxSize());

    if (tilingData.promptAttentionBaseParams.get_maskTypeByteNum() == (BYTE_BLOCK / BOOLSIZE)) {
        tensorSize.set_selectSpaceUbSize(
            GetSelectWithBytesMaskMinTmpSize(Shape({softmaxSOuterFactor, sInnerFactor}), Shape({1}), 1,
            Shape({softmaxSOuterFactor, sInnerFactor}), 1, false));
    } else {
        tensorSize.set_selectSpaceUbSize(0);
    }
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::PromptFlashAttentionComputeCVDiffParams(PromptFlashAttentionTilingData& tilingData,
    int64_t ubSize, int64_t l1Size, int64_t l0CSize, uint32_t typeByteSize,
    uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t maskTypeSize, uint32_t &softmaxSOuterFactor)
{
    bool res = false;
    int32_t l1SizeRemain = l1Size;
    if (AdjustBasicBlock(tilingData, sOuterFactor) != ge::GRAPH_SUCCESS) {
            return false;
    }

    if (inputType == ge::DT_INT8) {
        res = FindOptimalTilingSouter(tilingData, sOuterFactor, sInnerFactor, softmaxSOuterFactor, ubSize, typeByteSize, maskTypeSize);
    } else {
        res = FindOptimalTilingBasicBLock(tilingData, sOuterFactor, sInnerFactor, softmaxSOuterFactor, ubSize, typeByteSize, maskTypeSize);
    }
    OPS_ERR_IF(res == false,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "FindOptimalTilingBasicBLock failed!"),
                    return false);

    // kvcache antiquant tiling
    if (enableKvAntiquant) {
        int32_t sKvAntiquantFactor = sInnerFactor;
        uint32_t kvAntiquantApiSizeMax = 0;
        uint32_t kvAntiquantApiSize = 0;
        auto srcShape = Shape({sKvAntiquantFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
        auto scaleShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
        int64_t ubSizeRemainTmp = ubSizeRemain;
        do {
            srcShape = Shape({sKvAntiquantFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
            GetAscendAntiQuantMaxMinTmpSize(srcShape, scaleShape, false, ge::DT_INT8, inputType, kvAntiquantApiSizeMax, kvAntiquantApiSize);
            ubSizeRemain = ubSizeRemainTmp - kvAntiquantApiSize - tilingData.promptAttentionBaseParams.get_alignedHeadSize() * 2 * FLOAT16SIZE - // scale offset fp16
                (sKvAntiquantFactor * tilingData.promptAttentionBaseParams.get_alignedHeadSize() * (INT8SIZE + FLOAT16SIZE) * 1);   // 输入输出
            if (ubSizeRemain < 0) {
                sKvAntiquantFactor -= 1;
            }
        } while (ubSizeRemain < 0 && sKvAntiquantFactor > 0);
        OPS_ERR_IF(sKvAntiquantFactor <= 0,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "cannot find valid sKvAntiquantFactor!"),
                        return false);
        tilingData.promptAttentionTensorSizeRect.set_kvAntiquantUbSize(sKvAntiquantFactor * tilingData.promptAttentionBaseParams.get_alignedHeadSize());
        tilingData.promptAttentionSingleCoreParams.set_kvAntiquantSInnerSize(sKvAntiquantFactor);
    }

    const uint32_t dSplitFactorBmm2 = 128U;
    res = PromptFlashAttentionCheckBmm1(tilingData, tilingData.bmm1TilingDataRect,
            l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, true, true);
    OPS_ERR_IF(res == false,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "PromptFlashAttentionCheckBmm1 failed!"),
                    return false);

    res = PromptFlashAttentionCheckBmm2(tilingData, tilingData.bmm2TilingDataRect,
            l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, dSplitFactorBmm2, true, true);
    OPS_ERR_IF(res == false,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "PromptFlashAttentionCheckBmm2 failed!"),
                    return false);

    return true;
}

bool PromptFlashAttentionTiling::FindOptimalTilingSouter(PromptFlashAttentionTilingData& tilingData,
    uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
    int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize)
{
    // 此函数为固定Sinner为1024或kvs，减少Souter来使ub够用,当前只有Int8使用
    auto tmpShape = Shape({softmaxSOuterFactor, sInnerFactor});
    int64_t softmaxTmpSize = 0;
    int64_t softmaxFlashTmpSize = 0;
    int64_t queueBufferSize = 0;

    // 临时方案，先用int32_t类型的Tmp变量计算，后续做优化，将入参改为int32_t类型
    int32_t sOuterFactorTmp = (int32_t)sOuterFactor;
    int32_t sInnerFactorTmp = (int32_t)sInnerFactor;
    int32_t softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
    const int32_t sOuterFactorStep = 16;
    const int32_t softmaxSOuterFactorStep = 8;

    int64_t pseShiftBufferSize = 0;
    pseMaskMaxSize = std::max(maskTypeSize, pseShiftElemSize);

    uint32_t pseShiftCastSize = 0U;
    if (usePseShift == 1 && (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || pseShiftElemType == ge::DT_BF16)) {
        pseShiftCastSize = FLOAT32SIZE;   // 在高精度生效或者bf16情况，pse需要做cast，申请ub
    }

    uint32_t kvAntiquantApiSizeMax = 0U;
    uint32_t kvAntiquantApiSize = 0U;
    auto srcShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    auto scaleShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    GetAscendAntiQuantMaxMinTmpSize(srcShape, scaleShape, false, ge::DT_INT8, inputType, kvAntiquantApiSizeMax, kvAntiquantApiSize);
    // 最小antiquant ub: api + scale offset + 输入输出每次只处理一行
    int64_t minAntiquantUbSizeNeed = kvAntiquantApiSize + tilingData.promptAttentionBaseParams.get_alignedHeadSize() * 2 * FLOAT16SIZE + // scale offset fp16
                tilingData.promptAttentionBaseParams.get_alignedHeadSize() * (INT8SIZE + FLOAT16SIZE); // 输入int8 输出fp16

    ubSizeRemain = 0;
    while (ubSizeRemain <= 0 && sOuterFactorTmp > 0) {
        softmaxTmpSize = 0;
        softmaxFlashTmpSize = 0;
        while ((softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) && (softmaxSOuterFactorTmp > 0)) {
            tmpShape = Shape({softmaxSOuterFactorTmp, sInnerFactorTmp});
            softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
            softmaxFlashTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, sizeof(float), true, true);
            if (softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) {
                softmaxSOuterFactorTmp -= softmaxSOuterFactorStep;
            }
        }

        if (softmaxSOuterFactorTmp <= 0) {
            sOuterFactorTmp -= sOuterFactorStep;
            softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
            continue;
        }
        if (PromptFlashAttentionCVDiffSetTensorSize(tilingData, tilingData.promptAttentionTensorSizeRect,
                                                sOuterFactorTmp, sInnerFactorTmp, softmaxSOuterFactorTmp) != ge::GRAPH_SUCCESS) {
            return false;
        }

        queueBufferSize = tilingData.promptAttentionTensorSizeRect.get_attenMaskUbSize();
        pseShiftBufferSize = tilingData.promptAttentionTensorSizeRect.get_pseShiftUbSize();
        apiTmpSize = std::max(softmaxTmpSize, softmaxFlashTmpSize);

        int64_t maskBmm2ShareSize = std::max(int64_t(queueBufferSize * pseMaskMaxSize),
            int64_t(tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() * typeByteSize));
        ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * NUM_2 + // 2:2 mm ub
                    tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() +       // bmm2ResPrev常驻UB
                    SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                    typeByteSize - maskBmm2ShareSize - tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                    pseShiftBufferSize * pseShiftCastSize;
        if ((ubSizeRemain <= 0) || (enableKvAntiquant && ubSizeRemain < minAntiquantUbSizeNeed)) {
            sOuterFactorTmp -= sOuterFactorStep;
            sInnerFactorTmp = (int32_t)sInnerFactor;
            softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
        }
    }

    OPS_ERR_IF((sOuterFactorTmp <= 0) || (sInnerFactorTmp <= 0) || (softmaxSOuterFactorTmp <= 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "cannot find valid sOuterFactor, sInnerFactor and softmaxSOuterFactor!"),
                    return false);
    sOuterFactor = (uint32_t)sOuterFactorTmp;
    sInnerFactor = (uint32_t)sInnerFactorTmp;
    softmaxSOuterFactor = (uint32_t)softmaxSOuterFactorTmp;
    return true;
}

bool PromptFlashAttentionTiling::FindOptimalTilingBasicBLock(PromptFlashAttentionTilingData& tilingData,
    uint32_t& sOuterFactor, uint32_t &sInnerFactor, uint32_t &softmaxSOuterFactor,
    int64_t ubSize, uint32_t typeByteSize, uint32_t maskTypeSize)
{
    auto tmpShape = Shape({softmaxSOuterFactor, sInnerFactor});
    int64_t softmaxTmpSize = 0;
    int64_t softmaxFlashTmpSize = 0;
    int64_t queueBufferSize = 0;

    // 临时方案，先用int32_t类型的Tmp变量计算，后续做优化，将入参改为int32_t类型
    int32_t sOuterFactorTmp = (int32_t)sOuterFactor;
    int32_t sInnerFactorTmp = (int32_t)sInnerFactor;
    int32_t softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
    const int32_t sOuterFactorStep = 16;
    int32_t sInnerFactorStep = 64;
    const int32_t softmaxSOuterFactorStep = 8;

    int64_t pseShiftBufferSize = 0;
    pseMaskMaxSize = std::max(maskTypeSize, pseShiftElemSize);

    uint32_t pseShiftCastSize = 0U;
    if ((usePseShift == 1) && (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || pseShiftElemType == ge::DT_BF16)) {
        pseShiftCastSize = FLOAT32SIZE;   // 在高精度生效或者bf16情况，pse需要做cast，申请ub
    }
    if (enablePA) {
        sInnerFactorStep = tilingData.promptAttentionBaseParams.get_blockSize();
    }
    uint32_t kvAntiquantApiSizeMax = 0U;
    uint32_t kvAntiquantApiSize = 0U;
    auto srcShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    auto scaleShape = Shape({1, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    GetAscendAntiQuantMaxMinTmpSize(srcShape, scaleShape, false, ge::DT_INT8, inputType, kvAntiquantApiSizeMax, kvAntiquantApiSize);
    // 最小antiquant ub: api + scale offset + 输入输出每次只处理一行
    int64_t minAntiquantUbSizeNeed = kvAntiquantApiSize + tilingData.promptAttentionBaseParams.get_alignedHeadSize() * 2 * FLOAT16SIZE + // scale offset fp16
                tilingData.promptAttentionBaseParams.get_alignedHeadSize() * (INT8SIZE + FLOAT16SIZE); // 输入int8 输出fp16

    // post quant perchannel ub size
    int64_t postQuantUbSize = 0;
    if (tilingData.promptAttentionBaseParams.get_isQuant2Perchannel() == 1) {
        uint32_t floatSize = 4;
        uint32_t bf16Size = 2;
        postQuantUbSize = 2 * floatSize * tilingData.promptAttentionBaseParams.get_headSize();     // 2 为 scale2 和 offset2
        if (tilingData.promptAttentionBaseParams.get_isQuant2BF16() == 1) {
            postQuantUbSize += 2 * bf16Size * tilingData.promptAttentionBaseParams.get_headSize(); // 2 为 scale2 和 offset2
        }
    }

    // AscendQuant预留ub空间
    auto postQuantSrcShape = Shape({sOuterFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
    uint32_t bmm2ResTypeSize = (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || (inputType == ge::DT_BF16)) ? FLOAT32SIZE : FLOAT16SIZE;
    uint32_t postQuantApiSizeMax = 0U;
    uint32_t postQuantApiSizeMin = 0U;

    ubSizeRemain = 0;
    while (ubSizeRemain <= 0 && sOuterFactorTmp > 0) {
        while ((ubSizeRemain <= 0 && sInnerFactorTmp > 0) || (enableKvAntiquant && ubSizeRemain < minAntiquantUbSizeNeed && sInnerFactorTmp > 0)) {
            softmaxTmpSize = 0;
            softmaxFlashTmpSize = 0;
            while ((softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) && (softmaxSOuterFactorTmp > 0)) {
                tmpShape = Shape({softmaxSOuterFactorTmp, sInnerFactorTmp});
                softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
                softmaxFlashTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, sizeof(float), true, true);
                if (softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) {
                    softmaxSOuterFactorTmp -= softmaxSOuterFactorStep;
                }
            }

            if (softmaxSOuterFactorTmp <= 0) {
                sInnerFactorTmp -= sInnerFactorStep;
                softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
                continue;
            }

            if (PromptFlashAttentionCVDiffSetTensorSize(tilingData, tilingData.promptAttentionTensorSizeRect,
                                                    sOuterFactorTmp, sInnerFactorTmp, softmaxSOuterFactorTmp) != ge::GRAPH_SUCCESS) {
                return false;
            }

            queueBufferSize = tilingData.promptAttentionTensorSizeRect.get_attenMaskUbSize();
            pseShiftBufferSize = tilingData.promptAttentionTensorSizeRect.get_pseShiftUbSize();
            apiTmpSize = std::max(softmaxTmpSize, softmaxFlashTmpSize);

            if (outputType == ge::DT_INT8) {
                postQuantSrcShape = Shape({sOuterFactorTmp, tilingData.promptAttentionBaseParams.get_alignedHeadSize()});
                GetAscendQuantMaxMinTmpSize(postQuantSrcShape, bmm2ResTypeSize, postQuantApiSizeMax, postQuantApiSizeMin);
            }

            int64_t maskBmm2ShareSize = std::max(int64_t(queueBufferSize * pseMaskMaxSize),
                int64_t(tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() * typeByteSize));
            ubSizeRemain = ubSize - apiTmpSize - (tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * NUM_2 + // 2:2 mm ub
                        tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() +       // bmm2ResPrev常驻UB
                        SOFTMAX_BUFFER_NUM * tilingData.promptAttentionTensorSizeRect.get_softmaxExpSize()) *
                        typeByteSize - maskBmm2ShareSize - tilingData.promptAttentionTensorSizeRect.get_selectSpaceUbSize() -
                        pseShiftBufferSize * pseShiftCastSize - postQuantUbSize - postQuantApiSizeMin;
            if (ubSizeRemain <= 0 || (enableKvAntiquant && ubSizeRemain < minAntiquantUbSizeNeed)) {
                sInnerFactorTmp -= sInnerFactorStep;
                softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
            }
        }

        if ((ubSizeRemain <= 0) || (enableKvAntiquant && ubSizeRemain < minAntiquantUbSizeNeed)) {
            sOuterFactorTmp -= sOuterFactorStep;
            sInnerFactorTmp = (int32_t)sInnerFactor;
            softmaxSOuterFactorTmp = (int32_t)softmaxSOuterFactor;
        }
    }

    OPS_ERR_IF((sOuterFactorTmp <= 0) || (sInnerFactorTmp <= 0) || (softmaxSOuterFactorTmp <= 0),
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "cannot find valid sOuterFactor, sInnerFactor and softmaxSOuterFactor!"),
                    return false);
    sOuterFactor = (uint32_t)sOuterFactorTmp;
    sInnerFactor = (uint32_t)sInnerFactorTmp;
    softmaxSOuterFactor = (uint32_t)softmaxSOuterFactorTmp;
    return true;
}

ge::graphStatus PromptFlashAttentionTiling::AdjustCVTilingCVDiff(int64_t ubSize, int64_t l1Size, int64_t l0CSize,
    uint32_t maskElemSize, uint32_t& sOuterFactor, uint32_t& sInnerFactor, uint32_t& softmaxSOuterFactor,
    PromptFlashAttentionTilingData& tilingData)
{
    // 新softmax tiling策略，mm1 mm2 统一big tiling（如: mm1=256x512, mm2=256xhead_size), softmax根据ub空间横向切big tiling成多次long tiling计算（如：softmax=32x512）
    // softmax根据ub空间横向切big tiling成多次long tiling计算（如：softmax=32x512）
    uint32_t minFactor = CVDIFF_SOUTER_FACTOR_DEFAULT;
    uint32_t rectangleFactor = CVDIFF_SINNER_FACTOR_DEFAULT;
    const uint32_t softmaxUbSize = CVDIFF_MM1RES_UB_SIZE;
    if ((tilingData.promptAttentionBaseParams.get_seqInnerSize() <= CVDIFF_SMALL_KV_THRESHOLDS) && (inputType != ge::DT_INT8)) {
        rectangleFactor = CVDIFF_SINNER_FACTOR_SMALL_KVS;
    }

    softmaxSOuterFactor = softmaxUbSize / rectangleFactor;

    if (((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) ||
        (inputType == ge::DT_BF16)) {      // 高精度模式或bf16生效时，起始tiling块做调整
        if (tilingData.promptAttentionBaseParams.get_alignedHeadSize() >= 200) {            // D: [200, ...)
            minFactor = 64;
            rectangleFactor = 512;
            softmaxSOuterFactor = 8;
        } else if (tilingData.promptAttentionBaseParams.get_alignedHeadSize() >= 128) {     // D: [128, 200)
            minFactor = 128;
            rectangleFactor = 512;
            softmaxSOuterFactor = 8;
        } else if (tilingData.promptAttentionBaseParams.get_alignedHeadSize() >= 32) {      // D: [32, 128)
            minFactor = 128;
            rectangleFactor = 512;
            softmaxSOuterFactor = 16;
        } else {                                                                            // D: (0, 32)
            minFactor = 128;
            rectangleFactor = 512;
            softmaxSOuterFactor = 32;
        }
    }
    if (enablePA) {
        minFactor = 64;  // PA场景，Souter从64开始切，尽量保证Sinner不切，从而让Single是blockSize倍数
    }
    if (tilingData.promptAttentionBaseParams.get_seqSize() <= CVDIFF_SMALL_QS_THRESHOLDS) { // 最小基本块大小
        minFactor = CVDIFF_SMALL_QS_THRESHOLDS; // 调小S1，避免mm1 多余计算
        if ((tilingData.promptAttentionBaseParams.get_seqInnerSize() > CVDIFF_SINNER_FACTOR_SMALL_QS)
            && (tilingData.promptAttentionBaseParams.get_useMask() == 0)) { //只有在没有mask的场景可以设置为2048
            rectangleFactor = CVDIFF_SINNER_FACTOR_SMALL_QS; // 调大S2，提高softmax 吞吐量
        }
        softmaxSOuterFactor = softmaxUbSize / rectangleFactor;

        // 调小 softmaxouter，减小至真实 souter
        if ( tilingData.promptAttentionBaseParams.get_seqSize() < softmaxSOuterFactor) {
            softmaxSOuterFactor = tilingData.promptAttentionBaseParams.get_seqSize();
        }
    }

    if (enableKvAntiquant) {
        uint32_t sInnerMax = 1024 * 256 / tilingData.promptAttentionBaseParams.get_alignedHeadSize();   // workspace增加不大于50M
        sInnerMax = (sInnerMax + THIRTY_ONE) / UB_ALIGN * UB_ALIGN;
        rectangleFactor = rectangleFactor > sInnerMax ? sInnerMax : rectangleFactor;
        softmaxSOuterFactor = softmaxUbSize / rectangleFactor;
    }

    bool res = PromptFlashAttentionComputeCVDiffParams(tilingData, ubSize, l1Size, l0CSize,
                    softmaxDataTypeSize, minFactor, rectangleFactor, maskElemSize, softmaxSOuterFactor);
    OPS_ERR_IF(res == false,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "PromptFlashAttentionComputeCVDiffParams failed!"),
                    return ge::GRAPH_FAILED);

    sOuterFactor = minFactor;
    sInnerFactor = rectangleFactor;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPromptFlashAttention(gert::TilingContext* context) {
    if (context == nullptr) {
        OPS_LOG_E("PromptFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    if (context->GetRawTilingData() == nullptr) {
        OPS_LOG_E("PromptFlashAttention", "tiling context GetRawTilingData is nullptr!");
        return ge::GRAPH_FAILED;
    }
    PromptFlashAttentionTiling flashTiling(nullptr);
    PromptFlashAttentionTilingData tilingData;
    OPS_ERR_IF(memset_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
               0, context->GetRawTilingData()->GetCapacity()) != EOK,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "fail to memset tiling data"),
               return ge::GRAPH_FAILED);
    ContextParamsForPFATiling contextParamsForPFATiling = {
        .pseShift = nullptr,
        .attentionMask = nullptr,
        .actualSeqenceLengthQ = nullptr,
        .actualSeqenceLengthKV = nullptr,
        .antiquantScale = nullptr,
        .antiquantOffset = nullptr,
        .queryPaddingSize = nullptr,
        .kvPaddingSize = nullptr,
        .blockTable = nullptr,
        .keySharedPrefix = nullptr,
        .valueSharedPrefix = nullptr,
        .actualSharedPrefixLen = nullptr,
        .inputDataType = ge::DataType::DT_FLOAT16,
        .kDataType = ge::DataType::DT_FLOAT16,
        .vDataType = ge::DataType::DT_FLOAT16,
        .pseShiftDataType = ge::DataType::DT_FLOAT16,
        .maskDataType = ge::DataType::DT_FLOAT16,
        .blockTableType = ge::DataType::DT_FLOAT16,
        .outputDataType = ge::DataType::DT_FLOAT16,
        .opName = nullptr,
        .queryInputShape = nullptr,
        .keyInputShape = nullptr,
        .valueInputShape = nullptr,
        .pseShiftShape = nullptr,
        .attentionMaskShape = nullptr,
        .deqScale1Shape = nullptr,
        .scale1Shape = nullptr,
        .deqScale2Shape = nullptr,
        .scale2Shape = nullptr,
        .offset2Shape = nullptr,
        .antiquantScaleShape = nullptr,
        .antiquantOffsetShape = nullptr,
        .blockTableShape = nullptr,
        .outputShape = nullptr,
        .lseoutputShape = nullptr,
        .innerPrecisePtr = nullptr,
        .headsNumber = nullptr,
        .sparseMode = nullptr,
        .preToken = nullptr,
        .nextToken = nullptr,
        .scaleValue = nullptr,
        .blockSize = nullptr,
        .layout = nullptr,
        .numKeyValueHeads = nullptr,
        .workspaceSize = nullptr,
        .compileInfoPtr = nullptr,
        .deqScaleType = ge::DataType::DT_FLOAT16,
        .deqScale2Type = ge::DataType::DT_FLOAT16,
        .quantScale2Type = ge::DataType::DT_FLOAT16,
        .quantOffset2Type = ge::DataType::DT_FLOAT16,
        .isKvContinuous = 0,
        .kTensorList = {nullptr},
        .vTensorList = {nullptr},
        .maxKVs  =0,
        .fromFused = 0,
        .emptyTensor = 0,
        .isBSNDOut = 0,
        .softmaxLseFlag = nullptr,
        .isSoftMaxLseEnable = false,
        .fromTilingSink = 0
    };
    auto ret = ConvertContextToPFAParams(context, contextParamsForPFATiling);
    uint64_t tilingKey = 7; // 7: default tiling key
    uint32_t blockDimToBeSet;
    ret = flashTiling.RunBigKernelTilingWithParams(contextParamsForPFATiling, tilingKey, blockDimToBeSet, tilingData);
    tilingKey += BENCHMARK_TILING_KEY;
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDimToBeSet);
    flashTiling.PromptFlashAttentionSetTilingData(context, tilingData);
    return ret;
}
}