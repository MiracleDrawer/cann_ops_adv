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
 * \file fa_case.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad 测试用例.
 */

#include "fa_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/fa/tiling_data.h"
#include "tiling/tiling_templates_registry.h"

using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;
using FaCase = ops::adv::tests::fa::FaCase;

namespace optiling {
ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionScore(gert::TilingContext *context);
ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionGradScore(gert::TilingContext *context);
} // namespace optiling

namespace {

const size_t FAS_PREFIX_INPUT_INDEX = 7UL;
const size_t FAS_ACTUAL_SEQ_LENGTH_INPUT_INDEX = 8UL;
const size_t FAS_ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 9UL;
const size_t FAS_Q_START_IDX_INPUT_INDEX = 10UL;
const size_t FAS_KV_START_IDX_INPUT_INDEX = 11UL;
const size_t FAG_PREFIX_INPUT_INDEX = 12UL;
const size_t FAG_ACTUAL_SEQ_LENGTH_INPUT_INDEX = 13UL;
const size_t FAG_ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX = 14UL;

ASCENDC_EXTERN_C ge::graphStatus FlashAttentionScoreTilingFuncStub(gert::TilingContext *context)
{
    auto *faCase = static_cast<FaCase *>(Case::GetCurrentCase());
    if (faCase != nullptr) {
        FaCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.prefixTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_PREFIX_INPUT_INDEX));
        p.actSeqQLenTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_ACTUAL_SEQ_LENGTH_INPUT_INDEX));
        p.actSeqKVLenTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX));
        p.qStartIdxTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_Q_START_IDX_INPUT_INDEX));
        p.kvStartIdxTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAS_KV_START_IDX_INPUT_INDEX));

        if (faCase->DoOpTiling(p)) {
            return p.ret;
        }
        return faCase->fasOriginTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus FlashAttentionScoreGradTilingFuncStub(gert::TilingContext *context)
{
    auto *faCase = static_cast<FaCase *>(Case::GetCurrentCase());
    if (faCase != nullptr) {
        FaCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.prefixTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAG_PREFIX_INPUT_INDEX));
        p.actSeqQLenTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAG_ACTUAL_SEQ_LENGTH_INPUT_INDEX));
        p.actSeqKVLenTensor =
            const_cast<gert::Tensor *>(context->GetOptionalInputTensor(FAG_ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX));
        if (faCase->DoOpTiling(p)) {
            return p.ret;
        }
        return faCase->fagOriginTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

} // namespace

typedef void (*FasKernelFunc)(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                              __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                              __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
                              __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx,
                              __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                              __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling);

typedef void (*FagKernelFunc)(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dy,
                              __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *padding_mask,
                              __gm__ uint8_t *atten_mask, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                              __gm__ uint8_t *softmax_in, __gm__ uint8_t *attention_in, __gm__ uint8_t *prefix,
                              __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen,
                              __gm__ uint8_t *q_start_idx, __gm__ uint8_t *kv_start_idx, __gm__ uint8_t *dq,
                              __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse, __gm__ uint8_t *workspace,
                              __gm__ uint8_t *tiling_data);

using namespace ops::adv::tests::fa;
using TensorIntf = ops::adv::tests::utils::TensorIntf;

bool RunFlashAttention(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                       std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (FasKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[0]->GetDevData(),  // query
                inputs[1]->GetDevData(),  // key
                inputs[2]->GetDevData(),  // value
                inputs[3]->GetDevData(),  // pse
                inputs[4]->GetDevData(),  // dropMask
                inputs[5]->GetDevData(),  // paddingMask
                inputs[6]->GetDevData(),  // attenMask
                inputs[7]->GetDevData(),  // prefix
                inputs[8]->GetDevData(),  // actSeqQLens
                inputs[9]->GetDevData(),  // actSeqKVLens
                inputs[10]->GetDevData(), // qStartIdx
                inputs[11]->GetDevData(), // kvStartIdx
                outputs[0]->GetDevData(), // softmaxMax
                outputs[1]->GetDevData(), // softmaxSum
                outputs[2]->GetDevData(), // softmaxRes
                outputs[3]->GetDevData(), // attenRes
                workspace, tilingData);
    return true;
}

bool RunFlashAttentionGrad(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                           std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (FagKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[0]->GetDevData(),  // query
                inputs[1]->GetDevData(),  // key
                inputs[2]->GetDevData(),  // value
                inputs[3]->GetDevData(),  // dy
                inputs[4]->GetDevData(),  // pse
                inputs[5]->GetDevData(),  // dropMask
                inputs[6]->GetDevData(),  // paddingMask
                inputs[7]->GetDevData(),  // attenMask
                inputs[8]->GetDevData(),  // softmaxMax
                inputs[9]->GetDevData(),  // softmaxSum
                inputs[10]->GetDevData(), // softMaxRes
                inputs[11]->GetDevData(), // attenRes
                inputs[12]->GetDevData(), // prefix
                inputs[13]->GetDevData(), // actualSeqQLen
                inputs[14]->GetDevData(), // actualSeqKvLen
                inputs[15]->GetDevData(), // qStartIdx
                inputs[16]->GetDevData(), // kvStartIdx
                outputs[0]->GetDevData(), // dq
                outputs[1]->GetDevData(), // dk
                outputs[2]->GetDevData(), // dv
                outputs[3]->GetDevData(), // dpse
                workspace, tilingData);
    return true;
}

FaCase::FaCase() : FaCase("Undefined", true, "", OpInfo(), OpInfo(), FaParam(), kTilingTemplatePriority_Invalid)
{
}

FaCase::FaCase(const char *name, bool enable, const char *dbgInfo, OpInfo forward, OpInfo reverse, FaParam param,
               int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), forward(std::move(forward)), reverse(std::move(reverse)),
      param(std::move(param))
{
    this->forward.name = "FlashAttentionScore";
    this->reverse.name = "FlashAttentionScoreGrad";
}

bool FaCase::Run()
{
    if (!enable) {
        return true;
    }
    if (!forward.ProcessTiling(name)) {
        return false;
    }
    if (!forward.ProcessKernel(name)) {
        return false;
    }
    if (!reverse.ProcessTiling(name)) {
        return false;
    }
    if (!reverse.ProcessKernel(name)) {
        return false;
    }
    return true;
}

bool FaCase::InitParam()
{
    if (!param.Init()) {
        return false;
    }
    return true;
}

bool FaCase::InitOpInfo()
{
    std::string kernelSoRelPath = "src/transformer/flash_attention/libUTest_Fa_OpKernel_";
    if (param.dtype == ge::DataType::DT_FLOAT16) {
        kernelSoRelPath += "fp16.so";
    } else if (param.dtype == ge::DataType::DT_FLOAT) {
        kernelSoRelPath += "fp32.so";
    } else {
        kernelSoRelPath += "bf16.so";
    }

    bool rst = forwardCtx.SetOpName(forward.name.c_str());
    rst = rst && forwardCtx.SetDeterministic(forward.ctr.deterministic);
    rst = rst && forwardCtx.SetInputs({&param.query, &param.key, &param.value, &param.pse, &param.dropMask,
                                       &param.paddingMask, &param.attenMask, &param.prefix, &param.actualSeqQLen,
                                       &param.actualSeqKvLen, &param.qStartIdx, &param.kvStartIdx});
    rst = rst && forwardCtx.SetOutputs({&param.softmaxMax, &param.softmaxSum, &param.softmaxRes, &param.attenRes});
    rst = rst && forwardCtx.SetAttrs({{"scale_value", param.scale},
                                      {"keep_prob", param.keepProb},
                                      {"pre_tockens", param.preTokens},
                                      {"next_tockens", param.nxtTokens},
                                      {"head_num", param.n1},
                                      {"input_layout", param.layout},
                                      {"inner_precise", param.innerPrecise},
                                      {"sparse_mode", param.sparseMode},
                                      {"pse_type", param.pseType}});
    rst = rst && forwardCtx.SetKernelSoRelPath(kernelSoRelPath.c_str());
    rst = rst && forwardCtx.SetKernelFuncName("flash_attention_score");
    rst = rst && forwardCtx.SetKernelRunCbf(RunFlashAttention);
    rst = rst && forward.SetContext(&forwardCtx);
    rst = rst && reverseCtx.SetOpName(reverse.name.c_str());
    rst = rst && reverseCtx.SetDeterministic(reverse.ctr.deterministic);
    rst = rst && reverseCtx.SetInputs({&param.query, &param.key, &param.value, &param.dy, &param.pse, &param.dropMask,
                                       &param.paddingMask, &param.attenMask, &param.softmaxMax, &param.softmaxSum,
                                       &param.softmaxRes, &param.attenRes, &param.prefix, &param.actualSeqQLen,
                                       &param.actualSeqKvLen, &param.qStartIdx, &param.kvStartIdx});
    rst = rst && reverseCtx.SetOutputs({&param.dq, &param.dk, &param.dv, &param.dPse});
    rst = rst && reverseCtx.SetAttrs({{"scale_value", param.scale},
                                      {"keep_prob", param.keepProb},
                                      {"pre_tockens", param.preTokens},
                                      {"next_tockens", param.nxtTokens},
                                      {"head_num", param.n1},
                                      {"input_layout", param.layout},
                                      {"inner_precise", param.innerPrecise},
                                      {"sparse_mode", param.sparseMode},
                                      {"pse_type", param.pseType}});
    rst = rst && reverseCtx.SetTilingMaxDataSize(2560);
    rst = rst && reverseCtx.SetKernelSoRelPath(kernelSoRelPath.c_str());
    rst = rst && reverseCtx.SetKernelFuncName("flash_attention_score_grad");
    rst = rst && reverseCtx.SetKernelRunCbf(RunFlashAttentionGrad);
    rst = rst && reverse.SetContext(&reverseCtx);
    if (!rst) {
        return rst;
    }

    if (!this->InitOriginTilingFunc()) {
        return false;
    }
    IMPL_OP(FlashAttentionScore).Tiling(FlashAttentionScoreTilingFuncStub);
    IMPL_OP(FlashAttentionScoreGrad).Tiling(FlashAttentionScoreGradTilingFuncStub);

    return true;
}

bool FaCase::InitOriginTilingFunc()
{
    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    /* FlashAttentionScore FlashAttentionScoreGrad 需提供修改 TilingContext 功能 */
    /* FlashAttentionScoreGrad 需提供按指定优先级调用 Tiling 模板功能 */
    fasOriginTilingFunc =
        (gert::OpImplKernelRegistry::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingFlashAttentionScore");
    fagOriginTilingFunc =
        (gert::OpImplKernelRegistry::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingFlashAttentionGradScore");
    if (fasOriginTilingFunc == nullptr || fagOriginTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, Fas(%p), Fag(%p)", fasOriginTilingFunc, fagOriginTilingFunc);
        return false;
    }
    return true;
}

bool FaCase::InitCurrentCasePtr()
{
    Case::currentCasePtr = this;
    return true;
}

bool FaCase::DoOpTiling(DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (preTilingRunCbf != nullptr) {
        preTilingRunCbf(tilingParam);
    }
    /* 外部无法构造 Tensor 的数据, 此处进行处理 */
    if (tilingParam.prefixTensor != nullptr) {
        tilingParam.prefixTensor->SetData(gert::TensorData{param.prefixTensorData.data()});
    }
    if (tilingParam.actSeqQLenTensor != nullptr) {
        tilingParam.actSeqQLenTensor->SetData(gert::TensorData{param.actualSeqQLenTensorData.data()});
    }
    if (tilingParam.actSeqKVLenTensor != nullptr) {
        tilingParam.actSeqKVLenTensor->SetData(gert::TensorData{param.actualSeqKVLenTensorData.data()});
    }
    if (tilingParam.qStartIdxTensor != nullptr) {
        tilingParam.qStartIdxTensor->SetData(gert::TensorData{param.qStartIdxTensorData.data()});
    }
    if (tilingParam.kvStartIdxTensor != nullptr) {
        tilingParam.kvStartIdxTensor->SetData(gert::TensorData{param.kvStartIdxTensorData.data()});
    }
    /* 按优先级 Tiling */
    auto priority = tilingTemplatePriority;
    if (priority == Case::kTilingTemplatePriority_Invalid) {
        return false;
    }
    tilingParam.ret = optiling::TilingRegistry::GetInstance().DoTilingImpl(tilingParam.ctx, {priority});
    return true;
}

void FaCase::PreTilingRunCbf_SetPlatformInfoNull(FaCase::DoTilingParam &tilingParam)
{
    if (tilingParam.ctx == nullptr) {
        return;
    }
    const auto compute_node_info = tilingParam.ctx->GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
        return;
    }
    /* PlatformInfo 位于 Inputs 和 Outputs 之后 */
    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum() + 1U;
    auto kernelContext = (gert::KernelContext *)tilingParam.ctx;
    kernelContext->GetContext()->values[index] = nullptr;
}
