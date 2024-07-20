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
 * \file pfa_case.cpp
 * \brief PromptFlashAttention 测试用例.
 */
#include "pfa_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/pfa/tiling_data.h"

typedef void (*PfaKernelFunc)(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                              __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths,
                              __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* deq_scale1,
                              __gm__ uint8_t* quant_scale1, __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                              __gm__ uint8_t* quant_offset2, __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                              __gm__ uint8_t* tiling);

using namespace ops::adv::tests::pfa;
using Tensor = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunPromptFlashAttention(void* func, uint64_t tilingKey, int64_t blockDim, std::vector<Tensor*>& inputs,
                             std::vector<Tensor*>& outputs, uint8_t* workspace, uint8_t* tilingData) {
  // Kernel 运行
  auto kernelFunc = (PfaKernelFunc)func;
  ICPU_SET_TILING_KEY(tilingKey);
  ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(), inputs[3]->GetDevData(),
              inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(), inputs[7]->GetDevData(), inputs[8]->GetDevData(),
              inputs[9]->GetDevData(), inputs[10]->GetDevData(), inputs[11]->GetDevData(), outputs[0]->GetDevData(), workspace, tilingData);
  return true;
}

bool PfaCase::InitParam() {
  h = param.n * param.d;
  int64_t kvNum = param.n;
  if (param.kvNumHeads != 0) {
    kvNum = param.kvNumHeads;
  }
  int64_t kvH = kvNum * param.d;
  if (param.layout == "BSH") {
    query = Tensor("query", {param.b, 1, h}, "BSH", param.qDataType, ge::FORMAT_ND);
    key = Tensor("key", {param.b, param.s, kvH}, "BSH", param.kvDataType, ge::FORMAT_ND);
    value = Tensor("value", {param.b, param.s, kvH}, "BSH", param.kvDataType, ge::FORMAT_ND);
    attentionOut = Tensor("attentionOut", {param.b, 1, h}, "BSH", param.outDataType, ge::FORMAT_ND);
  } else if (param.layout == "BNSD") {
    query = Tensor("query", {param.b, param.n, 1, param.d}, "BNSD", param.qDataType, ge::FORMAT_ND);
    key = Tensor("key", {param.b, kvNum, param.s, param.d}, "BNSD", param.kvDataType, ge::FORMAT_ND);
    value = Tensor("value", {param.b, kvNum, param.s, param.d}, "BNSD", param.kvDataType, ge::FORMAT_ND);
    attentionOut = Tensor("attentionOut", {param.b, param.n, 1, param.d}, "BNSD", param.outDataType, ge::FORMAT_ND);
  } else if (param.layout == "BSND") {
    query = Tensor("query", {param.b, 1, param.n, param.d}, "BSND", param.qDataType, ge::FORMAT_ND);
    key = Tensor("key", {param.b, param.s, kvNum, param.d}, "BSND", param.kvDataType, ge::FORMAT_ND);
    value = Tensor("value", {param.b, param.s, kvNum, param.d}, "BSND", param.kvDataType, ge::FORMAT_ND);
    attentionOut = Tensor("attentionOut", {param.b, 1, param.n, param.d}, "BSND", param.outDataType, ge::FORMAT_ND);
  }
  if (param.attenMaskType == AttenMaskShapeType::B_N_1_S) {
    attenMask = Tensor("attenMask", {param.b, param.n, 1, param.s}, "B_N_1_S", ge::DataType::DT_BOOL, ge::FORMAT_ND);
  }else if(param.attenMaskType == AttenMaskShapeType::B_1_S){
    attenMask = Tensor("attenMask", {param.b,  1, param.s}, "B_1_S", ge::DataType::DT_BOOL, ge::FORMAT_ND);
  }

  if (param.actualSeqLength.size() != 0) {
    actualSeqLengths = Tensor("actualSeqLengths", {param.b}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
  }
  if (param.actualSeqLengthKV.size() != 0) {
    actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {param.b}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
  }

  if (param.quantType == QuantShapeType::ALL_1) {
    deqScale1 = Tensor("deqScale1", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    quantScale1 = Tensor("quantScale1", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    deqScale2 = Tensor("deqScale2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    quantScale2 = Tensor("quantScale2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    quantOffset2 = Tensor("quantOffset2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
  } else if (param.quantType == QuantShapeType::PER_1) {
    deqScale1 = Tensor("deqScale1", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    quantScale1 = Tensor("quantScale1", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    deqScale2 = Tensor("deqScale2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
  } else if (param.quantType == QuantShapeType::POST_1) {
    quantScale2 = Tensor("quantScale2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    quantOffset2 = Tensor("quantOffset2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
  }
  return true;
}

bool PfaCase::InitOpInfo() {
  bool rst = promptCtx.SetOpName("PromptFlashAttention");
  rst = rst && promptCtx.SetDeterministic(false);
  rst = rst && promptCtx.SetInputs({&query, &key, &value, &pseShift, &attenMask, &actualSeqLengths, &actualSeqLengthsKV,
                                    &deqScale1, &quantScale1, &deqScale2, &quantScale2, &quantOffset2});
  rst = rst && promptCtx.SetOutputs({&attentionOut});
  rst = rst && promptCtx.SetTilingMaxDataSize(4096);
  rst = rst && promptCtx.SetAttrs({{"num_heads", param.numHeads},
                                   {"scale_value", param.scaleValue},
                                   {"pre_tokens", param.preTokens},
                                   {"next_tokens", param.nextTokens},
                                   {"input_layout", param.layout},
                                   {"num_key_value_heads", param.kvNumHeads},
                                   {"block_size", param.blockSize},
                                   {"sparse_mode", param.sparseMode},
                                   {"inner_precise", param.innerPrecise}});
  rst = rst && promptCtx.SetKernelRunCbf(RunPromptFlashAttention);
  rst = rst && prompt.SetContext(&promptCtx);
  return rst;
}

bool PfaCase::InitCurrentCasePtr() {
  Case::currentCasePtr = this;
  return true;
}

bool PfaCase::Run() {
  if (!enable) {
    return true;
  }
  if (!prompt.ProcessTiling(name)) {
    return false;
  }
  if (!prompt.ProcessKernel(name)) {
    return false;
  }
  return true;
}

PfaCase::PfaCase(const char* name, bool enable, const char* dbgInfo, OpInfo prompt, Param param)
    : Case(name, enable, dbgInfo), prompt(std::move(prompt)), param(std::move(param)) {
  this->prompt.name = "PromptFlashAttention";
}

PfaCase::PfaCase() {
}

PfaCase::Param::Param() {
}

PfaCase::Param::Param(int64_t b, int64_t n, int64_t s, int64_t d, std::string layout, int64_t numHeads,
                      int64_t kvNumHeads, float scaleValue, int64_t blockSize, int64_t innerPrecise,
                      int64_t sparseMode, int64_t preTokens, int64_t nextTokens):
                       b(b), n(n), s(s), d(d), layout(layout), numHeads(numHeads), kvNumHeads(kvNumHeads),
                       scaleValue(scaleValue), blockSize(blockSize), innerPrecise(innerPrecise),
                       sparseMode(sparseMode), preTokens(preTokens), nextTokens(nextTokens) {
}
