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
 * \file fia_case.cpp
 * \brief FusedInferAttentionScore 测试用例.
 */
#include "fia_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/fia/tiling_data.h"
#include "tiling/tiling_base.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */
#define FIA_KERNEL_PARAM                                                                                               \
    (__gm__ uint8_t * query, __gm__ uint8_t * key, __gm__ uint8_t * value, __gm__ uint8_t * pse_shift,                 \
     __gm__ uint8_t * attenMask, __gm__ uint8_t * actualSeqLengths, __gm__ uint8_t * actualSeqLengthsKV,               \
     __gm__ uint8_t * deq_scale1, __gm__ uint8_t * quant_scale1, __gm__ uint8_t * deq_scale2,                          \
     __gm__ uint8_t * quant_scale2, __gm__ uint8_t * quant_offset2, __gm__ uint8_t * antiquantScale,                   \
     __gm__ uint8_t * antiquantOffset, __gm__ uint8_t * blocktable, __gm__ uint8_t * queryPaddingSize,                 \
     __gm__ uint8_t * kvPaddingSize, __gm__ uint8_t * keyAntiquantScale, __gm__ uint8_t * keyAntiquantOffset,          \
     __gm__ uint8_t * valueAntiquantScale, __gm__ uint8_t * valueAntiquantOffset, __gm__ uint8_t * keySharedPrefix,    \
     __gm__ uint8_t * valueSharedPrefix, __gm__ uint8_t * actualSharedPrefixLen, __gm__ uint8_t * attentionOut,        \
     __gm__ uint8_t * softmaxLse, __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

typedef void (*FiaKernelFunc) FIA_KERNEL_PARAM;

extern "C" __global__ __aicore__ void fused_infer_attention_score FIA_KERNEL_PARAM;

using namespace ops::adv::tests::fia;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunFusedInferAttentionScore(void* func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf*>& inputs,
                                 std::vector<TensorIntf*>& outputs, uint8_t* workspace, uint8_t* tilingData) {
  // Kernel 运行
  auto kernelFunc = (FiaKernelFunc)func;
  ICPU_SET_TILING_KEY(tilingKey);
  ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
              inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
              inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
              inputs[11]->GetDevData(), inputs[12]->GetDevData(), inputs[13]->GetDevData(), inputs[14]->GetDevData(),
              inputs[15]->GetDevData(), inputs[16]->GetDevData(), inputs[17]->GetDevData(), inputs[18]->GetDevData(),
              inputs[19]->GetDevData(), inputs[20]->GetDevData(), inputs[21]->GetDevData(), inputs[22]->GetDevData(),
              inputs[23]->GetDevData(), outputs[0]->GetDevData(), outputs[1]->GetDevData(), workspace, tilingData);
  return true;
}

bool FiaCase::InitParam() {
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
  return true;
}

bool FiaCase::InitOpInfo() {
  bool rst = fiaCtx.SetOpName("FusedInferAttentionScore");
  rst = rst && fiaCtx.SetDeterministic(false);
  rst = rst && fiaCtx.SetInputs({&query, &key, &value, &pseShift, &attenMask, &actualSeqLengths, &actualSeqLengthsKV,
                                 &deqScale1, &quantScale1, &deqScale2, &quantScale2, &quantOffset2, &antiquantScale,
                                 &antiquantOffset, &blocktable, &queryPaddinSize, &kvPaddingSize, &keyAntiquantScale,
                                 &keyAntiquantOffset, &valueAntiquantScale, &valueAntiquantOffset, &keySharedPrefix,
                                 &valueSharedPrefix, &actualSharedPrefixLen});

  rst = rst && fiaCtx.SetOutputs({&attentionOut, &softmaxLse});
  rst = rst && fiaCtx.SetAttrs({{"num_head", param.numHeads},
                                {"scale_value", param.scaleValue},
                                {"pre_tokens", param.pre_tokens},
                                {"next_tokens", param.next_tokens},
                                {"input_layout", param.layout},
                                {"num_key_value_heads", param.kvNumHeads},
                                {"sparse_mode", param.sparse_mode},
                                {"inner_precise", param.innerPrecise},
                                {"antiquant_mode", param.antiquant_mode},
                                {"softmax_lse_flag", param.softmax_lse_flag},
                                {"key_antiquant_mode", param.key_antiquant_mode},
                                {"value_antiquant_mode", param.value_antiquant_mode}});
  rst = rst && fiaCtx.SetKernelRunCbf(RunFusedInferAttentionScore);
  rst = rst && fiaCtx.SetKernelMainFunc((void *)fused_infer_attention_score);
  rst = rst && fia.SetContext(&fiaCtx);
  return rst;
}

bool FiaCase::InitCurrentCasePtr() {
  Case::currentCasePtr = this;
  return true;
}

bool FiaCase::Run() {
  if (!enable) {
    return true;
  }
  if (!fia.ProcessTiling(name)) {
    return false;
  }
  if (!fia.ProcessKernel(name)) {
    return false;
  }
  return true;
}

FiaCase::FiaCase(const char* name, bool enable, const char* dbgInfo, OpInfo fia, Param param)
    : Case(name, enable, dbgInfo), fia(std::move(fia)), param(std::move(param)) {
  this->fia.name = "FusedInferAttentionScore";
}

FiaCase::FiaCase() {
}
FiaCase::Param::Param() {
}

