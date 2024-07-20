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
 * \file ifa_case.cpp
 * \brief IncreFlashAttentionScore 测试用例.
 */
#include "ifa_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/ifa/tiling_data.h"
#include "tiling/tiling_base.h"

typedef void (*IfaKernelFunc)(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                              __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths,
                              __gm__ uint8_t* deqScale1, __gm__ uint8_t* quantScale1, __gm__ uint8_t* deqScale2,
                              __gm__ uint8_t* quantScale2, __gm__ uint8_t* quantOffset2, __gm__ uint8_t* antiquantScale,
                              __gm__ uint8_t* antiquantOffset, __gm__ uint8_t* blocktable,
                              __gm__ uint8_t* kvPaddingSize, __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                              __gm__ uint8_t* tiling);

using namespace ops::adv::tests::ifa;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunIncreFlashAttention(void* func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf*>& inputs,
                            std::vector<TensorIntf*>& outputs, uint8_t* workspace, uint8_t* tilingData) {
  // Kernel 运行
  auto kernelFunc = (IfaKernelFunc)func;
  ICPU_SET_TILING_KEY(tilingKey);
  ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
              inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
              inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
              inputs[11]->GetDevData(), inputs[12]->GetDevData(), inputs[13]->GetDevData(), inputs[14]->GetDevData(),
              outputs[0]->GetDevData(), workspace, tilingData);
  return true;
}

bool IfaCase::InitParam() {
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

  if (param.pseShiftType == PseShiftShapeType::B_N_1_S) {
    pseShift = Tensor("pseShift", {param.b, param.n, 1, param.s}, "B_N_1_S", param.qDataType, ge::FORMAT_ND);
  } else if (param.pseShiftType == PseShiftShapeType::_1_N_1_S) {
    pseShift = Tensor("pseShift", {1, param.n, 1, param.s}, "_1_N_1_S", param.qDataType, ge::FORMAT_ND);
  }

  if (param.actualSeqLength.size() == 1) {
    actualSeqLengths = Tensor("actualSeqLengths", {1}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND);
  } else if (param.actualSeqLength.size() != 0) {
    actualSeqLengths = Tensor("actualSeqLengths", {param.b}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
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

  if (param.antiQuantType == AntiQuantShapeType::_2_H) {
    antiquantScale = Tensor("antiquantScale", {2, kvH}, "2_H", param.qDataType, ge::FORMAT_ND);
    antiquantOffset = Tensor("antiquantOffset", {2, kvH}, "2_H", param.qDataType, ge::FORMAT_ND);
  } else if (param.antiQuantType == AntiQuantShapeType::_2_N_D) {
    antiquantScale = Tensor("antiquantScale", {2, kvNum, param.d}, "2_N_D", param.qDataType, ge::FORMAT_ND);
    antiquantOffset = Tensor("antiquantOffset", {2, kvNum, param.d}, "2_N_D", param.qDataType, ge::FORMAT_ND);
  } else if (param.antiQuantType == AntiQuantShapeType::_2_N_1_D) {
    antiquantScale =
        Tensor("antiquantScale", {2, kvNum, 1, param.d}, "2_N_1_D", param.qDataType, ge::FORMAT_ND);
    antiquantOffset =
        Tensor("antiquantOffset", {2, kvNum, 1, param.d}, "2_N_1_D",param.qDataType, ge::FORMAT_ND);
  }

  if (param.blocktable.size() == 2) {
    blocktable =
        Tensor("blocktable", {param.blocktable[0], param.blocktable[1]}, "A_B", ge::DataType::DT_INT32, ge::FORMAT_ND);
  }

  if(param.enbaleKvPaing){
    kvPaddingSize=Tensor("kvPaddingSize", {param.kvPaddingSize}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND);
  }
  return true;
}

bool IfaCase::InitOpInfo() {
  std::string kernelSoRealPath = "src/transformer/incre_flash_attention/libAsdcOpTestUt_Ifa_Kernel.so";
  bool rst = increCtx.SetOpName("IncreFlashAttention");
  rst = rst && increCtx.SetDeterministic(false);
  rst = rst && increCtx.SetInputs({&query, &key, &value, &pseShift, &attenMask, &actualSeqLengths, &deqScale1,
                                   &quantScale1, &deqScale2, &quantScale2, &quantOffset2, &antiquantScale,
                                   &antiquantOffset, &blocktable, &kvPaddingSize});
  rst = rst && increCtx.SetOutputs({&attentionOut});
  rst = rst && increCtx.SetAttrs({{"num_head", param.numHeads},
                                   {"scale_value", param.scaleValue},
                                   {"input_layout", param.layout},
                                   {"num_key_value_heads", param.kvNumHeads},
                                   {"block_size", param.blockSize},
                                   {"inner_precise", param.innerPrecise}});
  rst = rst && increCtx.SetKernelSoRelPath(kernelSoRealPath.c_str());
  rst = rst && increCtx.SetKernelFuncName("incre_flash_attention");
  rst = rst && increCtx.SetKernelRunCbf(RunIncreFlashAttention);
  rst = rst && incre.SetContext(&increCtx);
  return rst;
}

bool IfaCase::InitCurrentCasePtr() {
  Case::currentCasePtr = this;
  return true;
}

bool IfaCase::Run() {
  if (!enable) {
    return true;
  }
  if (!incre.ProcessTiling(name)) {
    return false;
  }
  if (!incre.ProcessKernel(name)) {
    return false;
  }
  return true;
}

IfaCase::IfaCase(const char* name, bool enable, const char* dbgInfo, OpInfo incre, Param param)
    : Case(name, enable, dbgInfo), incre(std::move(incre)), param(std::move(param)) {
  this->incre.name = "IncreFlashAttention";
}

IfaCase::IfaCase(){}
IfaCase::Param::Param() {}
IfaCase::Param::Param(int64_t b, int64_t n, int64_t s, int64_t d, std::string layout, int64_t numHeads,
                      int64_t kvNumHeads, float scaleValue, int64_t blockSize, int64_t innerPrecise,
                      std::vector<int64_t> actualSeqLength):
                       b(b), n(n), s(s), d(d), layout(layout), numHeads(numHeads), kvNumHeads(kvNumHeads),
                       scaleValue(scaleValue), blockSize(blockSize), innerPrecise(innerPrecise),
                       actualSeqLength(actualSeqLength){}
