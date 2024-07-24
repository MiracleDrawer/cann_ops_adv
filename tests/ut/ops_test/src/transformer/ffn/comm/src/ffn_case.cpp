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
 * \file ffn_case.cpp
 * \brief FFN 测试用例.
 */

#include "ffn_case.h"
#include <utility>
#include <tikicpulib.h>
#include <register/op_impl_registry.h>
#include <graph/utils/type_utils.h>
#include "tests/utils/log.h"
#include "tests/utils/io.h"
#include "tests/utils/platform.h"
#include "tiling/ffn/tiling_data.h"
#include "tiling/tiling_templates_registry.h"

using Case = ops::adv::tests::utils::Case;
using FFNCase = ops::adv::tests::ffn::FFNCase;
using ops::adv::tests::utils::ReadFile;
using ops::adv::tests::utils::WriteFile;

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT 参数所控制的
 * Kernel 入口一致.
 */

#define FFN_KERNEL_PARAM                                                                                               \
    (__gm__ uint8_t * x, __gm__ uint8_t * weight1, __gm__ uint8_t * weight2, __gm__ uint8_t * expertTokens,            \
     __gm__ uint8_t * bias1, __gm__ uint8_t * bias2, __gm__ uint8_t * scale, __gm__ uint8_t * offset,                  \
     __gm__ uint8_t * deqScale1, __gm__ uint8_t * deqScale2, __gm__ uint8_t * antiquant_scale1,                        \
     __gm__ uint8_t * antiquant_scale2, __gm__ uint8_t * antiquant_offset1, __gm__ uint8_t * antiquant_offset2,        \
     __gm__ uint8_t * y, __gm__ uint8_t * workSpace, __gm__ uint8_t * tiling)

typedef void (*FFNKernelFunc) FFN_KERNEL_PARAM;

extern "C" __global__ __aicore__ void ffn_quant_fp16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_quant_bf16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_fp16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_bf16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_a16w8_fp16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_a16w8_bf16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_a16w4_fp16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_a16w4_bf16 FFN_KERNEL_PARAM;

using namespace ops::adv::tests::ffn;
using ops::adv::tests::utils::Platform;
using ops::adv::tests::utils::Tensor;
using ops::adv::tests::utils::TensorIntf;

enum KernelParams {
    X = 0,
    WEIGHT1,
    WEIGHT2,
    EXPERT_TOKENS,
    BIAS1,
    BIAS2,
    SCALE,
    OFFSET,
    DEQ_SCALE1,
    DEQ_SCALE2,
    ANTIQUANT_SCALE1,
    ANTIQUANT_SCALE2,
    ANTIQUANT_OFFSET1,
    ANTIQUANT_OFFSET2
};

bool RunFFN(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
            std::vector<TensorIntf *> &output, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (FFNKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, 1, inputs[X]->GetDevData(), inputs[WEIGHT1]->GetDevData(), inputs[WEIGHT2]->GetDevData(),
                inputs[EXPERT_TOKENS]->GetDevData(), inputs[BIAS1]->GetDevData(), inputs[BIAS2]->GetDevData(),
                inputs[SCALE]->GetDevData(), inputs[OFFSET]->GetDevData(), inputs[DEQ_SCALE1]->GetDevData(),
                inputs[DEQ_SCALE2]->GetDevData(), inputs[ANTIQUANT_SCALE1]->GetDevData(),
                inputs[ANTIQUANT_SCALE2]->GetDevData(), inputs[ANTIQUANT_OFFSET1]->GetDevData(),
                inputs[ANTIQUANT_OFFSET2]->GetDevData(), output[0]->GetDevData(), workspace, tilingData);
    return true;
}

FFNCase::FFNCase() : FFNCase("Undefined", true, "", OpInfo(), Param(), 0)
{
}

FFNCase::FFNCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param,
                 int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), opInfo(std::move(opInfo)), param(std::move(param))
{
    this->opInfo.name = "FFN";
}

bool FFNCase::InitParam()
{
    if (param.expertTokensData.size() > 0) {
        size_t dataSize = param.expertTokensData.size() * sizeof(int64_t);
        uint8_t *addr = param.tensors["expertTokens"].AllocDevData(0, dataSize);
        if (addr == nullptr) {
            LOG_ERR("Tensor(%s, %ld) AllocDevData Failed.", param.tensors["expertTokens"].Name().c_str(), dataSize);
            return false;
        }
        std::string fileName = this->name + "_expertToken.bin";
        if (!WriteFile(fileName, param.expertTokensData.data(), dataSize)) {
            LOG_ERR("Write expertToken data to file[%s] failed", fileName.c_str());
            return false;
        }
        if (!ReadFile(fileName, dataSize, addr, dataSize)) {
            LOG_ERR("Read expertToken data[%s] to tensor failed", fileName.c_str());
            return false;
        }
    }
    return true;
}

bool FFNCase::InitOpInfo()
{
    auto *ffnKernelFunc = (void *)ffn_quant_fp16;;
    if (param.tensors["x"].GetDataType() == param.tensors["weight1"].GetDataType()) {
        if (param.tensors["weight1"].GetDataType() == ge::DataType::DT_INT8) { // 量化
            isQuant = true;
            if (param.tensors["y"].GetDataType() == ge::DataType::DT_FLOAT16) {
                ffnKernelFunc = (void *)ffn_quant_fp16;
            } else {
                ffnKernelFunc = (void *)ffn_quant_bf16;
            }
        } else { // 非量化
            if (param.tensors["weight1"].GetDataType() == ge::DataType::DT_FLOAT16) {
                ffnKernelFunc = (void *)ffn_fp16;
            } else {
                ffnKernelFunc = (void *)ffn_bf16;
            }
        }
    } else { // 伪量化
        if (param.tensors["weight1"].GetDataType() == ge::DataType::DT_INT8) {
            if (param.tensors["x"].GetDataType() == ge::DataType::DT_FLOAT16) {
                ffnKernelFunc = (void *)ffn_a16w8_fp16;
            } else {
                ffnKernelFunc = (void *)ffn_a16w8_bf16;
            }
        } else {
            if (param.tensors["x"].GetDataType() == ge::DataType::DT_FLOAT16) {
                ffnKernelFunc = (void *)ffn_a16w4_fp16;
            } else {
                ffnKernelFunc = (void *)ffn_a16w4_bf16;
            }
        }
    }

    bool rst = ctx.SetOpName(opInfo.name.c_str());
    rst = rst && ctx.SetDeterministic(opInfo.ctr.deterministic);
    rst = rst && ctx.SetInputs({&param.tensors["x"], &param.tensors["weight1"], &param.tensors["weight2"],
                                &param.tensors["expertTokens"], &param.tensors["bias1"], &param.tensors["bias2"],
                                &param.tensors["scale"], &param.tensors["offset"], &param.tensors["deqScale1"],
                                &param.tensors["deqScale2"], &param.tensors["antiquant_scale1"],
                                &param.tensors["antiquant_scale2"], &param.tensors["antiquant_offset1"],
                                &param.tensors["antiquant_offset2"]});
    rst = rst && ctx.SetOutputs({&param.tensors["y"]});
    rst = rst && ctx.SetAttrs({{"activation", param.activation},
                               {"inner_precise", param.innerPrecise},
                               {"output_dtype", param.outputDtype},
                               {"tokens_index_flag", param.tokensIndexFlag}});
    rst = rst && ctx.SetKernelRunCbf(RunFFN);
    rst = rst && ctx.SetKernelMainFunc((void *)ffnKernelFunc);
    rst = rst && opInfo.SetContext(&ctx);
    return rst;
}

bool FFNCase::Run()
{
    if (!enable) {
        return true;
    }
    if (!opInfo.ProcessTiling(name)) {
        return false;
    }
    FFNTilingData *ffnTiling = (FFNTilingData *)ctx.GetTilingData();
    if (ffnTiling == nullptr) {
        LOG_ERR("Tiling failed!");
        return false;
    }
    if (isQuant || ctx.GetTilingKey() == 15) { // 15: 伪量化msd模板
        if (ffnTiling->mm1TilingData.baseN * ffnTiling->mm1TilingData.baseK >
            16384) {                             // 16384: int8场景右矩阵占l0b的1/4
            ffnTiling->mm1TilingData.baseN /= 2; // 2: 将l0b占用大小减半
        }
        if (ffnTiling->mm2TilingData.baseN * ffnTiling->mm2TilingData.baseK >
            16384) {                             // 16384: int8场景右矩阵占l0b的1/4
            ffnTiling->mm2TilingData.baseN /= 2; // 2: 将l0b占用大小减半
        }
    }
    if (!opInfo.ProcessKernel(name)) {
        return false;
    }
    return true;
}

bool FFNCase::InitCurrentCasePtr()
{
    Case::currentCasePtr = this;
    return true;
}

Tensor ops::adv::tests::ffn::GenTensor(const char *name, const std::initializer_list<int64_t> &shape,
                                       ge::DataType dType, ge::Format format)
{
    return Tensor(name, shape, "", dType, format);
}