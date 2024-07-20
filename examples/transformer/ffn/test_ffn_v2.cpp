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
 * \file test_ffn_v2.cpp
 * \brief
 */

#include <string>
#include <vector>
#include <sys/stat.h>
#include "ffn_utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn.h"
#include "aclnnop/aclnn_ffn_v2.h"


using namespace ffn_example;

int main(int argc, char **argv)
{
    if (argv == nullptr || argc < 2) { // 2: exeFile and test_case
        LOG_PRINT("Number of nput parameter error, except >= 2 but got %d inputs.\n", argc);
        return 0;
    }
    std::string test_case(argv[1]);
    bool no_expert = test_case.find("no_expert") != std::string::npos;
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 如果需要修改shape值，需要同步修改../scripts/ffn_generate_data.py中 test_ffn_v2 分支下生成
    // 对应的shape值，并重新gen data，再执行
    int64_t bs = 64;
    int64_t h = 1024;
    int64_t n = 2048;
    int64_t expertNum = 4;
    std::vector<int64_t> xShape = {bs, h};
    std::vector<int64_t> weight1Shape = {expertNum, h, n};
    std::vector<int64_t> weight2Shape = {expertNum, n, h};
    std::vector<int64_t> bias1Shape = {expertNum, n};
    std::vector<int64_t> yShape = {bs, h};
    std::vector<int64_t> expertToken{16, 32, 48, 64};
    std::vector<int16_t> yData(bs * h, 0);
    if (no_expert) {
        weight1Shape = {h, n};
        weight2Shape = {n, h};
        bias1Shape = {n};
    }

    const char *activation = "relu";
    int64_t innerPrecise = 1;
    bool tokensIndexFlag = true;
    FFNParams params;
    FFNDevAddr addrs;
    if (!no_expert) {
        params.expertTokens = aclCreateIntArray(expertToken.data(), expertToken.size());
    }

    uint64_t workspaceSize = 0;

    std::string exeFile(argv[0]);
    std::string currentPath = std::string(exeFile.substr(0, exeFile.rfind('/')) + "/");
    std::string xFilePath = currentPath + "x.bin";
    ret = CreateAclTensor(xFilePath, xShape, 2, &addrs.x, aclDataType::ACL_FLOAT16, &params.x);
    CHECK_RET(ret == ACL_SUCCESS, FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    std::string wegiht1FilePath = currentPath + "weight1.bin";
    ret = CreateAclTensor(wegiht1FilePath, weight1Shape, 2, &addrs.weight1, aclDataType::ACL_FLOAT16, &params.weight1);
    CHECK_RET(ret == ACL_SUCCESS, FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    std::string weight2FilePath = currentPath + "weight2.bin";
    ret = CreateAclTensor(weight2FilePath, weight2Shape, 2, &addrs.weight2, aclDataType::ACL_FLOAT16, &params.weight2);
    CHECK_RET(ret == ACL_SUCCESS, FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    std::string bias1FilePath = currentPath + "bias1.bin";
    ret = CreateAclTensor(bias1FilePath, bias1Shape, 2, &addrs.bias1, aclDataType::ACL_FLOAT16, &params.bias1);
    CHECK_RET(ret == ACL_SUCCESS, FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    ret = CreateAclTensor(yData, yShape, &addrs.y, aclDataType::ACL_FLOAT16, &params.y);
    CHECK_RET(ret == ACL_SUCCESS, FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;
    // 调用aclnnFFNV2第一段接口
    ret = aclnnFFNV2GetWorkspaceSize(params.x, params.weight1, params.weight2, params.expertTokens, params.bias1,
                                     params.bias2, params.scale, params.offset, params.deqScale1, params.deqScale2,
                                     params.antiquantScale1, params.antiquantScale2, params.antiquantOffset1,
                                     params.antiquantOffset2, activation, innerPrecise, tokensIndexFlag, params.y,
                                     &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&addrs.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(params, addrs, deviceId, &context, &stream); return ret);
    }

    // 调用aclnnFFNV2第二段接口
    ret = aclnnFFNV2(addrs.workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV2 failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(params, addrs, deviceId, &context, &stream); return ret);

    std::string outFile = test_case + ".bin";
    SaveOutResult<float>(outFile, yShape, &addrs.y);

    // 6. 释放aclTensor，需要根据具体API的接口定义修改; 释放device资源
    FreeResource(params, addrs, deviceId, &context, &stream);

    return 0;
}
