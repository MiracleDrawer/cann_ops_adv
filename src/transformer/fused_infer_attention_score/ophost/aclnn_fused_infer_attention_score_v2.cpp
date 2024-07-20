/**
 * Copyright (c) 2023 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include <cstring>
#include "graph/types.h"
#include "aclnn_fused_infer_attention_score_v2.h"
 
#ifdef __cplusplus
extern "C" {
#endif
 
namespace {
extern aclnnStatus aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1,
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset,
    const aclTensor *blockTable,
    const aclTensor *queryPaddingSize,
    const aclTensor *kvPaddingSize,
    const aclTensor *keyAntiquantScale,
    const aclTensor *keyAntiquantOffset,
    const aclTensor *valueAntiquantScale,
    const aclTensor *valueAntiquantOffset,
    const aclTensor *keySharedPrefix,
    const aclTensor *valueSharedPrefix,
    const aclIntArray *actualSharedPrefixLen,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t antiquantMode,
    bool softmaxLseFlag,
    int64_t keyAntiquantMode,
    int64_t valueAntiquantMode,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
 
extern aclnnStatus aclnnInnerFusedInferAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
 
aclnnStatus aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *deqScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional,
    const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional,
    const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t antiquantMode,
    bool softmaxLseFlag,
    int64_t keyAntiquantMode,
    int64_t valueAntiquantMode,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {
        const aclTensor *placeHolder = nullptr;
        const aclTensor *tempTensor = nullptr;
        if (softmaxLseFlag == false) {
            std::vector<int64_t> shape = {2, 2, 1, 1};
            int64_t addr = 0xff;
            tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                                        shape.data(), 0, ACL_FORMAT_ND,
                                                        shape.data(), shape.size(), (void*)&addr);
            placeHolder = (softmaxLse == nullptr) ? tempTensor : softmaxLse;
        } else {
            placeHolder = softmaxLse;
        }
        aclnnStatus ret = aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(query, key, value, pseShiftOptional, attenMaskOptional,
                                                              actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                                                              deqScale1Optional, quantScale1Optional, deqScale2Optional,
                                                              quantScale2Optional, quantOffset2Optional, antiquantScaleOptional, 
                                                              antiquantOffsetOptional, blockTableOptional, queryPaddingSizeOptional, kvPaddingSizeOptional,
                                                              keyAntiquantScaleOptional, keyAntiquantOffsetOptional, valueAntiquantScaleOptional, valueAntiquantOffsetOptional,
                                                              keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenOptional, numHeads,
                                                              scaleValue, preTokens, nextTokens, inputLayout, 
                                                              numKeyValueHeads, sparseMode, innerPrecise, 
                                                              blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode,
                                                              attentionOut, placeHolder, workspaceSize, executor);
        if (softmaxLseFlag == false) {
            aclDestroyTensor(tempTensor);
        }
        return ret;
    }
 
aclnnStatus aclnnFusedInferAttentionScoreV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream) {
        return aclnnInnerFusedInferAttentionScore(workspace, workspaceSize, executor, stream);
    }
 
}
 
#ifdef __cplusplus
}
#endif