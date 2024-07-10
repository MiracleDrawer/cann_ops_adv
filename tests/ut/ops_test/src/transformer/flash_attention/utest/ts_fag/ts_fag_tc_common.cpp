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
 * \file ts_fag_tc_common.cpp
 * \brief FlashAttentionScoreGrad 反向用例.
 */

#include "ts_fag.h"

TEST_F(Ts_Fag_Ascend910B3, Tc_Common_InvalidShape_001)
{
    /**
     * 用例信息
     */
    // 用例 Shape 和 Attrs 信息
    case_->param.b = 2;
    case_->param.n2 = 16;
    case_->param.g = 1;
    case_->param.s1 = 2048;
    case_->param.s2 = 2048;
    case_->param.d = 32;
    case_->param.dtype = ge::DataType::DT_FLOAT16;
    case_->param.layoutType = LayoutType::BSH;
    case_->param.scale = 1.0f;
    case_->param.keepProb = 0.9f;
    case_->param.preTokens = 65536;
    case_->param.nxtTokens = 65536;
    case_->param.innerPrecise = 0;
    case_->param.sparseMode = 0;
    case_->param.pseShapeType = PseShapeType::NONE;
    case_->param.dropMaskShapeType = DropMaskShapeType::NONE;
    case_->param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    case_->param.attenMaskShapeType = AttenMaskShapeType::NONE;
    case_->param.attenMaskDtype = ge::DataType::DT_BOOL;
    case_->param.prefixShapeType = PrefixShapeType::NONE;

    // 用例 期望信息
    case_->reverse.exp.success = false;

    // 用例 信息初始化及修正
    ASSERT_TRUE(case_->Init());
    case_->param.attenRes = Tensor("atten_res", {case_->param.b + 1, case_->param.s1, case_->param.h1},
                                   case_->param.layout.c_str(), case_->param.dtype, ge::FORMAT_ND);

    // 用例 执行
    ASSERT_EQ(case_->Run(), case_->reverse.exp.success);
}
