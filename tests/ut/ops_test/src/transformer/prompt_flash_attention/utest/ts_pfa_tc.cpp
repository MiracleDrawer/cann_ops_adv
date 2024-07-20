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
 * \file ts_pfa_tc.cpp
 * \brief PromptFlashAttention用例.
 */

#include "ts_pfa.h"
class Ts_Pfa_Ascend910B2_Case : public Ts_Pfa_WithParam_Ascend910B2 {};
class Ts_Pfa_Ascend310P3_Case : public Ts_Pfa_WithParam_Ascend310P3 {};


TEST_F(Ts_Pfa_Ascend910B2, case_empty_query) {
  PfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 40;
  cs.param.scaleValue = 1.0f;
  cs.prompt.exp.success = false;
  ASSERT_TRUE(cs.Init());
  cs.query = Tensor("query", {cs.param.b, cs.param.n, 0, cs.param.d}, "BNSD", cs.param.qDataType, ge::FORMAT_ND);
  ASSERT_EQ(cs.Run(), cs.prompt.exp.success);
}