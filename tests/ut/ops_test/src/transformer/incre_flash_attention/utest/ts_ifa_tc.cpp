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
 * \file ts_ifa_tc.cpp
 * \brief IncreFlashAttention用例.
 */

#include "ts_ifa.h"
class Ts_Ifa_Ascend910B2_Case : public Ts_Ifa_WithParam_Ascend910B2 {};
class Ts_Ifa_Ascend310P3_Case : public Ts_Ifa_WithParam_Ascend310P3 {};
TEST_P(Ts_Ifa_Ascend910B2_Case, general_case) {
  ASSERT_TRUE(case_->Init());
  ASSERT_TRUE(case_->Run());
}
TEST_P(Ts_Ifa_Ascend310P3_Case, general_case) {
  ASSERT_TRUE(case_->Init());
  ASSERT_TRUE(case_->Run());
}

const auto Tc_Ifa_General_Case =
    ::testing::Values(IfaCase("case_001", true, "dbginfo",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              IfaCase::Param(1, 4, 1024, 128, "BSH", 4, 4, 1.0f, 0, 1, {})),
                      IfaCase("case_002", true, "dbginfo",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              IfaCase::Param(1, 5, 128, 128, "BSH", 5, 5, 1.0f, 0, 1, {})),
                      IfaCase("case_003", true, "dbginfo",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              IfaCase::Param(1, 40, 128, 128, "BSH", 40, 40, 1.0f, 0, 1, {})),
                      IfaCase("case_004", true, "dbginfo",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              IfaCase::Param(13, 20, 2048, 128, "BSH", 20, 20, 1.0f, 0, 1, {})),
                      IfaCase("case_005", true, "dbginfo",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              IfaCase::Param(20, 40, 2048, 128, "BSH", 40, 40, 1.0f, 0, 1, {})),
                      IfaCase("case_006", true, "dbginfo",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              IfaCase::Param(20, 40, 2048, 128, "BSH", 40, 40, 1.0f, 0, 1,
                                            {1024, 512,  2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048,
                                            2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048})),
                      IfaCase("case_007", true, "dbginfo",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              IfaCase::Param(1, 40, 128, 128, "BNSD", 40, 40, 1.0f, 0, 1, {})),
                      IfaCase("case_008", true, "dbginfo",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              IfaCase::Param(1, 40, 1, 128, "BSH", 40, 40, 1.0f, 0, 1, {})),
                      IfaCase("case_009", true, "dbginfo",
                              OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey,
                                                                          ExpectInfo::kInvalidTilingBlockDim)),
                              IfaCase::Param(2, 40, 4096, 128, "BSND", 40, 40, 1.0f, 0, 1, {}))                       
                              );

INSTANTIATE_TEST_SUITE_P(Ifa, Ts_Ifa_Ascend910B2_Case, Tc_Ifa_General_Case);
INSTANTIATE_TEST_SUITE_P(Ifa, Ts_Ifa_Ascend310P3_Case, Tc_Ifa_General_Case);



TEST_F(Ts_Ifa_Ascend910B2, case_atten_mask) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 20;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.numHeads = 20;
  cs.incre.exp.tilingKey = 11000000000100000;  // expected tiling key
  cs.incre.exp.tilingBlockDim = 24;            // expected block dim
  cs.incre.ctr.runTiling = true;
  cs.incre.ctr.runKernel = false;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_empty_query) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 40;
  cs.param.scaleValue = 1.0f;
  cs.incre.exp.success = false;  // expected exec result
  ASSERT_TRUE(cs.Init());
  cs.query = Tensor("query", {cs.param.b, cs.param.n, 0, cs.param.d}, "BNSD", cs.param.qDataType, ge::FORMAT_ND);
  ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_empty_query_bf16) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 40;
  cs.param.scaleValue = 1.0f;
  cs.param.qDataType=ge::DT_BF16;
  cs.incre.exp.success = false;
  ASSERT_TRUE(cs.Init());
  cs.query = Tensor("query", {cs.param.b, cs.param.n, 0, cs.param.d}, "BNSD", cs.param.qDataType, ge::FORMAT_ND);
  ASSERT_EQ(cs.Run(), cs.incre.exp.success);
  }

  TEST_F(Ts_Ifa_Ascend910B2, case_empty_key) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 100;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 40;
  cs.param.scaleValue = 1.0f;
  cs.param.qDataType=ge::DataType::DT_BF16;
  cs.incre.exp.success = false;
  ASSERT_TRUE(cs.Init());
  cs.key = Tensor("key", {cs.param.b, cs.param.n, 0, cs.param.d}, "BNSD", cs.param.kvDataType, ge::FORMAT_ND);
  cs.value = Tensor("value", {cs.param.b, cs.param.n, 0, cs.param.d}, "BNSD", cs.param.kvDataType, ge::FORMAT_ND);
  ASSERT_EQ(cs.Run(), cs.incre.exp.success);
  }

  TEST_F(Ts_Ifa_Ascend910B2, case_invalid_quant_1) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.qDataType = ge::DT_INT8;
  cs.param.kvDataType = ge::DT_INT8;
  cs.param.outDataType = ge::DT_INT8;
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.quantType = QuantShapeType::ALL_1;
  cs.param.actualSeqLength = {1000};
  cs.param.numHeads = 40;
  cs.incre.exp.success = false;
  ASSERT_TRUE(cs.Init());
  ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

  TEST_F(Ts_Ifa_Ascend910B2, case_invalid_atten_mask_1) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.actualSeqLength = {1000};
  cs.param.numHeads = 40;
  cs.incre.exp.success = false;
  ASSERT_TRUE(cs.Init());
  cs.attenMask= Tensor("attenMask", {2,40,1,1000}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
  ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_invalid_hd) {
  IfaCase cs;
  cs.param.b = 4;
  cs.param.n = 11;
  cs.param.s = 2014;
  cs.param.d = 128;
  cs.param.layout = "BSH";
  cs.param.numHeads = 11;
  cs.param.scaleValue = 1.0f;
  cs.param.attenMaskType = AttenMaskShapeType::B_1_S;
  cs.incre.exp.success = false;
  ASSERT_TRUE(cs.Init());
  cs.query = Tensor("query", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
  cs.key = Tensor("key", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
  cs.value = Tensor("value", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
  cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
  ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}


TEST_F(Ts_Ifa_Ascend910B2, case_atten_mask_2) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.numHeads = 40;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_BN_greater_than_core_number) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 49;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.numHeads = 49;
  cs.param.kvNumHeads = 1;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_Ifa_Ascend910B2, case_quant_1) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.outDataType = ge::DT_INT8;
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.quantType = QuantShapeType::POST_1;
  cs.param.actualSeqLength = {1000};
  cs.param.numHeads = 40;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiQuant_1) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.kvDataType = ge::DT_INT8;
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.antiQuantType = AntiQuantShapeType::_2_N_1_D;
  cs.param.actualSeqLength = {1000};
  cs.param.numHeads = 40;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_Ifa_Ascend310P3, case_kvAntiQuant_1) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.kvDataType = ge::DT_INT8;
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.antiQuantType = AntiQuantShapeType::_2_N_1_D;
  cs.param.actualSeqLength = {1000};
  cs.param.numHeads = 40;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_Ifa_Ascend910B2, case_qunat_kvAntiQuant_1) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 40;
  cs.param.s = 1000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.kvDataType = ge::DT_INT8;
  cs.param.outDataType = ge::DT_INT8;
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.antiQuantType = AntiQuantShapeType::_2_N_1_D;
  cs.param.quantType = QuantShapeType::POST_1;
  cs.param.actualSeqLength = {1000};
  cs.param.numHeads = 40;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiQuant_bf16) {
  IfaCase cs;
  cs.param.b = 5;
  cs.param.n = 40;
  cs.param.s = 16000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.qDataType = ge::DT_BF16;
  cs.param.kvDataType = ge::DT_INT8;
  cs.param.outDataType = ge::DT_INT8;
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.antiQuantType = AntiQuantShapeType::_2_N_1_D;
  cs.param.quantType = QuantShapeType::POST_1;
  cs.param.actualSeqLength = {1000,1000,1000,1000,1000};
  cs.param.numHeads = 40;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}


TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiQuant_unflash_splitB_largeS) {
  IfaCase cs;
  cs.param.b = 96;
  cs.param.n = 11;
  cs.param.s = 8192;
  cs.param.d = 128;
  cs.param.layout = "BSH";
  cs.param.qDataType = ge::DT_BF16;
  cs.param.kvDataType = ge::DT_INT8;
  cs.param.outDataType =ge::DT_BF16;
  cs.param.attenMaskType = AttenMaskShapeType::B_1_S;
  cs.param.antiQuantType = AntiQuantShapeType::_2_N_D;
  cs.param.numHeads = 11;
  cs.param.kvNumHeads=1;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_empty_kvPadding) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 20;
  cs.param.s = 4096;
  cs.param.d = 16;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 20;
  cs.param.kvNumHeads=2;
  cs.param.scaleValue = 1.0f;
  cs.param.actualSeqLength = {1};
  cs.param.enbaleKvPaing = true;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend310P3, case_empty_kvPadding) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 20;
  cs.param.s = 4096;
  cs.param.d = 16;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 20;
  cs.param.scaleValue = 1.0f;
  cs.param.actualSeqLength = {1};
  cs.param.enbaleKvPaing = true;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_Ifa_Ascend910B2, case_kvPadding) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 20;
  cs.param.s = 4096;
  cs.param.d = 16;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 20;
  cs.param.kvNumHeads = 2;
  cs.param.scaleValue = 1.0f;
  cs.param.actualSeqLength = {1};
  cs.param.enbaleKvPaing = true;
  cs.param.kvPaddingSize = 1;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend310P3, case_kvPadding) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 20;
  cs.param.s = 4096;
  cs.param.d = 16;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 20;
  cs.param.scaleValue = 1.0f;
  cs.param.actualSeqLength = {1};
  cs.param.enbaleKvPaing = true;
  cs.param.kvPaddingSize = 1;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvPadding_no_act_sqe_len) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 20;
  cs.param.s = 4096;
  cs.param.d = 16;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 20;
  cs.param.kvNumHeads = 2;
  cs.param.scaleValue = 1.0f;
  cs.param.enbaleKvPaing = true;
  cs.param.kvPaddingSize = 1;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiquant_bf16_quant_scale2_type_3) {
  IfaCase cs;
  cs.param.b = 5;
  cs.param.n = 40;
  cs.param.s = 16000;
  cs.param.d = 128;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 40;
  cs.param.kvNumHeads = 1;
  cs.param.scaleValue = 1.0f;
  cs.param.qDataType = ge::DT_BF16;
  cs.param.kvDataType = ge::DT_INT8;
  cs.param.outDataType = ge::DT_INT8;
  cs.param.actualSeqLength = {1};
  cs.param.quantType = QuantShapeType::POST_1;
  cs.param.antiQuantType = AntiQuantShapeType::_2_H;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}
TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiquant_workspace_opt_unflashSplitB_largeS) {
  IfaCase cs;
  cs.param.b = 96;
  cs.param.n = 11;
  cs.param.s = 8192;
  cs.param.d = 128;
  cs.param.layout = "BSH";
  cs.param.numHeads = 11;
  cs.param.kvNumHeads = 1;
  cs.param.scaleValue = 1.0f;
  cs.param.qDataType = ge::DT_BF16;
  cs.param.kvDataType = ge::DT_INT8;
  cs.param.outDataType = ge::DT_BF16;
  cs.param.actualSeqLength = {1};
  cs.param.antiQuantType = AntiQuantShapeType::_2_H;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvAntiquant_workspace_opt_unflashSplitB) {
  IfaCase cs;
  cs.param.b = 96;
  cs.param.n = 11;
  cs.param.s = 4096;
  cs.param.d = 128;
  cs.param.layout = "BSH";
  cs.param.numHeads = 11;
  cs.param.kvNumHeads = 1;
  cs.param.scaleValue = 1.0f;
  cs.param.qDataType = ge::DT_BF16;
  cs.param.kvDataType = ge::DT_INT8;
  cs.param.outDataType = ge::DT_BF16;
  cs.param.actualSeqLength = {1};
  cs.param.antiQuantType = AntiQuantShapeType::_2_H;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_invalid_hn_bsh) {
  IfaCase cs;
  cs.param.b = 4;
  cs.param.s = 2048;
  cs.param.layout = "BSH";
  cs.param.numHeads = 11;
  cs.param.scaleValue = 1.0f;
  cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
  ASSERT_TRUE(cs.Init());
  cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
  cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
  cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
  cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
  cs.incre.exp.success =false;
  ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_attenmask_fp16) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 4;
  cs.param.s = 8192;
  cs.param.d = 256;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 4;
  cs.param.actualSeqLength = {1};
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.pseShiftType = PseShiftShapeType::B_N_1_S;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend310P3, case_attenmask_fp16) {
  IfaCase cs;
  cs.param.b = 1;
  cs.param.n = 4;
  cs.param.s = 8192;
  cs.param.d = 256;
  cs.param.layout = "BNSD";
  cs.param.numHeads = 4;
  cs.param.actualSeqLength = {1};
  cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
  cs.param.pseShiftType = PseShiftShapeType::B_N_1_S;
  ASSERT_TRUE(cs.Init());
  ASSERT_TRUE(cs.Run());
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtype_float) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtype_bf16) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_BF16, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtype_bool) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_BOOL, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_BOOL, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtype_default) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT16, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT16, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_invalid_layout) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_1) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 20, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_2) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 20, 10, 11}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_bsh) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {3, 1, 10}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_bsnd) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSND";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {3, 1, 10, 11}, "BSND", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10, 11}, "BSND", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10, 11}, "BSND", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10,11}, "BSND", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_kvtensor_list) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSND";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {3, 1, 10, 11}, "BSND", cs.param.qDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {1, 2048, 10, 11}, "BSND", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {1, 2048, 10, 11}, "BSND", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {3, 1, 10,11}, "BSH", cs.param.outDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_keyshape_size) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 0}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 0}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_input_ouput_all_int8) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_input_ouput_int8_float16) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT16, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_input_ouput_all_float) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_input_ouput_int16_float) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", ge::DT_INT16, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend310P3, case_input_ouput_int8_float16) {
IfaCase cs;
cs.param.b = 1;
cs.param.n = 20;
cs.param.s = 4096;
cs.param.d = 16;
cs.param.layout = "BSND";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10, 16}, "BSND", ge::DT_INT8, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10, 16}, "BSND", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10, 16}, "BSND", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10, 16}, "BSND", ge::DT_FLOAT16, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_maskshape_size) {
IfaCase cs;
cs.param.b = 1;
cs.param.n = 40;
cs.param.s = 1000;
cs.param.d = 128;
cs.param.layout = "BNSD";
cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
cs.param.actualSeqLength = {1};
cs.param.numHeads = 40;
cs.incre.exp.success = true;
ASSERT_TRUE(cs.Init());
cs.attenMask= Tensor("attenMask", {2,40,0,1000}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_masksize_batchsize) {
IfaCase cs;
cs.param.b = 1;
cs.param.n = 40;
cs.param.s = 1000;
cs.param.d = 128;
cs.param.layout = "BNSD";
cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
cs.param.actualSeqLength = {1};
cs.param.numHeads = 40;
cs.incre.exp.success = false;
ASSERT_TRUE(cs.Init());
cs.attenMask= Tensor("attenMask", {2,40,1,1000}, "BNSD", ge::DT_FLOAT16, ge::FORMAT_ND);
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_masksize_maxcctualseq) {
IfaCase cs;
cs.param.b = 1;
cs.param.n = 40;
cs.param.s = 1000;
cs.param.d = 128;
cs.param.layout = "BNSD";
cs.param.attenMaskType = AttenMaskShapeType::B_N_1_S;
cs.param.actualSeqLength = {1};
cs.param.numHeads = 40;
cs.incre.exp.success = false;
ASSERT_TRUE(cs.Init());
cs.attenMask= Tensor("attenMask", {1,40,1,100}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_bsh) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {4, 1, 10}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_bnsd) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BNSD";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 1, 10}, "BNSD", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 1, 10}, "BNSD", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 1, 10}, "BNSD", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 1, 10}, "BNSD", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {4, 1, 1, 10}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_bh) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {4, 1}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_b1) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quantscale2_b5) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =true;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_pa_bsh) {
IfaCase cs;
cs.param.b = 3;
cs.param.n = 48;
cs.param.s = 10;
cs.param.d = 16;
cs.param.layout = "BSH";
cs.param.numHeads = 48;
cs.param.scaleValue = 1.0f;
cs.param.blockSize = 16;
cs.param.actualSeqLength = {1,1,1};
cs.param.blocktable = {5, 5};
ASSERT_TRUE(cs.Init());
cs.key = Tensor("key", {9, 4, 768}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {9, 4, 768}, "BSH", cs.param.qDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_pa_bnsd) {
IfaCase cs;
cs.param.b = 3;
cs.param.n = 48;
cs.param.s = 10;
cs.param.d = 16;
cs.param.layout = "BNSD";
cs.param.numHeads = 48;
cs.param.scaleValue = 1.0f;
cs.param.blockSize = 16;
cs.param.actualSeqLength = {1,1,1};
cs.param.blocktable = {5, 5};
ASSERT_TRUE(cs.Init());
cs.key = Tensor("key", {4, 1, 1, 10}, "BNSD", cs.param.qDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 1, 1, 10}, "BNSD", cs.param.qDataType, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quant_offset2_scale2) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.quantOffset2 = Tensor("quantOffset2", {2,2,2,2,2}, "2", ge::DT_FLOAT16, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_quant_offset2) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.quantOffset2 = Tensor("quantOffset2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =true;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_exist) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantOffset = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_nullptr) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantOffset = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_inputqtype) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantScale = Tensor("antiquantScale", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantOffset = Tensor("antiquantOffset", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_dim) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantScale = Tensor("antiquantScale", {2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantOffset = Tensor("antiquantOffset", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantScale_dim_bnsd) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantScale = Tensor("antiquantScale", {2,2,2,2}, "4", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantOffset = Tensor("quantScale2", {2,2,2,2}, "4", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_dim_bh) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantScale = Tensor("antiquantScale", {2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantOffset = Tensor("quantScale2", {2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}

TEST_F(Ts_Ifa_Ascend910B2, case_antiquantscale_antiquantoffset) {
IfaCase cs;
cs.param.b = 4;
cs.param.s = 2048;
cs.param.layout = "BSH";
cs.param.numHeads = 10;
cs.param.scaleValue = 1.0f;
cs.param.attenMaskType= AttenMaskShapeType::B_1_S;
ASSERT_TRUE(cs.Init());
cs.query=Tensor("query", {4, 1, 10}, "BSH", cs.param.kvDataType, ge::FORMAT_ND);
cs.key = Tensor("key", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.value = Tensor("value", {4, 2048, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.attentionOut = Tensor("attentionOut", {4, 1, 10}, "BSH", ge::DT_INT8, ge::FORMAT_ND);
cs.quantScale2 = Tensor("quantScale2", {2,2,2,2,2}, "2", ge::DT_FLOAT, ge::FORMAT_ND);
cs.antiquantScale = Tensor("antiquantScale", {2}, "1", cs.param.kvDataType, ge::FORMAT_ND);
cs.antiquantOffset = Tensor("antiquantOffset", {2}, "1", ge::DT_FLOAT, ge::FORMAT_ND);
cs.incre.exp.success =false;
ASSERT_EQ(cs.Run(), cs.incre.exp.success);
}