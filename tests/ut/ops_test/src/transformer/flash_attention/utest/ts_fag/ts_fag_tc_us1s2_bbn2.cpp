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
 * \file ts_fag_tc_us1s2_bbn2.cpp
 * \brief FlashAttentionScoreGrad 算子 Us1s2Bbn2 模板 UTest 用例.
 */

#include "ts_fag.h"

TEST_F(Ts_Fag_Ascend910B2, Tc_Us1s2Bbn2_IllegalAttenMaskShape_001)
{
    FagCase cs("Tc_Us1s2Bbn2_IllegalAttenMaskShape_001", true,         /* CaseName, Enable */
               "not support attenmask shape type",                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                       0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                       PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                       AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init());
    cs.param.attenMask = Tensor("atten_mask", {10000, 10000, cs.param.s1, cs.param.s2}, "X_X_S1_S2",
                                cs.param.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.reverse.exp.success);
}

TEST_F(Ts_Fag_Ascend910B2, Tc_Us1s2Bbn2_IllegalAttenMaskShape_002)
{
    FagCase cs("Tc_Us1s2Bbn2_IllegalAttenMaskShape_002", true,         /* CaseName, Enable */
               "not support attenmask shape type",                     /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                       0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                       PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                       AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init());
    cs.param.attenMask = Tensor("atten_mask", {cs.param.b}, "B", cs.param.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.reverse.exp.success);
}

TEST_F(Ts_Fag_Ascend910B2, Tc_Us1s2Bbn2_IllegalPseShape_001)
{
    FagCase cs("Tc_Us1s2Bbn2_IllegalPseShape_001", true,               /* CaseName, Enable */
               "not support pseShape",                                 /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                       0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                       PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_UINT8,                         /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init());
    cs.param.pse = Tensor("pse", {10000, 10000, cs.param.s1, cs.param.s2}, "X_X_S1_S2", cs.param.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.reverse.exp.success);
}

TEST_F(Ts_Fag_Ascend910B2, Tc_Us1s2Bbn2_IllegalPseShape_002)
{
    FagCase cs("Tc_Us1s2Bbn2_IllegalPseShape_002", true,               /* CaseName, Enable */
               "not support pseShape",                                 /* DebugInfo */
               OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfo(false,                                /* ExpectSuccess */
                                 ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                       0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                       PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_UINT8,                         /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init());
    cs.param.pse = Tensor("pse", {cs.param.b}, "B", cs.param.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.reverse.exp.success);
}

class Ts_Fag_Ascend910B2_Us1s2Bbn2 : public Ts_Fag_WithParam_Ascend910B2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->reverse.exp.success);
}

const auto Tc_Fag_Us1s2Bbn2_BatchCase = ::testing::Values(

    FagCase("Fag_Us1s2Bbn2_Case_000", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100020134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_001", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100002134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSND,        /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_002", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 1024, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 0, 0,                           /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_003", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 1023, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 0, 0,                           /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_004", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_005", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_006", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::_1_1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_007", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_008", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_FLOAT,                         /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_009", true,                         /* CaseName, Enable */
            "AttentionMask Dtype illegal",                          /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_INT32,                         /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_010", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_011", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 3,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_012", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 5,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_013", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_014", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::_1_N1_S1_S2,                      /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_015", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_ALIBI_S1_S2,                 /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_UINT8,                         /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_016", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::_1_N1_ALIBI_S1_S2,                /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_UINT8,                         /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_017", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(24, 10, 1, 2304, 2304, 64,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::_1_N1_S1_S2,                      /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_UINT8,                         /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_018", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011110020134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 9216, 77, 64,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 65504, 65504,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_019", true,                         /* CaseName, Enable */
            "compress atten_mask mode not support s1 s2 2048 2048", /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 3,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_020", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100020134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 64,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_021", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100020134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 64,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 512, -256,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_022", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100020134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 64,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, -128, 256,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_023", true,                         /* CaseName, Enable */
            "sparse pre_token next_token not support",              /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, -512, -256,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_024", true,                         /* CaseName, Enable */
            "sparse pre_token next_token not support",              /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, -512, 256,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_025", true,                         /* CaseName, Enable */
            "sparse pre_token next_token not support",              /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, -1025, 1300,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_026", true,                         /* CaseName, Enable */
            "sparse pre_token next_token not support",              /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 512, -680,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_027", true,                         /* CaseName, Enable */
            "sparse pre_token next_token not support",              /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                    0.08838f, 0.8f, 1200, -1025,                    /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_028", false,                        /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011000000134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(6, 10, 10, 2048, 1024, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 6,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::B,                             /* PrefixShapeType */
                    {332, 482, 196, 245, 177, 71},                  /* PrefixTensorData */
                    {},                                             /* ActualSeqQTensorData */
                    {}),                                            /* ActualSeqKVTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_029", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 64,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 512, -256,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_030", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000011100021134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 9216, 77, 64,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 65504, 65504,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
     FagCase("Fag_Us1s2Bbn2_Case_031", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(true,                                 /* ExpectSuccess */
                              10000000001100011134UL,               /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(128, 18, 1, 34, 34, 16,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::SBH,     /* Dtype, Layout */
                    0.08838f, 1, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                        /* PseShapeType */
                    DropMaskShapeType::NONE,                       /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_032", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                 /* ExpectSuccess */
                                ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                                ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 40, 1, 256, 128, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSND,        /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 8,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2, Tc_Fag_Us1s2Bbn2_BatchCase);

class Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape : public Ts_Fag_Ascend910B2_Us1s2Bbn2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape, Tc_AttenmaskErr)
{
    ASSERT_TRUE(case_->Init());
    case_->param.attenMask =
        Tensor("attenMask", {case_->param.s1, 1}, "S1_1(Invalid)", case_->param.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->reverse.exp.success);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape, Tc_AttenmaskErrDim)
{
    ASSERT_TRUE(case_->Init());
    case_->param.attenMask = Tensor("attenMask", {1, case_->param.s1, 1, case_->param.s2}, "1_S1_1_S2(Invalid)",
                                    case_->param.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->reverse.exp.success);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape, Tc_AttenmaskErrDimNum)
{
    ASSERT_TRUE(case_->Init());
    case_->param.attenMask =
        Tensor("attenMask", {1, case_->param.s1, 1}, "1_S1_1(Invalid)", case_->param.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->reverse.exp.success);
}

const auto Tc_Fag_Us1s2Bbn2_InvalidShape_BatchCase = ::testing::Values(

    FagCase("Fag_Us1s2Bbn2_Case_000", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape, Tc_Fag_Us1s2Bbn2_InvalidShape_BatchCase);

class Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidPrefixCompressShape : public Ts_Fag_Ascend910B2_Us1s2Bbn2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidPrefixCompressShape, Tc_AttenmaskErrShapeForPrefixCompress)
{
       ASSERT_TRUE(case_->Init());
       case_->param.attenMask =
           Tensor("attenMask", {2048, 2048}, "not_3072_2048(Invalid)", case_->param.dtype, ge::FORMAT_ND);
       ASSERT_EQ(case_->Run(), case_->reverse.exp.success);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidPrefixCompressShape, Tc_PrefixErrShapeForPrefixCompress)
{
       ASSERT_TRUE(case_->Init());
       case_->param.prefix = Tensor("prefix", {110, 110, 110, 110, 110, 110, 110}, "prefixN_gt_B(Invalid)",
                                    case_->param.dtype, ge::FORMAT_ND);
       ASSERT_EQ(case_->Run(), case_->reverse.exp.success);
}

const auto Tc_Fag_Us1s2Bbn2_InvalidPrefixCompressShape_BatchCase = ::testing::Values(

    FagCase("Fag_Us1s2Bbn2_Case_001", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfo(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfo(false,                                /* ExpectSuccess */
                              ExpectInfo::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfo::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(6, 10, 10, 2048, 1024, 128,                     /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 6,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::B,                             /* PrefixShapeType */
                    {332, 482, 196, 245, 177, 71},                  /* PrefixTensorData */
                    {},                                             /* ActualSeqQTensorData */
                    {}),                                            /* ActualSeqKVTensorData */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidPrefixCompressShape,
                         Tc_Fag_Us1s2Bbn2_InvalidPrefixCompressShape_BatchCase);
