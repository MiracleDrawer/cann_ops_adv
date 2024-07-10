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
 * \file ts_fas_tc_sparse_mode.cpp
 * \brief
 */

#include "ts_fas.h"
#include "tiling/fa/tiling_data.h"

namespace {
uint8_t GetSparseType(void *tilingData)
{
    auto *fasTilingData = (FlashAttentionScoreGeneralTilingData *)tilingData;
    return fasTilingData->inputParams.sparseType;
}

uint8_t GetImplMode(void *tilingData)
{
    auto *fasTilingData = (FlashAttentionScoreGeneralTilingData *)tilingData;
    return fasTilingData->inputParams.implMode;
}
} // namespace

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_001)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 2;
    cs.param.n2 = 8;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::B_N1_1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 2048;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 3;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
    ASSERT_EQ(GetImplMode(cs.forwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_002)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 10;
    cs.param.g = 1;
    cs.param.s1 = 4096;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 4096;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
    ASSERT_EQ(GetImplMode(cs.forwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_003)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_004)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 16;
    cs.param.g = 1;
    cs.param.s1 = 4096;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 2048;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_005)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 4;
    cs.param.n2 = 4;
    cs.param.g = 1;
    cs.param.s1 = 128;
    cs.param.s2 = 1024;
    cs.param.d = 125;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 500;
    cs.param.nxtTokens = 300;
    cs.param.layoutType = LayoutType::BSND;
    cs.param.innerPrecise = 1;
    cs.param.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
    ASSERT_EQ(GetImplMode(cs.forwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_006)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 4;
    cs.param.n2 = 4;
    cs.param.g = 1;
    cs.param.s1 = 128;
    cs.param.s2 = 1024;
    cs.param.d = 125;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 500;
    cs.param.nxtTokens = 300;
    cs.param.layoutType = LayoutType::BSND;
    cs.param.innerPrecise = 1;
    cs.param.sparseMode = 99;

    /**
     * 期望信息
     */
    cs.forward.exp.success = false;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.forward.exp.success);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_007)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 4;
    cs.param.n2 = 4;
    cs.param.g = 1;
    cs.param.s1 = 128;
    cs.param.s2 = 1024;
    cs.param.d = 125;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 500;
    cs.param.nxtTokens = 300;
    cs.param.layoutType = LayoutType::BSND;
    cs.param.innerPrecise = 1;
    cs.param.sparseMode = 100;

    /**
     * 期望信息
     */
    cs.forward.exp.success = false;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.forward.exp.success);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_008)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_009)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 1024;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_010)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 2048;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_011)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 2049;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_012)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 2048;
    cs.param.nxtTokens = 2049;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_013)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = -900;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_014)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = -900;
    cs.param.nxtTokens = 1024;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_015)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_016)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 0;
    cs.param.nxtTokens = 1024;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_017)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_018)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 3096;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_019)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 2048;
    cs.param.nxtTokens = 100;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_020)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 1024;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_021)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = -900;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_022)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = -900;
    cs.param.nxtTokens = 1024;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_023)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 10;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 2048;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_024)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 10;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 2048;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 2;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
    ASSERT_EQ(GetImplMode(cs.forwardCtx.GetTilingData()), 2);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_025)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_026)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::B;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::TND;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 6;
    cs.param.prefixTensorData = {2500};
    cs.param.actualSeqQLenTensorData = {2048};
    cs.param.actualSeqKVLenTensorData = {3028};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_027)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 1;
    cs.param.g = 1;
    cs.param.s1 = 317;
    cs.param.s2 = 317;
    cs.param.d = 80;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::B;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 6;
    cs.param.prefixTensorData = {0};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_028)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 4;
    cs.param.n2 = 4;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 21;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::BSND;
    cs.param.innerPrecise = 1;
    cs.param.sparseMode = 0;

    /**
     * 期望信息
     */
    cs.forward.exp.success = true;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.forward.exp.success);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_029)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 4;
    cs.param.n2 = 16;
    cs.param.g = 1;
    cs.param.s1 = 512;
    cs.param.s2 = 512;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_030)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 4;
    cs.param.n2 = 16;
    cs.param.g = 1;
    cs.param.s1 = 512;
    cs.param.s2 = 512;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_031)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 16;
    cs.param.g = 1;
    cs.param.s1 = 512;
    cs.param.s2 = 512;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::B;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 10;
    cs.param.nxtTokens = 1000;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 5;
    cs.param.prefixTensorData = {100};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_032)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 16;
    cs.param.g = 1;
    cs.param.s1 = 512;
    cs.param.s2 = 512;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::B;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 6;
    cs.param.prefixTensorData = {0};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_033)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 4;
    cs.param.n2 = 16;
    cs.param.g = 1;
    cs.param.s1 = 4096;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_034)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 4;
    cs.param.n2 = 16;
    cs.param.g = 1;
    cs.param.s1 = 4096;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 1);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_035)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_036)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 3096;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_037)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 2048;
    cs.param.nxtTokens = 100;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 2;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_038)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 1024;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_039)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = -900;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_040)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = -900;
    cs.param.nxtTokens = 1024;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_041)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 10;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 2048;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_042)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 10;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 1024;
    cs.param.nxtTokens = 2048;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 2;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
    ASSERT_EQ(GetImplMode(cs.forwardCtx.GetTilingData()), 2);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_043)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 64;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 3;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 3);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_044)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 16;
    cs.param.g = 1;
    cs.param.s1 = 4096;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::B;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 5;
    cs.param.prefixTensorData = {100};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_045)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 16;
    cs.param.g = 1;
    cs.param.s1 = 4096;
    cs.param.s2 = 4096;
    cs.param.d = 128;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::B;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 6;
    cs.param.prefixTensorData = {0};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_046)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 9;
    cs.param.n2 = 13;
    cs.param.g = 1;
    cs.param.s1 = 16;
    cs.param.s2 = 16;
    cs.param.d = 64;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 0;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_047)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 9;
    cs.param.n2 = 13;
    cs.param.g = 1;
    cs.param.s1 = 16;
    cs.param.s2 = 16;
    cs.param.d = 256;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 1;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_048)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 9;
    cs.param.n2 = 13;
    cs.param.g = 1;
    cs.param.s1 = 16;
    cs.param.s2 = 16;
    cs.param.d = 256;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::B_1_S1_S2;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::B;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 5;
    cs.param.prefixTensorData = {2, 3, 4, 5, 6, 7, 8, 9, 10};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_049)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 9;
    cs.param.n2 = 13;
    cs.param.g = 1;
    cs.param.s1 = 16;
    cs.param.s2 = 16;
    cs.param.d = 256;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::B;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::SBH;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 6;
    cs.param.prefixTensorData = {2, 3, 4, 5, 6, 7, 8, 9, 10};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_050)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 1;
    cs.param.n2 = 2;
    cs.param.g = 1;
    cs.param.s1 = 2048;
    cs.param.s2 = 2048;
    cs.param.d = 64;
    cs.param.dtype = ge::DT_FLOAT;
    cs.param.pseShapeType = PseShapeType::NONE;
    cs.param.dropMaskShapeType = DropMaskShapeType::NONE;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::PREFIXCOMPRESS;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::B;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 65536;
    cs.param.nxtTokens = 0;
    cs.param.layoutType = LayoutType::TND;
    cs.param.innerPrecise = 0;
    cs.param.sparseMode = 6;
    cs.param.prefixTensorData = {2500};
    cs.param.actualSeqQLenTensorData = {2048};
    cs.param.actualSeqKVLenTensorData = {3028};

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 5);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_051)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 40;
    cs.param.n2 = 12;
    cs.param.g = 1;
    cs.param.s1 = 256;
    cs.param.s2 = 256;
    cs.param.d = 144;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 100;
    cs.param.nxtTokens = 200;
    cs.param.layoutType = LayoutType::BSND;
    cs.param.innerPrecise = 1;
    cs.param.sparseMode = 4;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_TRUE(cs.Run());
    ASSERT_EQ(GetSparseType(cs.forwardCtx.GetTilingData()), 4);
    ASSERT_EQ(GetImplMode(cs.forwardCtx.GetTilingData()), 0);
}

TEST_F(Ts_Fas_Ascend910B2, Tc_SparseMode_052)
{
    /**
     * 用例 Shape 和 Attrs 信息
     */
    FasCase cs;
    cs.param.b = 40;
    cs.param.n2 = 12;
    cs.param.g = 1;
    cs.param.s1 = 256;
    cs.param.s2 = 256;
    cs.param.d = 144;
    cs.param.dtype = ge::DT_FLOAT16;
    cs.param.pseShapeType = PseShapeType::B_N1_S1_S2;
    cs.param.dropMaskShapeType = DropMaskShapeType::B_N1_S1_S2;
    cs.param.paddingMaskShapeType = PaddingMaskShapeType::NONE;
    cs.param.attenMaskShapeType = AttenMaskShapeType::SPARSE;
    cs.param.attenMaskDtype = ge::DT_BOOL;
    cs.param.prefixShapeType = PrefixShapeType::NONE;
    cs.param.scale = 0.5;
    cs.param.keepProb = 0.9;
    cs.param.preTokens = 100;
    cs.param.nxtTokens = 200;
    cs.param.layoutType = LayoutType::BSND;
    cs.param.innerPrecise = 1;
    cs.param.sparseMode = 4;

    /**
     * 用例 预制条件修改, 期望结果设置
     */
    cs.preTilingRunCbf = FaCase::PreTilingRunCbf_SetPlatformInfoNull;

    /**
     * 运行用例
     */
    ASSERT_TRUE(cs.Init());
    ASSERT_EQ(cs.Run(), cs.forward.exp.success);
}