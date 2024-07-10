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
 * \file control_info.cpp
 * \brief 测试用例控制信息.
 */

#include "tests/utils/control_info.h"

using namespace ops::adv::tests::utils;

ControlInfo::ControlInfo(bool runTiling, bool runKernel, bool deterministic)
    : runTiling(runTiling), runKernel(runKernel), deterministic(deterministic)
{
}
