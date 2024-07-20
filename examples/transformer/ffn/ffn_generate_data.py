#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np

if __name__ == '__main__':
    x = np.random.uniform(-0.1, 0.1, (64, 1024)).astype(np.float16)
    weight1 = np.random.uniform(-0.1, 0.1, (4, 1024, 2048)).astype(np.float16)
    weight2 = np.random.uniform(-0.1, 0.1, (4, 2048, 1024)).astype(np.float16)
    bias1 = np.random.uniform(-0.1, 0.1, (4, 2048)).astype(np.float16)
    x.tofile('x.bin')
    weight1.tofile('weight1.bin')
    weight2.tofile('weight2.bin')
    bias1.tofile('bias1.bin')
