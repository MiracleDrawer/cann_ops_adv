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
 * \file ifa_public_define.h
 * \brief
 */
#ifndef IFA_PUBLIC_DEFINE_H
#define IFA_PUBLIC_DEFINE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::WaitFlag;
using matmul::Matmul;
using matmul::MatmulType;
using AscendC::SoftmaxConfig;

#define SYNC_BEFORE_DATACOPY()                                                                     \
  do {                                                                                             \
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3)); \
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);                                                    \
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);                                                   \
  } while (0)
#define FLT_MAX 3.402823466e+38F

constexpr MatmulConfig CFG_NORM_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig CFG_MDL_EXCEED = GetMDLConfig(true);

// CFG_NORM_EXCEED_INIT: doNorm, enable intrinsicsCheck and Init
constexpr MatmulConfig CFG_NORM_EXCEED_INIT{true,
                                            false,
                                            false,
                                            0,
                                            0,
                                            0,
                                            true,
                                            false,
                                            false,
                                            false,
                                            false,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            true,
                                            false,
                                            false,
                                            false,
                                            false,
                                            false,
                                            BatchMode::NONE,
                                            false,
                                            false,
                                            true,
                                            false,
                                            true,
                                            false,
                                            true,
                                            IterateMode::ITERATE_MODE_ALL};

// CFG_MDL_EXCEED_INIT: enable MDL, intrinsicsCheck and Init
constexpr MatmulConfig CFG_MDL_EXCEED_INIT{false,
                                           false,
                                           true,
                                           0,
                                           0,
                                           0,
                                           true,
                                           false,
                                           false,
                                           false,
                                           false,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           false,
                                           false,
                                           false,
                                           false,
                                           false,
                                           false,
                                           BatchMode::NONE,
                                           false,
                                           false,
                                           true,
                                           false,
                                           true,
                                           false,
                                           true,
                                           IterateMode::ITERATE_MODE_ALL};

// CFG_MDL_EXCEED_INIT_CALLBACK: enable MDL, intrinsicsCheck and Init, enable CALLBACK, enable unitflag
constexpr MatmulConfig CFG_MDL_EXCEED_INIT_CALLBACK{false,
                                            false,
                                            true,
                                            0,
                                            0,
                                            0,
                                            true,
                                            false,
                                            false,
                                            false,
                                            false,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            true, // enable unitflag
                                            false,
                                            false,
                                            false,
                                            false,
                                            false,
                                            BatchMode::NONE,
                                            false,
                                            false,
                                            true,
                                            false,
                                            true,
                                            false,
                                            true,
                                            IterateMode::ITERATE_MODE_ALL,
                                            false};

constexpr SoftmaxConfig IFA_SOFTMAX_FLASHV2_CFG = { false }; // 将isCheckTiling设置为false

constexpr uint32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
constexpr float FLOAT_ZERO = 0;
constexpr float FLOAT_MAX = FLT_MAX;

constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;

constexpr uint32_t MAX_UINT16 = 65535;
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
constexpr uint32_t IFA_MAX_REPEAT_TIMES = 256;

#define VMLA_ONE_REPEATE_ROW_COUNT 4
#define VMLA_ONE_REPEATE_COLUMN_COUNT 16
#define BMM2_PARALLEL_ROW_COUNT 32
#define FP16_ONE_BLOCK_SIZE 16
#define FP32_ONE_BLOCK_SIZE 8
#define INT32_ONE_BLOCK_SIZE 8
#define FP16_ONE_REPEATE_SIZE 128
#define FP32_ONE_REPEATE_SIZE 64

enum class LAYOUT { BSH = 0, SBH, BNSD, BSND };

template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, const bool PAGE_ATTENTION = false,
          const bool FLASH_DECODE = false, LAYOUT LAYOUT_T = LAYOUT::BSH, const bool PER_TOKEN = false,
          const bool SHARED_PREFIX = false, typename... Args>
struct IFAType {
  using queryType = Q_T;
  using kvType = KV_T;
  using outputType = OUT_T;
  using orginalType = ORIGIN_T;
  static constexpr bool pageAttention = PAGE_ATTENTION;
  static constexpr bool flashDecode = FLASH_DECODE;
  static constexpr LAYOUT layout = LAYOUT_T;
  static constexpr bool perToken = PER_TOKEN;
  static constexpr bool sharedPrefix = SHARED_PREFIX;
};

#endif  // IFA_PUBLIC_DEFINE_H