# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_IFA_TilingSourcesExt
        ${OPS_ADV_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling.cc
        ${OPS_ADV_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling_register.cc
)

set(_IFA_KernelTilingDataDefH
        ${OPS_ADV_DIR}/src/utils/inc/tiling/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling.h
)

set(_IFA_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers
)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/transformer/incre_flash_attention _IFA_KernelSourcesExt)

OpsTest_Level2_AddOp(
         BRIEF                                         Ifa
         SNAKE                                         incre_flash_attention
         KERNEL_TILING_DATA_DEF_H                      ${_IFA_KernelTilingDataDefH}
         KERNEL_SOURCES_EXT                            ${_IFA_KernelSourcesExt}
         TILING_SOURCES_EXT                            ${_IFA_TilingSourcesExt}
         UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT       ${_IFA_TargetPrivateLinkLibrariesExt}
)
