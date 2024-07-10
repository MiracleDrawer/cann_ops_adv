# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

# 正向
set(_FAS_OpApiSourcesExt
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score.cpp
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/aclnn_flash_attention_score.cpp
)
set(_FAS_OpProtoSourcesExt
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score_proto.cpp
)
file(GLOB _FAS_TilingSourcesExt_cc
        "${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score_tiling*.cc"
)
file(GLOB _FAS_TilingSourcesExt_cpp
        "${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score_tiling*.cpp"
)
set(_FAS_TilingSourcesExt
        ${_FAS_TilingSourcesExt_cc}
        ${_FAS_TilingSourcesExt_cpp}
)
set(_FAS_TilingPrivateIncludesExt
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost
)
set(_FAS_KernelTilingDataDefH
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score/ophost/flash_attention_score_tiling.h
)
aux_source_directory(${OPS_ADV_DIR}/src/transformer/flash_attention_score _FAS_KernelSourcesExt)
set(_FAS_TargetPrivateIncludeExt
)
set(_FAS_TargetPrivateLinkLibrariesExt
)

# 反向
set(_FAG_OpApiSourcesExt
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad.cpp
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/aclnn_flash_attention_score_grad.cpp
)
set(_FAG_OpProtoSourcesExt
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_proto.cpp
)
file(GLOB _FASG_TilingSourcesExt_cc
        "${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling*.cc"
)
file(GLOB _FASG_TilingSourcesExt_cpp
        "${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling*.cpp"
)
set(_FASG_TilingSourcesExt
        ${_FASG_TilingSourcesExt_cc}
        ${_FASG_TilingSourcesExt_cpp}
)
set(_FASG_TilingPrivateIncludesExt
)
file(GLOB _FASG_KernelTilingDataDefH_def
        "${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling*_def.h"
)
set(_FASG_KernelTilingDataDefH
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling_common.h
        ${_FASG_KernelTilingDataDefH_def}
        ${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling.h
)
aux_source_directory(${OPS_ADV_DIR}/src/transformer/flash_attention_score_grad _FASG_KernelSourcesExt)
set(_FAG_TargetPrivateIncludeExt
)
set(_FAG_TargetPrivateLinkLibrariesExt
        ops_utils_tiling_headers            # 用于 FAG 制定模板优先级相关实现
)

# 自动反向
set(_FA_OpApiSourcesExt
        ${_FAS_OpApiSourcesExt}
        ${_FAG_OpApiSourcesExt}
)
set(_FA_OpProtoSourcesExt
        ${_FAS_OpProtoSourcesExt}
        ${_FAG_OpProtoSourcesExt}
)
set(_FA_TilingSourcesExt
        ${_FAS_TilingSourcesExt}
        ${_FASG_TilingSourcesExt}
)
set(_FA_TilingPrivateIncludesExt
        ${_FAS_TilingPrivateIncludesExt}
        ${_FASG_TilingPrivateIncludesExt}
)
set(_FA_KernelSourcesExt
        ${_FAS_KernelSourcesExt}
        ${_FASG_KernelSourcesExt}
)
set(_FA_KernelTilingDataDefH
        ${OPS_ADV_DIR}/src/utils/inc/tiling/data_copy_transpose_tiling_def.h
        ${_FAS_KernelTilingDataDefH}
        ${_FASG_KernelTilingDataDefH}
)
set(_FA_KernelPrivateCompileDefinitionsExt
        KernelSoSuffix fp16 ORIG_DTYPE_QUERY=DT_FLOAT16 DTYPE_DQ=half
        KernelSoSuffix bf16 ORIG_DTYPE_QUERY=DT_BF16    DTYPE_DQ=bfloat16_t
        KernelSoSuffix fp32 ORIG_DTYPE_QUERY=DT_FLOAT   DTYPE_DQ=float
)
set(_FA_TargetPrivateIncludeExt
        ${_FAS_TargetPrivateIncludeExt}
        ${_FAG_TargetPrivateIncludeExt}
)
set(_FA_TargetPrivateLinkLibrariesExt
        ${_FAS_TargetPrivateLinkLibrariesExt}
        ${_FAG_TargetPrivateLinkLibrariesExt}
)

OpsTest_Level2_AddOp(
        BRIEF                                   Fa
        SNAKE                                   flash_attention
        OPAPI_SOURCES_EXT                       ${_FA_OpApiSourcesExt}
        PROTO_SOURCES_EXT                       ${_FA_OpProtoSourcesExt}
        TILING_SOURCES_EXT                      ${_FA_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_FA_TilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                      ${_FA_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_FA_KernelTilingDataDefH}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_FA_KernelPrivateCompileDefinitionsExt}
        TARGET_PRIVATE_INCLUDES_EXT             ${_FA_TargetPrivateIncludeExt}
        TARGET_PRIVATE_LINK_LIBRARIES_EXT       ${_FA_TargetPrivateLinkLibrariesExt}
)
