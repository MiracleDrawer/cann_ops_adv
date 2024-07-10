# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# Custom 包场景, Host 侧各 Target 公共编译配置
# 注意: 为保证与 built-in 包编译流程兼容, intf_pub 名称不可变更
add_library(intf_pub INTERFACE)
target_include_directories(intf_pub
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/include
            ${ASCEND_CANN_PACKAGE_PATH}/include/external
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
)
target_link_directories(intf_pub
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_compile_options(intf_pub
        INTERFACE
            -fPIC
            -O2
            -Wall
            $<$<CONFIG:Debug>:-g>
            $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
)
target_compile_definitions(intf_pub
        INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=0>     # 必须设置, 以保证与 CANN 包内其他依赖库兼容
            $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
)
target_link_options(intf_pub
        INTERFACE
            $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
            $<$<CONFIG:Release>:-s>
            -Wl,-z,relro
            -Wl,-z,now
            -Wl,-z,noexecstack
)
