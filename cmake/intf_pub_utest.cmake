# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# UTest 场景, 严格检查源码实现的编译选项
add_library(intf_utest_strict_compile_options INTERFACE)
target_compile_options(intf_utest_strict_compile_options
        INTERFACE
            -Wall -Werror -fno-common -fno-strict-aliasing
)

# UTest 场景, 公共配置
add_library(intf_pub_utest INTERFACE)
target_compile_definitions(intf_pub_utest
        INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=0>    # 必须设置, 以保证与 CANN 包内其他依赖库兼容
            ASCENDC_OP_TEST
            ASCENDC_OP_TEST_UT
            $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
            $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
)
target_compile_options(intf_pub_utest
        INTERFACE
            -Werror
            -fPIC
            $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
            -g
            $<$<BOOL:${ENABLE_GCOV}>:--coverage -fprofile-arcs -ftest-coverage>
            $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>
)
target_include_directories(intf_pub_utest
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/include
            ${ASCEND_CANN_PACKAGE_PATH}/include/external
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)
target_link_directories(intf_pub_utest
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/lib64
            ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
            ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64/stub
)
target_link_libraries(intf_pub_utest
        INTERFACE
            $<$<BOOL:${ENABLE_GCOV}>:gcov>
            pthread
)
target_link_options(intf_pub_utest
        INTERFACE
            $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
            $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
)

# UTest 场景, 编译 Target 名称公共前缀
set(UTest_NamePrefix UTest)
