# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

file(GLOB_RECURSE _Sources src "*.cc" "*.cpp")
set(_PublicInclude
        inc
)
set(_PrivateInclude src)
set(_PublicLinkLibraries register graph_base ascendalog ascendcl nnopbase ascend_hal)
set(_PrivateLinkLibraries $<BUILD_INTERFACE:intf_utest_strict_compile_options> dl tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE})

# OpsTestUt_Utils
set(_Target ${UTest_NamePrefix}_Utils)
add_library(${_Target} SHARED)
target_sources(${_Target} PRIVATE ${_Sources})
target_include_directories(${_Target} PUBLIC ${_PublicInclude})
target_include_directories(${_Target} PRIVATE ${_PrivateInclude})
target_compile_definitions(${_Target} PUBLIC LOG_CPP PROCESS_LOG)
target_link_libraries(${_Target} PUBLIC ${_PublicLinkLibraries} $<BUILD_INTERFACE:intf_pub_utest>)
target_link_libraries(${_Target} PRIVATE ${_PrivateLinkLibraries})
target_compile_options(${_Target} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>)
