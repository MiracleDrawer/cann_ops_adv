# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_executable(test_ffn
        test_ffn.cpp
)

add_executable(test_ffn_v2
        test_ffn_v2.cpp
)

add_executable(test_ffn_v3
        test_ffn_v3.cpp
)

if (NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend310p")
        add_execute_example(
                TARGET_NAME test_ffn
                SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ffn_case.sh
                TEST_CASE ffn_case
                ACLNN_FUNC "aclnnFFN"
        )
        add_execute_example(
                TARGET_NAME test_ffn_v2
                SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ffn_case.sh
                TEST_CASE ffn_v2_case
                ACLNN_FUNC "aclnnFFNV2"
        )
        add_execute_example(
                TARGET_NAME test_ffn_v3
                SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ffn_case.sh
                TEST_CASE ffn_v3_case
                ACLNN_FUNC "aclnnFFNV3"
        )
endif()

add_execute_example(
        TARGET_NAME test_ffn_v2
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ffn_case.sh
        TEST_CASE ffn_v2_case_no_expert
        ACLNN_FUNC "aclnnFFNV2"
)
add_execute_example(
        TARGET_NAME test_ffn_v3
        SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ffn_case.sh
        TEST_CASE ffn_v3_case_no_expert
        ACLNN_FUNC "aclnnFFNV3"
)

install(TARGETS test_ffn test_ffn_v2 test_ffn_v3
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        OPTIONAL
)
