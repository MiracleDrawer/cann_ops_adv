# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

########################################################################################################################
# 预定义变量
########################################################################################################################

# 缓存此次编译 Kernel 目标
set(_OpsTestUt_KernelLibraries "" CACHE INTERNAL "" FORCE)

# 缓存所有算子 Utest 场景 Tiling 动态库相关信息
set(_OpsTestUt_TilingSources            "" CACHE INTERNAL "" FORCE)  # Sources
set(_OpsTestUt_TilingPrivateIncludesExt "" CACHE INTERNAL "" FORCE)  # PrivateIncludesExt
set(_OpsTestUt_TilingLinkLibrariesExt   "" CACHE INTERNAL "" FORCE)  # LinkLibrariesExt

# 缓存所有算子 Utest 场景 OpApi 动态库相关信息
set(_OpsTestUt_OpApiSources            "" CACHE INTERNAL "" FORCE)  # Sources
set(_OpsTestUt_OpApiPrivateIncludesExt "" CACHE INTERNAL "" FORCE)  # PrivateIncludesExt
set(_OpsTestUt_OpApiLinkLibrariesExt   "" CACHE INTERNAL "" FORCE)  # LinkLibrariesExt

# 缓存所有算子 Utest 场景 OpProto 动态库相关信息
set(_OpsTestUt_OpProtoSources            "" CACHE INTERNAL "" FORCE)  # Sources
set(_OpsTestUt_OpProtoPrivateIncludesExt "" CACHE INTERNAL "" FORCE)  # PrivateIncludesExt
set(_OpsTestUt_OpProtoLinkLibrariesExt   "" CACHE INTERNAL "" FORCE)  # LinkLibrariesExt

# 缓存所有算子 Utest 场景 Target 动态库相关信息
set(_OpsTestUt_OpsUtestLibraries "" CACHE INTERNAL "" FORCE)


########################################################################################################################
# 编译方法
########################################################################################################################

# Level1, 添加算子 OpApi 动态库
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      SOURCES_EXT                   : 可选参数, 额外源文件
      PRIVATE_INCLUDES_EXT          : 可选参数, 额外头文件搜索路径
      PRIVATE_LINK_LIBRARIES_EXT    : 可选参数, 额外链接库
备注说明:
  本函数提供编译算子对应 opapi.so 的功能. 下面介绍在调用本函数时所需了解的一些背景知识和注意事项.
  1. 本函数假设算子对应 OpApi 的源文件位于 'src/${SUB_SYSTEM}/${SNAKE}/ophost' 路径, 并自动添加以下内容:
     a) 源文件: ${SNAKE}.cpp, aclnn_${SNAKE}.cpp
     b) 头文件搜索路径: 'src/${SUB_SYSTEM}/${SNAKE}/ophost'
  2. 若算子 '需要额外的源文件’ 或 ‘源文件不满足上述约定的默认路径', 则可通过 SOURCES_EXT 参数指定 '额外的源文件';
  3. 若算子 '需要额外头文件搜索路径’ 或 ‘本函数现有实现的头文件搜索路径设置不满足',
     则可通过 PRIVATE_INCLUDES_EXT 参数指定 '额外的头文件搜索路径'; 参数 PRIVATE_LINK_LIBRARIES_EXT 设置逻辑同理;
]]
function(OpsTest_Level1_AddOpOpApiShared)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE"
            "SOURCES_EXT;PRIVATE_INCLUDES_EXT;PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    get_filename_component(_L0_Src "${OPS_ADV_DIR}/src/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/ophost/${TMP_SNAKE}.cpp" REALPATH)
    if (NOT EXISTS "${_L0_Src}")
        set(_L0_Src)
    endif ()
    get_filename_component(_L2_Src "${OPS_ADV_DIR}/src/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/ophost/aclnn_${TMP_SNAKE}.cpp" REALPATH)
    if (NOT EXISTS "${_L2_Src}")
        set(_L2_Src)
    endif ()

    set(_Sources ${TMP_SOURCES_EXT} ${_L0_Src} ${_L2_Src})
    list(REMOVE_DUPLICATES _Sources)
    if (_Sources)
        set(_PrivateIncludeDirs ${TMP_PRIVATE_INCLUDES_EXT} ${OPS_ADV_DIR}/src/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/ophost)
        list(REMOVE_DUPLICATES _PrivateIncludeDirs)
        set(_OpsTestUt_OpApiSources            ${_OpsTestUt_OpApiSources}            ${_Sources}                       CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpApiPrivateIncludesExt ${_OpsTestUt_OpApiPrivateIncludesExt} ${_PrivateIncludeDirs}            CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpApiLinkLibrariesExt   ${_OpsTestUt_OpApiLinkLibrariesExt}   ${TMP_PRIVATE_LINK_LIBRARIES_EXT} CACHE INTERNAL "" FORCE)
    endif ()
endfunction()

# Level1, 添加算子 OpProto 动态库
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      SOURCES_EXT                   : 可选参数, 额外源文件
      PRIVATE_INCLUDES_EXT          : 可选参数, 额外头文件搜索路径
      PRIVATE_LINK_LIBRARIES_EXT    : 可选参数, 额外链接库
备注说明:
  本函数提供编译算子对应 OpProto.so 的功能. 下面介绍在调用本函数时所需了解的一些背景知识和注意事项.
  1. 本函数假设算子对应 OpProto 的源文件位于 'src/${SUB_SYSTEM}/${SNAKE}/ophost' 路径, 并自动添加以下内容:
     a) 源文件: ${SNAKE}_proto.cpp
     b) 头文件搜索路径: 'src/${SUB_SYSTEM}/${SNAKE}/ophost'
  2. 若算子 '需要额外的源文件’ 或 ‘源文件不满足上述约定的默认路径', 则可通过 SOURCES_EXT 参数指定 '额外的源文件';
  3. 若算子 '需要额外头文件搜索路径’ 或 ‘本函数现有实现的头文件搜索路径设置不满足',
     则可通过 PRIVATE_INCLUDES_EXT 参数指定 '额外的头文件搜索路径'; 参数 PRIVATE_LINK_LIBRARIES_EXT 设置逻辑同理;
]]
function(OpsTest_Level1_AddOpOpProtoShared)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE"
            "SOURCES_EXT;PRIVATE_INCLUDES_EXT;PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    get_filename_component(_Src "${OPS_ADV_DIR}/src/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/ophost/${TMP_SNAKE}_proto.cpp" REALPATH)
    if (NOT EXISTS "${_Src}")
        set(_Src)
    endif ()

    set(_Sources ${TMP_SOURCES_EXT} ${_Src})
    list(REMOVE_DUPLICATES _Sources)
    if (_Sources)
        set(_PrivateIncludeDirs ${TMP_PRIVATE_INCLUDES_EXT} ${OPS_ADV_DIR}/src/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/ophost)
        list(REMOVE_DUPLICATES _PrivateIncludeDirs)
        set(_OpsTestUt_OpProtoSources            ${_OpsTestUt_OpProtoSources}            ${_Sources}                       CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpProtoPrivateIncludesExt ${_OpsTestUt_OpProtoPrivateIncludesExt} ${_PrivateIncludeDirs}            CACHE INTERNAL "" FORCE)
        set(_OpsTestUt_OpProtoLinkLibrariesExt   ${_OpsTestUt_OpProtoLinkLibrariesExt}   ${TMP_PRIVATE_LINK_LIBRARIES_EXT} CACHE INTERNAL "" FORCE)
    endif ()
endfunction()

# Level1, 添加算子 Tiling 动态库
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      SOURCES_EXT                   : 可选参数, 额外源文件
      PRIVATE_INCLUDES_EXT          : 可选参数, 额外头文件搜索路径
      PRIVATE_LINK_LIBRARIES_EXT    : 可选参数, 额外链接库
备注说明:
  本函数提供编译算子对应 tiling.so 的功能. 下面介绍在调用本函数时所需了解的一些背景知识和注意事项.
  1. 本函数假设算子对应 Tiling 的源文件位于 'src/${SUB_SYSTEM}/${SNAKE}/ophost/${SNAKE}_tiling.cpp/.cc' 路径;
  2. 若算子 '需要额外的源文件’ 或 ‘源文件不满足上述约定的默认路径', 则可通过 SOURCES_EXT 参数指定 '额外的源文件';
  3. 若算子 '需要额外头文件搜索路径’ 或 ‘本函数现有实现的头文件搜索路径设置不满足',
     则可通过 PRIVATE_INCLUDES_EXT 参数指定 '额外的头文件搜索路径'; 参数 PRIVATE_LINK_LIBRARIES_EXT 设置逻辑同理;
]]
function(OpsTest_Level1_AddOpTilingShared)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE"
            "SOURCES_EXT;PRIVATE_INCLUDES_EXT;PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    file(GLOB _Src1 "${OPS_ADV_DIR}/src/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/ophost/${TMP_SNAKE}_tiling.cc")
    file(GLOB _Src2 "${OPS_ADV_DIR}/src/${TMP_SUB_SYSTEM}/${TMP_SNAKE}/ophost/${TMP_SNAKE}_tiling.cpp")
    list(APPEND _Sources ${TMP_SOURCES_EXT} ${_Src1} ${_Src2})
    list(REMOVE_DUPLICATES _Sources)

    set(_OpsTestUt_TilingSources            ${_OpsTestUt_TilingSources}            ${_Sources}                       CACHE INTERNAL "" FORCE)
    set(_OpsTestUt_TilingPrivateIncludesExt ${_OpsTestUt_TilingPrivateIncludesExt} ${TMP_PRIVATE_INCLUDES_EXT}       CACHE INTERNAL "" FORCE)
    set(_OpsTestUt_TilingLinkLibrariesExt   ${_OpsTestUt_TilingLinkLibrariesExt}   ${TMP_PRIVATE_LINK_LIBRARIES_EXT} CACHE INTERNAL "" FORCE)
endfunction()

# Level1, 添加算子 Kernel 动态库
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      SOURCES_EXT                       : 可选参数, 额外源文件
      TILING_DATA_DEF_H                 : 可选参数, 算子 Kernel 所需 TilingData 定义头文件
      PRIVATE_COMPILE_DEFINITIONS_EXT   : 可选参数, 额外编译宏
备注说明:
  本函数提供编译算子对应 kernel.a 的功能. 下面介绍在调用本函数时所需了解的一些背景知识和注意事项.
  1. 本函数假设算子对应 Kernel 的源文件位于 'src/${SUB_SYSTEM}/${SNAKE}/${SNAKE}.cpp' 路径;
  2. 若算子 '需要额外的源文件’ 或 ‘源文件不满足上述约定的默认路径', 则可通过 SOURCES_EXT 参数指定 '额外的源文件';
  3. 当前 Ascend C 融合算子 Kernel 一般需要多个(>2) TilingData 定义头文件,
     如 flash_attention_score 算子需要 data_copy_transpose_tiling_def.h 和 flash_attention_score_tiling.h;
     - 在 NPU 编译时, 编译框架使用 ccec 编译器并使用 -include 编译选项注入所需的多个 TilingData 定义头文件 至 Kernel 源文件;
     - 但 CPU 编译时, CMake 早期版本在处理 -include 选项时有 Bug(只第一个 -include 指定的头文件生效).
       故本框架通过新增一个 {op_brief_name}_tiling_data.h, 再在该文件中 include 所需的多个 TilingData 定义头文件,
       再通过 -include 编译选项注入 {op_brief_name}_tiling_data.h 的方式规避 CPU 编译时 CMake 不能处理多个 -include 选项的 Bug;
  4. PRIVATE_COMPILE_DEFINITIONS_EXT 设置格式如下:
        optional{KernelSoSuffix suffix} optional{OtherCompileDefinitions} optional{OtherCompileDefinitions}
            其中 KernelSoSuffix 为标识 suffix 起点关键字
     4.1 当不设置 KernelSoSuffix 时, 本函数仅会编译一个 Kernel.so 并以 OtherCompileDefinitions 设置编译 Definitions(如设置);
     4.2 若当设置 KernelSoSuffix 时, 本函数会按 KernelSoSuffix 个数逐个编译 Kernel.so,
         并以 KernelSoSuffix 间 OtherCompileDefinitions 设置编译宏定义;
]]
function(OpsTest_Level1_AddOpKernelShared)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE"
            "SOURCES_EXT;TILING_DATA_DEF_H;PRIVATE_COMPILE_DEFINITIONS_EXT"
            ""
            ${ARGN}
    )
    # 生成 Kernel 所需的结构体表示的对应 tiling.h
    string(TOLOWER ${TMP_BRIEF} tmp_brief)
    set(_tmp_files ${TMP_TILING_DATA_DEF_H})
    list(REMOVE_DUPLICATES _tmp_files)
    set(_ori_files)
    set(_define_py ${OPS_ADV_DIR}/cmake/scripts/utest/gen_tiling_data_stub.py)
    foreach (_tmp ${_tmp_files})
        if (EXISTS ${_tmp})
            list(APPEND _ori_files "-s=${_tmp}")
        else ()
            message(FATAL_ERROR "${_tmp} not exist.")
        endif ()
    endforeach ()
    # 生成目标根目录
    get_filename_component(_OpsTest_GenDir "${CMAKE_CURRENT_BINARY_DIR}/gen" REALPATH)
    get_filename_component(_OpsTest_GenDirInc "${_OpsTest_GenDir}/inc" REALPATH)
    execute_process(
            COMMAND ${HI_PYTHON} ${_define_py} "-o=${tmp_brief}" ${_ori_files} "-d=${_OpsTest_GenDirInc}"
    )
    set(_Target ${UTest_NamePrefix}_${TMP_BRIEF}_OpTilingDataDef)
    add_library(${_Target} INTERFACE)
    target_include_directories(${_Target} INTERFACE ${_OpsTest_GenDirInc})

    # 编译变量处理
    set(_TargetPrefix  ${UTest_NamePrefix}_${TMP_BRIEF}_Kernel)
    aux_source_directory(${OPS_ADV_DIR}/src/${TMP_SUB_SYSTEM}/${TMP_SNAKE} _Sources)
    list(APPEND _Sources ${TMP_SOURCES_EXT})
    set(_PrivateIncludeDirectories
            ${_OpsTest_GenDirInc}
            ${OPS_ADV_DIR}/src/${TMP_SUB_SYSTEM}/${TMP_SNAKE}
            ${OPS_ADV_DIR}/src/utils/inc/kernel
    )
    set(_PrivateCompileOptions
            -include ${_OpsTest_GenDirInc}/tiling/${tmp_brief}/tiling_stub.h
    )
    set(_PrivateLinkLibraries
            -Wl,--as-needed
            -Wl,--no-whole-archive
            c_sec
            $<BUILD_INTERFACE:intf_pub_utest>
    )
    # 多 Kernel 处理
    set(_OpsTestUt_KernelLibraries CACHE INTERNAL "" FORCE)
    list(FIND TMP_PRIVATE_COMPILE_DEFINITIONS_EXT KernelSoSuffix _TMP_IDX)
    if ("${_TMP_IDX}" STREQUAL "-1")
        # 不存在多 Kernel 配置时, 不添加后缀, 编译一个 Kernel.so
        set(_Target ${_TargetPrefix})
        add_library(${_Target} SHARED)
        target_sources(${_Target} PRIVATE ${_Sources})
        target_include_directories(${_Target} PRIVATE ${_PrivateIncludeDirectories})
        target_compile_definitions(${_Target} PRIVATE ${TMP_PRIVATE_COMPILE_DEFINITIONS_EXT})
        target_compile_options(${_Target} PRIVATE ${_PrivateCompileOptions})
        target_link_libraries(${_Target} PRIVATE ${_PrivateLinkLibraries})
        target_link_libraries(${_Target} PUBLIC tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE})
        set(_OpsTestUt_KernelLibraries ${_OpsTestUt_KernelLibraries} ${_Target} CACHE INTERNAL "" FORCE)
    else ()
        # 存在 1/n 多 Kernel 配置时, 添加后缀, 编译多 Kernel.so
        while (NOT "${_TMP_IDX}" STREQUAL "-1")
            # 获取当前 Suffix
            math(EXPR _SuffixIdx "${_TMP_IDX} + 1")
            list(GET TMP_PRIVATE_COMPILE_DEFINITIONS_EXT ${_SuffixIdx} _SuffixVal)
            math(EXPR _SubLstBgnIdx "${_TMP_IDX} + 2")
            list(SUBLIST TMP_PRIVATE_COMPILE_DEFINITIONS_EXT ${_SubLstBgnIdx} -1 TMP_PRIVATE_COMPILE_DEFINITIONS_EXT)
            # 获取此 Suffix 对应 CompileDefinitions
            set(_DefBgnIdx 0)
            list(FIND TMP_PRIVATE_COMPILE_DEFINITIONS_EXT KernelSoSuffix _TMP_IDX)
            list(SUBLIST TMP_PRIVATE_COMPILE_DEFINITIONS_EXT 0 ${_TMP_IDX} _SubCompileDefinitions)
            # 编译目标
            set(_Target ${_TargetPrefix}_${_SuffixVal})
            add_library(${_Target} SHARED)
            target_sources(${_Target} PRIVATE ${_Sources})
            target_include_directories(${_Target} PRIVATE ${_PrivateIncludeDirectories})
            target_compile_definitions(${_Target} PRIVATE ${_SubCompileDefinitions})
            target_compile_options(${_Target} PRIVATE ${_PrivateCompileOptions})
            target_link_libraries(${_Target} PRIVATE ${_PrivateLinkLibraries})
            target_link_libraries(${_Target} PUBLIC tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE})
            set(_OpsTestUt_KernelLibraries ${_OpsTestUt_KernelLibraries} ${_Target} CACHE INTERNAL "" FORCE)
        endwhile ()
    endif ()
endfunction()

# Level1, 添加算子 Utest 动态库
#[[
调用参数:
  one_value_keywords:
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      SOURCES_EXT                   : 可选参数, 额外源文件
      PRIVATE_INCLUDES_EXT          : 可选参数, 额外头文件搜索路径
      PRIVATE_LINK_LIBRARIES_EXT    : 可选参数, 额外链接库
备注说明:
  本函数提供编译算子对应 Utest 动态库 的功能.
  1. 在 Utest 场景下, 本函数将会将 comm 及 utest 路径下所有源文件编译成一个动态库, 在所有算子的动态库编译完成后,
     在 OpsTestUt_AddLaunch 函数内链接这些动态库(_OpsTestUt_OpsUtestLibraries)
     并添加 GTest 所需的 main 函数, 统一编译成一个可执行程序;
]]
function(OpsTest_Level1_AddOpUTestShared)
    cmake_parse_arguments(
            TMP
            ""
            "BRIEF;SNAKE"
            "SOURCES_EXT;PRIVATE_INCLUDES_EXT;PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )
    set(_Sources ${TMP_SOURCES_EXT})
    set(_PrivateInclude
            ${ASCEND_CANN_PACKAGE_PATH}/include
            ${TMP_PRIVATE_INCLUDES_EXT}
    )
    set(_PrivateLinkLibraries
            -Wl,--as-needed
            -Wl,--no-whole-archive
            c_sec exe_graph
            ${TMP_PRIVATE_LINK_LIBRARIES_EXT}
            $<BUILD_INTERFACE:intf_utest_strict_compile_options>
            ${UTest_NamePrefix}_${TMP_BRIEF}_OpTilingDataDef
            tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE}
    )
    set(_Target ${UTest_NamePrefix}_${TMP_BRIEF}_Utest)
    add_library(${_Target} SHARED)

    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/comm)
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/comm/inc)
            set(_Common_Target ${UTest_NamePrefix}_${TMP_BRIEF}_Common)
            add_library(${_Common_Target} SHARED)
            aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/comm/src _Common_Sources)
            target_sources(${_Common_Target} PRIVATE ${_Common_Sources})
            target_include_directories(${_Common_Target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/comm/inc)
            target_include_directories(${_Common_Target} PRIVATE
                    ${TMP_PRIVATE_INCLUDES_EXT}
            )
            target_link_libraries(${_Common_Target} PRIVATE
                    -Wl,--as-needed
                    -Wl,--no-whole-archive
                    c_sec exe_graph
                    ${TMP_PRIVATE_LINK_LIBRARIES_EXT}
                    $<BUILD_INTERFACE:intf_utest_strict_compile_options>
                    tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE}
                    $<BUILD_INTERFACE:intf_pub_utest>
                    ${UTest_NamePrefix}_Utils
                    ${UTest_NamePrefix}_OpTiling
                    ${UTest_NamePrefix}_OpProto
                    ${UTest_NamePrefix}_${TMP_BRIEF}_OpTilingDataDef
            )
            list(APPEND _PrivateLinkLibraries ${_Common_Target})
        else ()
            file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/comm/*.cc")
            file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/comm/*.cpp")
            list(APPEND _Sources ${_Src1} ${_Src2})
            list(APPEND _PrivateInclude ${CMAKE_CURRENT_SOURCE_DIR}/comm)
        endif ()
    endif ()

    file(GLOB_RECURSE _Src1 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cc")
    file(GLOB_RECURSE _Src2 "${CMAKE_CURRENT_SOURCE_DIR}/utest/*.cpp")
    list(APPEND _Sources ${TMP_SOURCES_EXT} ${_Src1} ${_Src2})
    list(APPEND _PrivateInclude ${CMAKE_CURRENT_SOURCE_DIR}/utest)
    list(APPEND _PrivateLinkLibraries
            $<BUILD_INTERFACE:intf_pub_utest>
            ${UTest_NamePrefix}_Utils
            ${UTest_NamePrefix}_Utest
            ${UTest_NamePrefix}_OpTiling
    )
    set(_OpsTestUt_OpsUtestLibraries ${_OpsTestUt_OpsUtestLibraries} ${_Target} CACHE INTERNAL "" FORCE)
    add_dependencies(${_Target} ${_OpsTestUt_KernelLibraries})

    list(REMOVE_DUPLICATES _Sources)
    target_sources(${_Target} PRIVATE ${_Sources})
    target_include_directories(${_Target} PRIVATE ${_PrivateInclude})
    target_link_libraries(${_Target} PRIVATE ${_PrivateLinkLibraries})
endfunction()

# Level2, 添加算子 Utest 动态库 及其所需全部库(Tiling.so, Kernel.so)
#[[
调用参数:
  one_value_keywords:
      SUB_SYSTEM : 必选参数, 用于指定算子所属子系统, 如 transformer
      BRIEF : 必选参数, 用于指定算子缩略名(建议以大驼峰命名, 与算子实际名称无强制对应关系), 如 Fag/Fas/Fa
      SNAKE : 必选参数, 用于指定算子全名, 如 flash_attention_score_grad
  multi_value_keywords:
      OPAPI_SOURCES_EXT                         : 透传参数, 详情参见 OpsTest_Level1_AddOpOpApiShared 函数说明
      OPAPI_PRIVATE_INCLUDES_EXT                : 透传参数, 详情参见 OpsTest_Level1_AddOpOpApiShared 函数说明
      OPAPI_PRIVATE_LINK_LIBRARIES_EXT          : 透传参数, 详情参见 OpsTest_Level1_AddOpOpApiShared 函数说明
      PROTO_SOURCES_EXT                         : 透传参数, 详情参见 OpsTest_Level1_AddOpOpProtoShared 函数说明
      PROTO_PRIVATE_INCLUDES_EXT                : 透传参数, 详情参见 OpsTest_Level1_AddOpOpProtoShared 函数说明
      PROTO_PRIVATE_LINK_LIBRARIES_EXT          : 透传参数, 详情参见 OpsTest_Level1_AddOpOpProtoShared 函数说明
      TILING_SOURCES_EXT                        : 透传参数, 详情参见 OpsTest_Level1_AddOpTilingShared 函数说明
      TILING_PRIVATE_INCLUDES_EXT               : 透传参数, 详情参见 OpsTest_Level1_AddOpTilingShared 函数说明
      TILING_PRIVATE_LINK_LIBRARIES_EXT         : 透传参数, 详情参见 OpsTest_Level1_AddOpTilingShared 函数说明
      KERNEL_SOURCES_EXT                        : 透传参数, 详情参见 OpsTest_Level1_AddOpKernelShared 函数说明
      KERNEL_TILING_DATA_DEF_H                  : 透传参数, 详情参见 OpsTest_Level1_AddOpKernelShared 函数说明
      KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT    : 透传参数, 详情参见 OpsTest_Level1_AddOpKernelShared 函数说明
      TARGET_SOURCES_EXT                        : 透传参数, 详情参见 OpsTest_Level1_AddOpUTestShared 函数说明
      TARGET_PRIVATE_INCLUDES_EXT               : 透传参数, 详情参见 OpsTest_Level1_AddOpUTestShared 函数说明
      TARGET_PRIVATE_LINK_LIBRARIES_EXT         : 透传参数, 详情参见 OpsTest_Level1_AddOpUTestShared 函数说明

]]
function(OpsTest_Level2_AddOp)
    cmake_parse_arguments(
            TMP
            ""
            "SUB_SYSTEM;BRIEF;SNAKE"
            "OPAPI_SOURCES_EXT;OPAPI_PRIVATE_INCLUDES_EXT;OPAPI_PRIVATE_LINK_LIBRARIES_EXT;PROTO_SOURCES_EXT;PROTO_PRIVATE_INCLUDES_EXT;PROTO_PRIVATE_LINK_LIBRARIES_EXT;TILING_SOURCES_EXT;TILING_PRIVATE_INCLUDES_EXT;TILING_PRIVATE_LINK_LIBRARIES_EXT;KERNEL_SOURCES_EXT;KERNEL_TILING_DATA_DEF_H;KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT;TARGET_SOURCES_EXT;TARGET_PRIVATE_INCLUDES_EXT;TARGET_PRIVATE_LINK_LIBRARIES_EXT"
            ""
            ${ARGN}
    )

    OpsTest_Level1_AddOpOpApiShared(
            SUB_SYSTEM                  ${TMP_SUB_SYSTEM}
            BRIEF                       ${TMP_BRIEF}
            SNAKE                       ${TMP_SNAKE}
            SOURCES_EXT                 ${TMP_OPAPI_SOURCES_EXT}
            PRIVATE_INCLUDES_EXT        ${TMP_OPAPI_PRIVATE_INCLUDES_EXT}
            PRIVATE_LINK_LIBRARIES_EXT  ${TMP_OPAPI_PRIVATE_LINK_LIBRARIES_EXT}
    )
    OpsTest_Level1_AddOpOpProtoShared(
            SUB_SYSTEM                  ${TMP_SUB_SYSTEM}
            BRIEF                       ${TMP_BRIEF}
            SNAKE                       ${TMP_SNAKE}
            SOURCES_EXT                 ${TMP_PROTO_SOURCES_EXT}
            PRIVATE_INCLUDES_EXT        ${TMP_PROTO_PRIVATE_INCLUDES_EXT}
            PRIVATE_LINK_LIBRARIES_EXT  ${TMP_PROTO_PRIVATE_LINK_LIBRARIES_EXT}
    )
    OpsTest_Level1_AddOpTilingShared(
            SUB_SYSTEM                  ${TMP_SUB_SYSTEM}
            BRIEF                       ${TMP_BRIEF}
            SNAKE                       ${TMP_SNAKE}
            SOURCES_EXT                 ${TMP_TILING_SOURCES_EXT}
            PRIVATE_INCLUDES_EXT        ${TMP_TILING_PRIVATE_INCLUDES_EXT}
            PRIVATE_LINK_LIBRARIES_EXT  ${TMP_TILING_PRIVATE_LINK_LIBRARIES_EXT}
    )
    OpsTest_Level1_AddOpKernelShared(
            SUB_SYSTEM                       ${TMP_SUB_SYSTEM}
            BRIEF                            ${TMP_BRIEF}
            SNAKE                            ${TMP_SNAKE}
            SOURCES_EXT                      ${TMP_KERNEL_SOURCES_EXT}
            TILING_DATA_DEF_H                ${TMP_KERNEL_TILING_DATA_DEF_H}
            PRIVATE_COMPILE_DEFINITIONS_EXT  ${TMP_KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT}
    )
    OpsTest_Level1_AddOpUTestShared(
            BRIEF                       ${TMP_BRIEF}
            SNAKE                       ${TMP_SNAKE}
            SOURCES_EXT                 ${TMP_TARGET_SOURCES_EXT}
            PRIVATE_INCLUDES_EXT        ${TMP_TARGET_PRIVATE_INCLUDES_EXT}
            PRIVATE_LINK_LIBRARIES_EXT  ${TMP_TARGET_PRIVATE_LINK_LIBRARIES_EXT}
    )
endfunction()

# 生成包含多个算子的 UT 可执行程序
#[[
]]
function(OpsTestUt_AddLaunch)
    if (NOT _OpsTestUt_OpsUtestLibraries)
        # 当 _OpsTestUt_OpsUtestLibraries 为空时不会生成 ops_test_utest
        # 此时说明 区分领域的 TESTS_UT_OPS_TEST 选项被错误设置, 需排查对应触发 ut 的 python 脚本逻辑及 build.sh.
        message(FATAL_ERROR "_OpsTestUt_OpsUtestLibraries empty.")
    endif ()

    # OpApi 动态库
    list(REMOVE_DUPLICATES _OpsTestUt_OpApiSources)
    list(REMOVE_DUPLICATES _OpsTestUt_OpApiPrivateIncludesExt)
    list(REMOVE_DUPLICATES _OpsTestUt_OpApiLinkLibrariesExt)
    if (NOT "${_OpsTestUt_OpApiSources}" STREQUAL "")
        set(_Target ${UTest_NamePrefix}_OpApi)
        add_library(${_Target} SHARED)
        target_sources(${_Target} PRIVATE ${_OpsTestUt_OpApiSources})
        target_include_directories(${_Target} PRIVATE
                ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
                ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn_kernels
                ${_OpsTestUt_OpApiPrivateIncludesExt}
        )
        target_compile_options(${_Target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
        )
        target_compile_definitions(${_Target} PRIVATE
                ACLNN_LOG_FMT_CHECK
                LOG_CPP
                PROCESS_LOG
        )
        target_link_libraries(${_Target} PRIVATE
                -Wl,--whole-archive
                ${_OpsTestUt_OpApiLinkLibrariesExt}
                $<BUILD_INTERFACE:intf_utest_strict_compile_options>
                $<BUILD_INTERFACE:intf_pub_utest>
                -Wl,--no-whole-archive
                -lopapi
                nnopbase
                profapi
                ge_common_base
                ascend_dump
                ascendalog
                dl
        )
    endif ()

    # OpProto 动态库
    list(REMOVE_DUPLICATES _OpsTestUt_OpProtoSources)
    list(REMOVE_DUPLICATES _OpsTestUt_OpProtoPrivateIncludesExt)
    list(REMOVE_DUPLICATES _OpsTestUt_OpProtoLinkLibrariesExt)
    if (NOT "${_OpsTestUt_OpProtoSources}" STREQUAL "")
        set(_Target ${UTest_NamePrefix}_OpProto)
        add_library(${_Target} SHARED)
        target_sources(${_Target} PRIVATE ${_OpsTestUt_OpProtoSources})
        target_include_directories(${_Target}
                PRIVATE
                    ${_OpsTestUt_OpProtoPrivateIncludesExt}
        )
        target_compile_options(${_Target}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
        )
        target_compile_definitions(${_Target}
                PRIVATE
                    LOG_CPP
                    PROCESS_LOG
        )
        target_link_libraries(${_Target}
                PRIVATE
                    -Wl,--whole-archive
                    ${_OpsTestUt_OpProtoLinkLibrariesExt}
                    $<BUILD_INTERFACE:intf_utest_strict_compile_options>
                    $<BUILD_INTERFACE:intf_pub_utest>
                    $<BUILD_INTERFACE:ops_utils_proto_headers>
                    -Wl,--no-whole-archive
                    ascendalog
    )
    endif ()

    # Tiling 动态库
    list(REMOVE_DUPLICATES _OpsTestUt_TilingSources)
    list(REMOVE_DUPLICATES _OpsTestUt_TilingPrivateIncludesExt)
    list(REMOVE_DUPLICATES _OpsTestUt_TilingLinkLibrariesExt)
    list(APPEND _OpsTestUt_TilingSources ${OPS_ADV_UTEST_OPS_TEST_DIR}/stubs/tiling/tiling_templates_registry.cpp)
    set(_Target ${UTest_NamePrefix}_OpTiling)
    add_library(${_Target} SHARED)
    target_sources(${_Target} PRIVATE ${_OpsTestUt_TilingSources})
    target_include_directories(${_Target} PRIVATE
            ${_OpsTestUt_TilingPrivateIncludesExt}
    )
    target_compile_definitions(${_Target} PRIVATE
            OP_TILING_LIB
            LOG_CPP
            PROCESS_LOG
    )
    target_compile_options(${_Target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
    )
    target_link_libraries(${_Target} PRIVATE
            -Wl,--as-needed
            -Wl,--no-whole-archive
            ${_OpsTestUt_TilingLinkLibrariesExt}
            $<BUILD_INTERFACE:intf_utest_strict_compile_options>
            $<BUILD_INTERFACE:intf_pub_utest>
            $<BUILD_INTERFACE:ops_utils_tiling_headers>
            graph
            graph_base
            exe_graph
            platform
            register
            ascendalog
            tiling_api
            c_sec
    )

    # 可执行程序
    set(_Target ops_test_utest)
    add_executable(${_Target})
    target_sources(${_Target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    target_compile_options(${_Target} PRIVATE -fPIC)
    target_link_libraries(${_Target} PRIVATE
            -Wl,--no-as-needed
            -Wl,--whole-archive
            GTest::gtest
            ${_OpsTestUt_OpsUtestLibraries}
            ${UTest_NamePrefix}_Stubs
            ${UTest_NamePrefix}_OpApi
            -Wl,--as-needed
            -Wl,--no-whole-archive
            c_sec
            $<BUILD_INTERFACE:intf_utest_strict_compile_options>
            $<BUILD_INTERFACE:intf_pub_utest>
            ${UTest_NamePrefix}_Utils
            ${UTest_NamePrefix}_OpTiling
    )

    # 执行用例
    if (NOT UT_NO_EXEC)
        if (ENABLE_ASAN)
            # 谨慎修改 ASAN_OPTIONS_ 取值, 当前出现 ASAN 告警会使 UT 失败.
            set(LD_PRELOAD_ "LD_PRELOAD=${ASAN_SHARED_PATH}:${STDC_SHARED_PATH}")
            set(ASAN_OPTIONS_ "ASAN_OPTIONS=halt_on_error=0")
            message(STATUS "${LD_PRELOAD_}")
            message(STATUS "${ASAN_OPTIONS_}")
            # 用例执行
            add_custom_command(
                    TARGET ops_test_utest POST_BUILD
                    COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${ASCEND_CANN_PACKAGE_PATH}/compiler/lib64:${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/simulator/Ascend910B1/lib && export ${LD_PRELOAD_} && ulimit -s 32768 && ${ASAN_OPTIONS_} ./ops_test_utest
                    COMMENT "Run ops_test_utest with asan"
            )
        else()
            # 用例执行
            add_custom_command(
                    TARGET ops_test_utest POST_BUILD
                    COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} && ./ops_test_utest
                    COMMENT "Run ops_test_utest"
            )
        endif ()

        if (ENABLE_GCOV)
            find_program(LCOV lcov REQUIRED)
            get_filename_component(GEN_COV_PY ${OPS_ADV_CMAKE_DIR}/scripts/utest/gen_coverage.py REALPATH)
            get_filename_component(ASCEND_CANN_PACKAGE_PATH_PARENT "${ASCEND_CANN_PACKAGE_PATH}/../" REALPATH)
            get_filename_component(GEM_COV_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}" REALPATH)
            # 获取 gcc 默认头文件搜索路径
            execute_process(
                    COMMAND ${CMAKE_C_COMPILER} --print-sysroot-headers-suffix
                    RESULT_VARIABLE _RST
                    OUTPUT_VARIABLE _SUFFIX
                    ERROR_QUIET
            )
            if (_RST)
                get_filename_component(SYS_ROOT "/usr/include" REALPATH)
            else ()
                get_filename_component(SYS_ROOT "${_SUFFIX}/usr/include" REALPATH)
            endif ()
            set(FILTER_DIRS
                    "-f=${SYS_ROOT}"
                    "-f=${GTEST_GTEST_INC}"
                    "-f=${OPS_ADV_UTEST_OPS_TEST_DIR}"
                    "-f=${ASCEND_CANN_PACKAGE_PATH_PARENT}"
            )
            add_custom_command(
                    TARGET ops_test_utest POST_BUILD
                    COMMAND ${HI_PYTHON} ${GEN_COV_PY} "-s=${OPS_ADV_DIR}" "-c=${GEM_COV_DATA_DIR}" ${FILTER_DIRS}
                    COMMENT "Generate gcov"
            )
        endif ()
    endif ()
endfunction()

# 添加算子UT路径
#[[
]]
function(OpsTestUt_AddSubdirectory)
    cmake_parse_arguments(
            TMP
            ""
            ""
            ""
            ${ARGN}
    )

    if ("${TESTS_UT_OPS_TEST}" STREQUAL "")
        return()
    elseif ("ALL" IN_LIST TESTS_UT_OPS_TEST OR "all" IN_LIST TESTS_UT_OPS_TEST)
        file(GLOB sub_dirs ${CMAKE_CURRENT_SOURCE_DIR}/*)
        foreach (_dir ${sub_dirs})
            if (IS_DIRECTORY ${_dir})
                add_subdirectory(${_dir})
            endif ()
        endforeach ()
    else ()
        set(_added_op_type_list)
        foreach (_op_type ${TESTS_UT_OPS_TEST})
            if (DEFINED ${_op_type}_alias)
                set(_op_type ${${_op_type}_alias})
            endif ()
            if (NOT "${_op_type}" IN_LIST _added_op_type_list)
                set(_dir ${CMAKE_CURRENT_SOURCE_DIR}/${_op_type})
                if (IS_DIRECTORY ${_dir})
                    add_subdirectory(${_dir})
                    list(APPEND _added_op_type_list ${_op_type})
                endif ()
            endif ()
        endforeach ()
    endif ()
endfunction()
