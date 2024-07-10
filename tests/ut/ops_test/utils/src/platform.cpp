/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file platform.cpp
 * \brief 提供平台相关接口的打桩及辅助功能.
 */

#include <dlfcn.h>
#include <map>
#include "tests/utils/platform.h"
#include "tests/utils/log.h"

namespace {
ops::adv::tests::utils::Platform *g_PlatformPtr = nullptr;
} // namespace

namespace ops::adv::tests::utils {

bool Platform::SocSpec::Get(const char *label, const char *key, std::any &value) const
{
    auto kvPair = spec.find(label);
    if (kvPair == spec.end()) {
        LOG_ERR("Unknown Spec(%s:%s)'s label(%s).", label, key, label);
        return false;
    }
    auto val = kvPair->second.find(key);
    if (val == kvPair->second.end()) {
        LOG_ERR("Unknown Spec(%s:%s)'s key(%s).", label, key, key);
        return false;
    }
    value = val->second;
    return true;
}

bool Platform::SocSpec::Get(const char *label, const char *key, std::string &value) const
{
    std::any any;
    if (this->Get(label, key, any)) {
        if (any.type() == typeid(std::string)) {
            value = std::any_cast<std::string>(any);
        } else if (any.type() == typeid(uint32_t)) {
            value = std::to_string(std::any_cast<uint32_t>(any));
        } else if (any.type() == typeid(uint64_t)) {
            value = std::to_string(std::any_cast<uint64_t>(any));
        } else {
            LOG_ERR("Unknown Spec(%s:%s)'s dtype(%s).", label, key, any.type().name());
            return false;
        }
        return true;
    }
    return false;
}

bool Platform::SocSpec::Get(const char *label, const char *key, uint32_t &value) const
{
    std::any any;
    if (this->Get(label, key, any)) {
        if (any.type() == typeid(uint32_t)) {
            value = std::any_cast<uint32_t>(any);
        } else {
            LOG_ERR("Unknown Spec(%s:%s)'s dtype(%s).", label, key, any.type().name());
            return false;
        }
        return true;
    }
    return false;
}

bool Platform::SocSpec::Get(const char *label, const char *key, uint64_t &value) const
{
    std::any any;
    if (this->Get(label, key, any)) {
        if (any.type() == typeid(uint64_t)) {
            value = std::any_cast<uint64_t>(any);
        } else {
            LOG_ERR("Unknown Spec(%s:%s)'s dtype(%s).", label, key, any.type().name());
            return false;
        }
        return true;
    }
    return false;
}

void Platform::SetGlobalPlatform(Platform *platform)
{
    g_PlatformPtr = platform;
}

Platform *Platform::GetGlobalPlatform()
{
    return g_PlatformPtr;
}

bool Platform::InitArgsInfo(int argc, char **argv)
{
    /**
     * 解析输入, 获取 current work directory
     */
    if (argc < 1 || argv == nullptr || argv[0] == nullptr) {
        LOG_ERR("Environment error, Argc=%d, Argv=%p, Argv[0]=%p", argc, argv, argv == nullptr ? nullptr : argv[0]);
        return false;
    }
    std::string exeFile(argv[0]);
    exeAbsPath_ = std::string(exeFile.substr(0, exeFile.rfind('/')) + "/");
    return true;
}

const char *Platform::GetExeAbsPath()
{
    return exeAbsPath_.c_str();
}

Platform &Platform::SetSocVersion(const SocVersion &socVersion)
{
    switch (socVersion) {
        case SocVersion::Ascend910B1:
            socSpec.spec = {
                {
                    "version",
                    {
                        {"SoC_version", std::string("Ascend910B1")},
                        {"Short_SoC_version", std::string("Ascend910B")},
                    },
                },
                {
                    "SoCInfo",
                    {
                        {"core_type_list", std::string("CubeCore,VectorCore")},
                        {"ai_core_cnt", std::uint32_t(24)},
                        {"cube_core_cnt", std::uint32_t(24)},
                        {"vector_core_cnt", std::uint32_t(48)},
                    },
                },
                {
                    "AICoreSpec",
                    {
                        {"ubblock_size", std::uint32_t(32)},
                    },
                },
                {
                    "DtypeMKN",
                    {
                        {"Default", std::string("16,16,16")},
                    },
                },
                {
                    "LocalMemSize",
                    {
                        {"0", std::uint64_t(65536)},    /* L0_A */
                        {"1", std::uint64_t(65536)},    /* L0_B */
                        {"2", std::uint64_t(131072)},   /* L0_C */
                        {"3", std::uint64_t(524288)},   /* L1 */
                        {"4", std::uint64_t(33554432)}, /* L2 */
                        {"5", std::uint64_t(196608)},   /* UB */
                        {"6", std::uint64_t(0)},        /* HBM */
                    },
                },
                {
                    "LocalMemBw",
                    {
                        {"4", std::uint64_t(110)}, /* L2 */
                        {"6", std::uint64_t(32)},  /* HBM */
                    },
                },
            };
            break;
        case SocVersion::Ascend910B2:
            socSpec.spec = {
                {
                    "version",
                    {
                        {"SoC_version", std::string("Ascend910B2")},
                        {"Short_SoC_version", std::string("Ascend910B")},
                    },
                },
                {
                    "SoCInfo",
                    {
                        {"core_type_list", std::string("CubeCore,VectorCore")},
                        {"ai_core_cnt", std::uint32_t(24)},
                        {"cube_core_cnt", std::uint32_t(24)},
                        {"vector_core_cnt", std::uint32_t(48)},
                    },
                },
                {
                    "AICoreSpec",
                    {
                        {"ubblock_size", std::uint32_t(32)},
                    },
                },
                {
                    "DtypeMKN",
                    {
                        {"Default", std::string("16,16,16")},
                    },
                },
                {
                    "LocalMemSize",
                    {
                        {"0", std::uint64_t(65536)},    /* L0_A */
                        {"1", std::uint64_t(65536)},    /* L0_B */
                        {"2", std::uint64_t(131072)},   /* L0_C */
                        {"3", std::uint64_t(524288)},   /* L1 */
                        {"4", std::uint64_t(33554432)}, /* L2 */
                        {"5", std::uint64_t(196608)},   /* UB */
                        {"6", std::uint64_t(0)},        /* HBM */
                    },
                },
                {
                    "LocalMemBw",
                    {
                        {"4", std::uint64_t(110)}, /* L2 */
                        {"6", std::uint64_t(32)},  /* HBM */
                    },
                },
            };
            break;
        case SocVersion::Ascend910B3:
            socSpec.spec = {
                {
                    "version",
                    {
                        {"SoC_version", std::string("Ascend910B3")},
                        {"Short_SoC_version", std::string("Ascend910B")},
                    },
                },
                {
                    "SoCInfo",
                    {
                        {"core_type_list", std::string("CubeCore,VectorCore")},
                        {"ai_core_cnt", std::uint32_t(20)},
                        {"cube_core_cnt", std::uint32_t(20)},
                        {"vector_core_cnt", std::uint32_t(40)},
                    },
                },
                {
                    "AICoreSpec",
                    {
                        {"ubblock_size", std::uint32_t(32)},
                    },
                },
                {
                    "DtypeMKN",
                    {
                        {"Default", std::string("16,16,16")},
                    },
                },
                {
                    "LocalMemSize",
                    {
                        {"0", std::uint64_t(65536)},    /* L0_A */
                        {"1", std::uint64_t(65536)},    /* L0_B */
                        {"2", std::uint64_t(131072)},   /* L0_C */
                        {"3", std::uint64_t(524288)},   /* L1 */
                        {"4", std::uint64_t(33554432)}, /* L2 */
                        {"5", std::uint64_t(196608)},   /* UB */
                        {"6", std::uint64_t(0)},        /* HBM */
                    },
                },
                {
                    "LocalMemBw",
                    {
                        {"4", std::uint64_t(110)}, /* L2 */
                        {"6", std::uint64_t(32)},  /* HBM */
                    },
                },
            };
            break;
        default:
            break;
    }
    return *this;
}

uint32_t Platform::GetCoreNum() const
{
    uint32_t coreNum = 0;
    return socSpec.Get("SoCInfo", "vector_core_cnt", coreNum) ? coreNum : 0;
}

int64_t Platform::GetBlockDim() const
{
    auto blockDim = static_cast<int64_t>(this->GetCoreNum()) / 2;
    return blockDim;
}

void *Platform::LoadSo(const char *absPath)
{
    if (absPath == nullptr) {
        LOG_ERR("File nullptr.");
        return nullptr;
    }
    auto soHdl = dlopen(absPath, RTLD_NOW | RTLD_GLOBAL);
    if (soHdl == nullptr) {
        LOG_ERR("Can't dlopen %s, %s", absPath, dlerror());
        return nullptr;
    }
    return soHdl;
}

bool Platform::UnLoadSo(void *hdl)
{
    if (dlclose(hdl) != 0) {
        LOG_ERR("dlclose error %s", dlerror());
        return false;
    }
    return true;
}

void *Platform::LoadSoSym(void *hdl, const char *name)
{
    if (hdl == nullptr) {
        LOG_ERR("handle nullptr");
        return nullptr;
    }
    return dlsym(hdl, name);
}

bool Platform::LoadTilingSo(const char *relPath)
{
    std::string absPath = exeAbsPath_ + relPath;
    tilingSoHdl_ = this->LoadSo(absPath.c_str());
    return tilingSoHdl_ != nullptr;
}

bool Platform::UnLoadTilingSo()
{
    auto rst = this->UnLoadSo(this->tilingSoHdl_);
    this->tilingSoHdl_ = nullptr;
    return rst;
}

void *Platform::LoadTilingSoSym(const char *name)
{
    return this->LoadSoSym(tilingSoHdl_, name);
}

bool Platform::LoadProtoSo(const char *relPath)
{
    std::string absPath = exeAbsPath_ + relPath;
    protoSoHdl_ = this->LoadSo(absPath.c_str());
    return protoSoHdl_ != nullptr;
}

bool Platform::UnLoadProtoSo()
{
    auto rst = this->UnLoadSo(this->protoSoHdl_);
    this->protoSoHdl_ = nullptr;
    return rst;
}

} // namespace ops::adv::tests::utils
