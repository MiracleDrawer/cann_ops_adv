# cann-ops-adv（融合算子）


## 概述

cann-ops-adv，是基于昇腾硬件的融合算子库（adv表示advanced）。

融合算子是指将多个独立的“小算子”融合成一个“大算子”，多个小算子的功能和大算子的功能等价，融合算子在性能或者内存等方面优于独立的小算子。可以根据具体算法的实现自由融合向量（Vector）、矩阵（Cube）算子以达到性能或者内存收益。

例如对于大语言模型（Large Language Model, LLM）中最核心的一个融合算子Flash Attention，其核心实现如下图，图中Matmul算子（Cube）、Scale算子（Vector）、Mask算子（Vector）、SoftMax算子（Vector）、flash update算子（Vector）融合为Flash Attention融合算子。

![原理图](./docs/fig/readme图.png)

融合算子通常有如下优势：

- 减少计算量：融合算子可以将多个算子合并为一个，简化计算过程，减少计算量，提高计算效率。
- 减少内存占用：融合算子可以将多个算子的中间结果合并为一个，从而减少内存占用，提高内存利用率。
- 优化数据流：融合算子可以优化数据流，减少数据在不同算子之间的传输，从而提高数据处理效率。
- 简化代码实现：融合算子可以简化代码实现，减少代码量，提高代码可读性和可维护性。


## 目录结构说明

融合算子代码目录结构如下：

  ```
  ├── docs                                           # 算子使用说明和资料
  ├── examples                                       # 所有算子的使用示例代码
  |   ├── transformer                                # 大模型算子
  |       ├── flash_attention_score                  # 训练FA算子示例代码
  |       ├── flash_attention_score_grad             # 训练FAG算子示例代码
  |       ├── incre_flash_attention                  # 推理IFA算子示例代码
  |       ├── prompt_flash_attention                 # 推理PFA算子示例代码
  |       ├── fused_infer_attention_score            # 推理FIA算子示例代码
  |       ├── ffn                                    # 推理FFN算子示例代码
  |
  ├── src                                            # 所有算子的源代码
  |   ├── transformer                                # 大模型算子
  |   |   ├── flash_attention_score                  # 训练FA算子源代码
  |   |   |   ├── ophost                             # ophost目录，包含tiling策略、aclnn接口、算子原型、信息库配置
  |   |   |   ├── flash_attention_score*.*           # FA算子kernel源文件
  |   |   ├── flash_attention_score_grad             # 训练FAG算子源代码
  |   |   |   ├── ophost                             # ophost目录，包含tiling策略、aclnn接口、算子原型、信息库配置
  |   |   |   ├── flash_attention_score_grad*.*      # FAG算子kernel源文件
  |   |   ├── incre_flash_attention                  # 推理IFA算子源代码
  |   |   |   ├── ophost                             # ophost目录，包含tiling策略、aclnn接口、算子原型、信息库配置
  |   |   |   ├── incre_flash_attention*.*           # IFA算子kernel源文件
  |   |   ├── prompt_flash_attention                 # 推理PFA算子源代码
  |   |   |   ├── ophost                             # ophost目录，包含tiling策略、aclnn接口、算子原型、信息库配置
  |   |   |   ├── prompt_flash_attention*.*          # PFA算子Kernel源文件
  |   |   ├── fused_infer_attention_score            # 推理FIA算子源代码
  |   |   |   ├── ophost                             # ophost目录，包含tiling策略、aclnn接口、算子原型、信息库配置
  |   |   |   ├── fused_infer_attention_score*.*     # FIA算子Kernel源文件
  |   |   ├── ffn                                    # 推理FFN算子源代码
  |   |       ├── ophost                             # ophost目录，包含tiling策略、aclnn接口、算子原型、信息库配置
  |   |       ├── ffn*.*                             # FFN算子Kernel源文件
  |   |
  |   ├── utils                                      # 所有算子用到的公共接口
  |       ├── inc                                    # 公共头文件
  |       |   ├── error                              # 错误码上报头文件
  |       |   ├── log                                # 日志头文件
  |       |   ├── tiling                             # 公共tiling头文件
  |       ├── src                                    # 公共接口源代码
  ├── tests
      ├── ut                                         # 算子UT用例
  ```

## 融合算子列表


| 算子名                     | 概述                                                         | 实现接口                                                     |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| FlashAttentionScore        | 使用FlashAttention算法实现self-attention（自注意力）的计算。 | <li>[FlashAttentionScore](./docs/FlashAttentionScore.md)<br> <li>[FlashAttentionVarLenScore](./docs/FlashAttentionVarLenScore.md) |
| FlashAttentionScoreV2      | 训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。**相较于FlashAttentionScore，新增psetype、q_start_idx、kv_start_idx参数**。 | <li>[FlashAttentionScoreV2](./docs/FlashAttentionScoreV2.md)<br/> <li>[FlashAttentionVarLenScoreV2](./docs/FlashAttentionVarLenScoreV2.md) |
| FlashAttentionScoreGrad    | 完成FlashAttentionScore算子的反向计算。                      | <li>[FlashAttentionScoreGrad](./docs/FlashAttentionScoreGrad.md) <br> <li>[FlashAttentionUnpaddingScoreGrad](./docs/FlashAttentionUnpaddingScoreGrad.md) |
| FlashAttentionScoreGradV2  | FlashAttentionScoreV2的反向计算，**相较于FlashAttentionScoreGard，新增psetype、q_start_idx、kv_start_idx参数**。 | <li>[FlashAttentionScoreGradV2](./docs/FlashAttentionScoreGradV2.md) <br/> <li>[FlashAttentionUnpaddingScoreGradV2](./docs/FlashAttentionUnpaddingScoreGradV2.md) |
| FFN                        | 将Transformer网络结构中的FFN融合成一个算子进行计算。         | [FFN](./docs/FFN.md)                                         |
| FFNV2                      | 相较于FFN，新增了tokensIndexFlag参数，以支持输入expertTokens为索引值。 | [FFNV2](./docs/FFNV2.md)                                     |
| FFNV3                      | 相较于FFNV2，expertTokens从aclIntArray指针输入改为了aclTensor指针输入。 | [FFNV3](./docs/FFNV3.md)                                     |
| PromptFlashAttention       | 使用FlashAttention算法实现self-attention（自注意力）的计算。 | [PromptFlashAttention](./docs/PromptFlashAttention.md)       |
| PromptFlashAttentionV2     | 相较于PromptFlashAttention新增量化特性、sparse特性、指定key/value的有效Sequence Length特性                   | [PromptFlashAttentionV2](./docs/PromptFlashAttentionV2.md)   |
| PromptFlashAttentionV3     | 相较于PromptFlashAttentionV2新增支持指定精度模式特性    | [PromptFlashAttentionV3](./docs/PromptFlashAttentionV3.md)   |
| IncreFlashAttention        | 使用FlashAttention算法实现self-attention（自注意力）的计算。 | [IncreFlashAttention](./docs/IncreFlashAttention.md)         |
| IncreFlashAttentionV2      | 在IncreFlashAttention基础上新增量化特性                      | [IncreFlashAttentionV2](./docs/IncreFlashAttentionV2.md)     |
| IncreFlashAttentionV3      | 在IncreFlashAttentionV2基础上新增位置编码，page attention，kv cache反量化特性 | [IncreFlashAttentionV3](./docs/IncreFlashAttentionV3.md)     |
| IncreFlashAttentionV4      | 在IncreFlashAttentionV3基础上新增kv左Padding特性             | [IncreFlashAttentionV4](./docs/IncreFlashAttentionV4.md)     |
| FusedInferAttentionScore   | 融合PromptFlashAttentionV3，IncreFlashAttentionV4的功能 <br>IFA新增:  lse输出，per-token伪量化特性<br>PFA新增: lse输出，伪量化，左Padding特性 | [FusedInferAttentionScores](./docs/FusedInferAttentionScore.md) |
| FusedInferAttentionScoreV2 | 在FusedInferAttentionScore基础上， IFA 新增kv伪量化参数分离  | [FusedInferAttentionScoreV2](./docs/FusedInferAttentionScoreV2.md) |

以上算子当前支持Atlas A2 训练系列产品。
## 环境准备<a name="1"></a>

cann-ops-adv支持由源码编译，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **获取CANN软件包**

   请从[Link](https://www.hiascend.com/developer/download/community/result?module=cann)获取配套版本的CANN开发套件包Ascend-cann-toolkit\_\<cann_version>\_linux\-\<arch>.run和算子二进制包Ascend-cann-kernels-\<soc_version>_\<cann_version>\_linux.run（二进制包，算子运行时依赖），开放项目与CANN社区版本的配套关系可参见[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)。

   支持的操作系统请参见对应[用户手册](https://hiascend.com/document/redirect/CannCommunityInstSoftware)的“支持的操作系统”章节。**注意：请选择配套版本的软件包与文档**。

2. **安装依赖**

   **以下所列仅为cann-ops-adv源码编译用到的依赖，python、gcc、cmake以及CANN开发套件包其他依赖的安装方法请参见配套版本[用户手册](https://hiascend.com/document/redirect/CannCommunityInstDepend)的“安装依赖”章节。**

   - python >= 3.7.0

   - gcc >= 7.3.0

   - cmake >= 3.16.0

   - protobuf <=3.20.x

     算子编译时，protobuf版本需低于3.20.x，您可以执行**pip3 list**命令查询当前环境中的protobuf版本，如果版本高于3.20.x，则执行如下命令重新安装，以重新安装3.20.0版本为例：

     ```bash
     pip3 install protobuf==3.20.0
     ```

     如果使用非root用户安装，需要在安装命令后加上--user，例如**pip3 install protobuf==3.20.0 --user**。

   - googletest（可选，仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

     如下以[googletest源码](https://github.com/google/googletest.git)编译安装为例，安装命令如下：

     ```bash
     mkdir temp && cd temp                 # 在googletest源码根目录下创建临时目录并进入
     cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
     make
     make install                         # root用户安装googletest
     # sudo make install                  # 非root用户安装googletest
     ```

3. **安装CANN软件包**

   执行安装命令时，请确保安装用户对软件包具有可执行权限。

   - 使用默认路径安装

     ```bash
     # CANN开发套件包安装命令示例：
     ./Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install
     # 算子二进制包安装命令示例：
     ./Ascend-cann-kernels-<soc_version>_<cann_version>_linux.run --install
     ```

     - 若使用root用户安装，安装完成后CANN开发套件包存储在`/usr/local/Ascend/ascend-toolkit/latest`路径；算子二进制包存储在`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`路径。
     - 若使用非root用户安装，安装完成后CANN开发套件包存储在`$HOME/Ascend/ascend-toolkit/latest`路径；算子二进制包存储在`${HOME}/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`路径。

   - 指定路径安装

     ```bash
     # CANN开发套件包安装命令示例：
     ./Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install --install-path=${install_path}
     # 算子二进制包安装命令示例：
     ./Ascend-cann-kernels-<soc_version>_<cann_version>_linux.run --install --install-path=${install_path}
     ```

     安装完成后，CANN开发套件包存储在\${install_path}指定路径；算子二进制包存储在`${install_path}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`路径。

4. **设置环境变量**

   - 默认路径，root用户安装

     ```bash
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     ```

   - 默认路径，非root用户安装

     ```bash
     source $HOME/Ascend/ascend-toolkit/set_env.sh
     ```

   - 指定路径安装

     ```bash
     source ${install_path}/ascend-toolkit/set_env.sh
     ```

   **注意：若环境中已安装多个版本的CANN软件包，设置上述环境变量时，请确保${install_path}/ascend-toolkit/latest目录指向的是配套版本的软件包。**

## 源码下载

执行如下命令，下载cann-ops-adv仓源码：

  ```bash
  git clone https://gitee.com/ascend/cann-ops-adv.git
  ```


## 编译执行

### 自定义算子包编译

  ```bash
  cd cann-ops-adv
  mkdir build && cd build     # 在融合算子源码根目录下创建临时目录并进入
  cmake ..
  make package -j 并发数      # 编译并生成自定义算子run包，并发数请替换为实际取值
  ```

**说明：**

编译时间较长，请耐心等待：在无缓存场景，使用72核编译器，**-j 144**并发执行，编译大约耗时13分钟。您可以通过**grep 'processor' /proc/cpuinfo | wc -l**命令查询当前服务器cpu核数，并发数=cpu核数*2。

若提示如下信息，则说明编译成功。

  ```
  Self-extractable archive "CANN-custom_ops-<cann_version>-linux.<arch>.run" successfully created.
  ```

编译成功后在 `cann-ops-adv/output` 目录生成自定义算子包：CANN-custom_ops-\<cann_version>-linux.\<arch>.run。

### 自定义算子包安装<a name="2"></a>

安装前，需确保所安装的自定义算子包与所安装CANN开发套件包CPU架构一致，并且要先设置CANN开发套件包环境变量，然后再进行安装，仅支持在配套版本安装自定义算子包，安装命令如下：

  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh # 设置CANN开发套件包环境变量，以root用户默认路径为例，如已设置，则请忽略该操作
  ./CANN-custom_ops-<cann_version>-linux.<arch>.run --quiet         # 安装自定义算子run包
  ```

执行上述命令后，自定义算子run包会默认安装到CANN软件包目录，例如，`/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/` 目录。

### 单元测试编译执行

UT（单元测试用例），用来看护编译是否正常，依次执行如下命令：

  ```bash
  cd cann-ops-adv
  mkdir build && cd build             # 在融合算子源码根目录下创建临时目录并进入
  cmake .. -DTESTS_UT_OPS_TEST=ALL    # 指定编译所有融合算子的单元测试用例
  make ops_test_utest -j 并发数        # 编译并执行所有融合算子的单元测试用例，并发数请替换为实际取值
  ```

执行UT用例依赖googletest单元测试框架，关于googletest更多功能请参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

### 示例工程编译执行

此操作需要在真实NPU环境上进行，并且依赖CANN开发套件包和算子二进制包，因此在编译前，需要参见[环境准备](#1)章节安装配套版本的CANN开发套件包和算子二进制包，并设置环境变量。

  ```bash
  cd cann-ops-adv
  mkdir build && cd build                 # 在融合算子源码根目录下创建临时目录并进入
  cmake .. -DTESTS_EXAMPLE_OPS_TEST=ALL   # 指定编译所有examples用例示例
  make                                    # 编译并执行所有examples用例
  ```

上述cmake编译参数详细解释请参见[cmake编译参数说明](./docs/common/cmake编译参数说明.md)。

**说明**：当前还提供了一键式编译脚本，命令如下，执行完自定义算子包一键式编译命令，需要完成[自定义算子包安装](#2)后，才能执行后续命令。

  ```bash
  cd cann-ops-adv
  bash build.sh         #自定义算子包编译
  bash build.sh -t      #单元测试编译执行
  bash build.sh -e      #示例工程编译执行
  ```

## 开发指导

### 融合算子源码导读

学习一个全新的代码库并不是一件容易的事情，特别是CANN软件栈中的融合算子库，组件间以较为隐蔽的方式进行交互。在本导读中，我们尝试通过一个简单的、具体的示例来说明整个融合算子的编译、执行过程。对于每一个重要的步骤，我们都会给出其在代码中的具体位置，希望通过本导读，能够让新加入的开发人员和感兴趣的用户更快地、更深入地的理解融合算子代码库，单击[Link](./docs/common/融合算子源码导读.md)查看详情。

### 融合算子设计介绍

cann-ops-adv仓提供了如下融合算子的代码实现设计，方便开发人员更深入的理解融合算子：
- [FA/FAG算子设计介绍](./docs/common/FA-FAG算子设计介绍.md)
- [IFA算子设计介绍](./docs/common/IFA算子设计介绍.md)
- [PFA算子设计介绍](./docs/common/PromptFlashAttention算子设计介绍.md)
- [FFN算子设计介绍](./docs/common/FFN算子设计介绍.md)

## 贡献指南

cann-ops-adv仓欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitee.com/ascend/cann-community)了解行为准则，进行CLA协议签署，以及参与源码仓贡献的详细流程。

 针对cann-ops-adv仓，开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。

## 许可证

[CANN Open Software License Agreement Version 1.0](LICENSE)
