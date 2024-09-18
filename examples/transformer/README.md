## transformer目录文件介绍

```
├── ffn                        # FFN算子样例目录
|   ├── CMakeLists.txt         # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   ├── ffn_generate_data.py   # 样例数据的生成脚本，通过numpy生成随机数据，并保存到二进制文件中
|   ├── ffn_print_result.py    # 样例结果输出脚本，用于展示输出数据结果，对于float16格式的数据，避免引入复杂的c++依赖
|   ├── ffn_utils.h            # 样例代码中的公共部分，如创建aclTensor、读二进制输入数据、保存二进制输出、初始化npu资源和释放aclTensor
|   ├── test_ffn_v2.cpp        # FFNV2接口测试用例代码
|   ├── test_ffn_v3_quant.cpp  # FFNV3接口量化场景测试用例代码
|   ├── test_ffn_v3.cpp        # FFNV3接口非量化测试用例代码
|   ├── run_ffn_case.sh        # 执行CMakeLists.txt中配置的测试用例，由三个步骤组成：
|                                 1.执行ffn_generate_data.py生成输入数据二进制文件
|                                 2.执行CMakeLists.txt中配置编译的样例二进制程序
|                                 3.执行ffn_print_result.py输出测试结果
|
|—— fused_infer_attention_score       # FIA算子样例目录
|   |—— CMakeLists.txt                # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_fused_infer_attention_score_v2_ifa_antiquant.cpp        # FIA伪量化接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_leftpad.cpp          # FIA左padding接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_Lse.cpp              # FIA Lse接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_PA.cpp               # FIA page attention接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_postquant.cpp        # FIA后量化接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_system_prefix.cpp    # FIA prefix接口测试用例代码
|   |—— test_fused_infer_attention_score_v2.cpp                      # FIAV2接口测试用例代码
|   |—— test_fused_infer_attention_score.cpp                         # FIAV1接口测试用例代码
|   |—— run_fia_case.sh                                              # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|
|—— incre_flash_attention             # IFA算子样例目录
|   |—— CMakeLists.txt                # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_incre_flash_attention.cpp               # IFAV1接口测试用例代码
|   |—— test_incre_flash_attention_v2.cpp            # IFAV2接口测试用例代码
|   |—— test_incre_flash_attention_v3.cpp            # IFAV1接口测试用例代码
|   |—— test_incre_flash_attention_v4.cpp            # IFAV1接口测试用例代码
|   |—— run_ifa_case.sh                              # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|
```
