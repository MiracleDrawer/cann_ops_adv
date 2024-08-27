## transformer目录文件介绍

```
├── ffn                      # FFN算子样例目录
|   ├── CMakeLists.txt       # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   ├── ffn_generate_data.py # 样例数据的生成脚本，通过numpy生成随机数据，并保存到二进制文件中
|   ├── ffn_print_result.py  # 样例结果输出脚本，用于展示输出数据结果，对于float16格式的数据，避免引入复杂的c++依赖
|   ├── ffn_utils.h          # 样例代码中的公共部分，如创建aclTensor、读二进制输入数据、保存二进制输出、初始化npu资源和释放aclTensor
|   ├── test_ffn_v2.cpp      # FFNV2接口测试用例代码
|   ├── test_ffn_v3.cpp      # FFNV3接口测试用例代码
|   ├── run_ffn_case.sh      # 执行CMakeLists.txt中配置的测试用例，由三个步骤组成：
                               1.执行ffn_generate_data.py生成输入数据二进制文件
                               2.执行CMakeLists.txt中配置编译的样例二进制程序
                               3.执行ffn_print_result.py输出测试结果                           
```
