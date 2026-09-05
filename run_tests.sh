#!/bin/bash
# ==============================================================================
# FlexKV 测试运行脚本
# ------------------------------------------------------------------------------
# 作用：在跑 Python 测试之前，先把 FlexKV 编译产物（C++/CUDA 扩展的动态库）
#       以及 PyTorch 的库路径注入到 LD_LIBRARY_PATH，这样 `import flexkv` 时
#       Python 才能找到并加载底层的 .so 扩展模块，避免 "undefined symbol" 或
#       "libflexkv_c_ext.so: cannot open shared object file" 之类的错误。
#
# 用法：
#   ./run_tests.sh                      # 运行默认测试（见下方说明）
#   ./run_tests.sh tests/test_kvmanager.py
#   ./run_tests.sh -m pytest tests/ -v  # 第一个参数非空时，整行原样交给 python3
#
# 注意：本脚本只是"设置环境变量 + 转发命令"的薄封装，不负责编译。
#       若还没构建，请先执行 ./build.sh 生成 build/lib 下的动态库。
# ==============================================================================

# 获取脚本自身所在的目录（转成绝对路径）。
# 用 BASH_SOURCE[0] 而不是 $0：当脚本被 source 或经过符号链接调用时，
# $0 可能不是脚本真实路径，而 BASH_SOURCE[0] 始终指向脚本文件本身。
# 先 cd 进去再 pwd，是为了拿到规范化的绝对路径（不带 .. 或相对成分）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置动态库搜索路径（Linux 下动态链接器会按此顺序查找 .so）。
# 三个路径的含义：
#   1) ${SCRIPT_DIR}/build/lib
#      build.sh 编译产出的第三方库与 FlexKV 自身的动态库（如 libcudart 包装、
#      libflexkv*.so 等）。这是最关键的一项，缺失会导致扩展模块加载失败。
#   2) /usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs
#      CUDA 的 stubs 目录，里面是"桩"版 libcuda.so（符号齐全但函数为空实现）。
#      链接期用它代替真实的驱动库，可避免在没有 GPU / NVIDIA 驱动的机器上
#      因找不到 libcuda.so 而报错。运行时由驱动提供的真实库接管。
#   3) .../site-packages/torch/lib
#      PyTorch 自带的 libtorch.so、libtorch_cuda.so 等，
#      FlexKV 的 C++ 扩展部分依赖 torch 的 C++ ABI，需要一并暴露。
# 写法说明：把原有的 ${LD_LIBRARY_PATH} 追加在末尾，路径之间用冒号分隔，
# 这样不会覆盖调用者环境中已配置的其他库路径。
export LD_LIBRARY_PATH="${SCRIPT_DIR}/build/lib:/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs:/data/home/phaedonsun/.local/lib/python3.6/site-packages/torch/lib:${LD_LIBRARY_PATH}"
# 提示：上面第 2、3 项是写死的绝对路径（CUDA 11.8 + 某台机器上的 python3.6 环境），
# 换机器 / 换 CUDA 版本 / 换 Python 版本时需要改成你本机的真实路径，
# 否则这两项无效（不影响第 1 项生效，通常仍能正常跑测试）。

# 打印分隔线与当前生效的库路径，方便排查"库找不到"类问题时一眼确认。
echo "=========================================="
echo "FlexKV 测试运行脚本"
echo "=========================================="
echo "库路径已设置:"
echo "  - ${SCRIPT_DIR}/build/lib"
echo "  - /usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs"
echo "  - PyTorch 库路径"
echo ""

# 检查是否传入了测试文件参数：
# [ -z "$1" ] 判断第一个参数是否为空字符串（未传参）。
if [ -z "$1" ]; then
    # 未传参：运行内置默认测试（分布式 RadixTree 基础用例）。
    # 注意：这是用 `python3 文件` 直接执行脚本的方式，而不是通过 pytest 运行，
    # 因此该测试文件必须自带 `if __name__ == "__main__":` 入口才有效；
    # 另外 tests/ 目录下目前已不存在 test_dis_radixtree_basic.py，
    # 直接执行会报文件不存在。建议显式指定要跑的测试文件。
    echo "运行默认测试: tests/test_dis_radixtree_basic.py"
    python3 "${SCRIPT_DIR}/tests/test_dis_radixtree_basic.py"
else
    # 传了参数：把整条命令行原样转发给 python3。
    # "$@" 会保留每个参数的原始分词（带空格的参数不会被拆开），
    # 所以既可以传测试文件路径，也可以传 `-m pytest ...` 之类的组合。
    echo "运行测试: $1"
    python3 "$@"
fi

# 捕获上面测试命令的退出码（0 表示成功，非 0 表示失败或异常）。
# if/else 的退出码就是其中最后一条被执行命令的退出码，所以这里能正确拿到。
exit_code=$?

echo ""
echo "=========================================="
# 按退出码给出人可读的结论。
if [ $exit_code -eq 0 ]; then
    echo "✓ 测试完成 (退出码: $exit_code)"
else
    echo "✗ 测试失败 (退出码: $exit_code)"
fi
echo "=========================================="

# 把测试的退出码作为脚本自身的退出码返回，
# 这样在 CI 里 `./run_tests.sh && echo ok` 才能正确感知测试是否通过。
exit $exit_code
