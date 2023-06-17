set -uex
mkdir -p build-ios-vulkan
cd build-ios-vulkan
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../toolchains/iosxc-arm64.toolchain.cmake \
    -DENABLE_BITCODE=OFF \
    -DVulkan_INCLUDE_DIR=${VULKAN_SDK}/MoltenVK/include \
    -DNCNN_VULKAN=ON \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DCMAKE_BUILD_TYPE=Release
make -j4
