set -uex
mkdir -p build-ios-vulkan
cd build-ios-vulkan
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../toolchains/iosxc-arm64.toolchain.cmake \
    -DENABLE_BITCODE=OFF \
    -DVulkan_INCLUDE_DIR=${VULKAN_SDK}/MoltenVK/include \
    -DVulkan_LIBRARY=${VULKAN_SDK}/MoltenVK/iOS/MoltenVK.framework/MoltenVK \
    -DNCNN_VULKAN=ON \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install
make -j4
make install
