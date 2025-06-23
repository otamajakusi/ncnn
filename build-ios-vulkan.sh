set -uex
mkdir -p build-ios-vulkan
cd build-ios-vulkan
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../toolchains/iosxc-arm64.toolchain.cmake \
    -DENABLE_BITCODE=OFF \
    -DVulkan_INCLUDE_DIR=${VULKAN_SDK}/iOS/include \
    -DVulkan_LIBRARY=${VULKAN_SDK}/macOS/lib/MoltenVK.xcframework/ios-arm64 \
    -DNCNN_VULKAN=ON \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install
make -j4
make install
