# Standard settings
# set(UNIX True)
# set(Darwin True)
# set(IOS True)
set (CMAKE_SYSTEM_NAME Darwin)
set (CMAKE_SYSTEM_VERSION 1)
set (UNIX True)
set (APPLE True)
set (IOS True)

# suppress -rdynamic
# set(CMAKE_SYSTEM_NAME Generic)

# Locate gcc
execute_process(COMMAND /usr/bin/xcrun -sdk iphoneos -find clang
                OUTPUT_VARIABLE CMAKE_C_COMPILER
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Locate g++
execute_process(COMMAND /usr/bin/xcrun -sdk iphoneos -find clang++
                OUTPUT_VARIABLE CMAKE_CXX_COMPILER
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Set the CMAKE_OSX_SYSROOT to the latest SDK found
execute_process(COMMAND /usr/bin/xcrun -sdk iphoneos --show-sdk-path
                OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
                OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "gcc found at: ${CMAKE_C_COMPILER}")
message(STATUS "g++ found at: ${CMAKE_CXX_COMPILER}")
message(STATUS "Using iOS SDK: ${CMAKE_OSX_SYSROOT}")

set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})

set(_CMAKE_TOOLCHAIN_PREFIX )

set(CMAKE_IOS_SDK_ROOT ${CMAKE_OSX_SYSROOT})

# Set the sysroot default to the most recent SDK
set(CMAKE_OSX_SYSROOT ${CMAKE_IOS_SDK_ROOT} CACHE PATH "Sysroot used for iOS support")

# set the architecture for iOS
set(IOS_ARCH arm64 arm64e)

set(CMAKE_OSX_ARCHITECTURES ${IOS_ARCH} CACHE STRING "Build architecture for iOS")

if(NOT DEFINED ENABLE_BITCODE)
    # enable bitcode support by default
    set(ENABLE_BITCODE TRUE CACHE BOOL "enable bitcode")
endif()

if(ENABLE_BITCODE)
    # enable bitcode
    set(CMAKE_C_FLAGS "-fembed-bitcode ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fembed-bitcode ${CMAKE_CXX_FLAGS}")
endif()

# Set the find root to the iOS developer roots and to user defined paths
set(CMAKE_FIND_ROOT_PATH ${CMAKE_IOS_DEVELOPER_ROOT} ${CMAKE_IOS_SDK_ROOT} ${CMAKE_PREFIX_PATH} CACHE STRING "iOS find search path root")

# searching for frameworks only
set(CMAKE_FIND_FRAMEWORK FIRST)

# set up the default search directories for frameworks
set(CMAKE_SYSTEM_FRAMEWORK_PATH
    ${CMAKE_IOS_SDK_ROOT}/System/Library/Frameworks
)
