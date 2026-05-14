cmake_minimum_required(VERSION 3.14)
project(SmartVideoAnalysis VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# 编译选项设置
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ARM Linux特定优化选项
if(ARM_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+crypto+fp+simd -mtune=cortex-a76.cortex-a55")
    add_definitions(-DARM_BUILD)
endif()

# ============================================================================
# 依赖库查找
# ============================================================================

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# ONNX Runtime
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation directory")
if(ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")
    
    if(WIN32)
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
    else()
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")
    endif()
    
    message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")
else()
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h 
             PATHS /usr/local/include/onnxruntime /usr/include/onnxruntime)
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime 
                PATHS /usr/local/lib /usr/lib)
    message(STATUS "ONNX Runtime found in system paths")
endif()

# Protobuf (用于ONNX模型解析)
find_package(Protobuf QUIET)
if(Protobuf_FOUND)
    message(STATUS "Protobuf version: ${Protobuf_VERSION}")
else()
    message(STATUS "Protobuf not found, will try to find manually")
    find_library(PROTOBUF_LIB NAMES protobuf)
endif()

# ONNX (用于模型解析和优化)
set(ONNX_INCLUDE_DIR "" CACHE PATH "Path to ONNX include directory")
set(ONNX_LIB_DIR "" CACHE PATH "Path to ONNX library directory")

# ============================================================================
# RKNN Runtime 支持（可选）
# 在 RK3588 等瑞芯微设备上启用：
#   cmake -DUSE_RKNN=ON -DRKNN_DIR=/path/to/rknn-toolkit2/runtime/Linux/librknn_api ..
# ============================================================================
option(USE_RKNN "Enable RKNN Runtime inference backend (for RK3588)" OFF)
if(USE_RKNN)
    set(RKNN_DIR "" CACHE PATH "Path to RKNN Runtime directory (contains include/ and lib/)")
    if(RKNN_DIR)
        set(RKNN_INCLUDE_DIR "${RKNN_DIR}/include")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(RKNN_LIB "${RKNN_DIR}/lib/librknnrt.so")
        else()
            set(RKNN_LIB "${RKNN_DIR}/lib/librknnrt.so")
            message(WARNING "USE_RKNN is ON but target is not aarch64. "
                            "RKNN Runtime only runs on ARM Linux / RK3588.")
        endif()
    else()
        find_path(RKNN_INCLUDE_DIR  NAMES rknn_api.h
                  PATHS /usr/include /usr/local/include)
        find_library(RKNN_LIB NAMES rknnrt
                     PATHS /usr/lib /usr/local/lib)
    endif()

    if(RKNN_INCLUDE_DIR AND RKNN_LIB)
        message(STATUS "RKNN Runtime include: ${RKNN_INCLUDE_DIR}")
        message(STATUS "RKNN Runtime library: ${RKNN_LIB}")
        add_definitions(-DUSE_RKNN)
    else()
        message(FATAL_ERROR
            "USE_RKNN=ON but rknn_api.h / librknnrt.so not found. "
            "Set -DRKNN_DIR=<path> to the RKNN Runtime SDK directory.")
    endif()
endif()

# ============================================================================
# 源文件定义
# ============================================================================

# 核心模块源文件
set(CORE_SOURCES
    src/core/Logger.cpp
    src/core/Config.cpp
    src/core/ErrorHandling.cpp
)

# 视频输入模块
set(VIDEO_INPUT_SOURCES
    src/modules/video_input/VideoInput.cpp
)

# 预处理模块
set(PREPROCESSING_SOURCES
    src/modules/preprocessing/ImagePreprocessor.cpp
)

# 模型处理模块
set(MODEL_PROCESSOR_SOURCES
    src/modules/model_processor/ModelProcessor.cpp
)

# 推理模块（ORT 始终编译；RKNN 始终编译，但 USE_RKNN 未定义时为空实现）
set(INFERENCE_SOURCES
    src/modules/inference/InferenceEngine.cpp
    src/modules/inference/RKNNInferenceEngine.cpp
)

# 后处理模块
set(POSTPROCESSING_SOURCES
    src/modules/postprocessing/DetectionPostprocessor.cpp
)

# 可视化模块
set(VISUALIZATION_SOURCES
    src/modules/visualization/Visualizer.cpp
)

# 算子适配模块
set(OPERATOR_ADAPTER_SOURCES
    src/modules/operator_adapter/OperatorAdapter.cpp
)

# 系统集成
set(SYSTEM_SOURCES
    src/SmartVideoAnalysisSystem.cpp
)

# 主程序
set(MAIN_SOURCES
    src/main.cpp
)

# ============================================================================
# 可执行文件构建
# ============================================================================

add_executable(smart_video_analysis
    ${CORE_SOURCES}
    ${VIDEO_INPUT_SOURCES}
    ${PREPROCESSING_SOURCES}
    ${MODEL_PROCESSOR_SOURCES}
    ${INFERENCE_SOURCES}
    ${POSTPROCESSING_SOURCES}
    ${VISUALIZATION_SOURCES}
    ${OPERATOR_ADAPTER_SOURCES}
    ${SYSTEM_SOURCES}
    ${MAIN_SOURCES}
)

# ============================================================================
# 包含目录
# ============================================================================

target_include_directories(smart_video_analysis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# 添加ONNX和Protobuf包含目录（如果指定）
if(ONNX_INCLUDE_DIR)
    target_include_directories(smart_video_analysis PRIVATE ${ONNX_INCLUDE_DIR})
endif()
if(Protobuf_INCLUDE_DIRS)
    target_include_directories(smart_video_analysis PRIVATE ${Protobuf_INCLUDE_DIRS})
endif()

# 添加 RKNN 包含目录（如果启用）
if(USE_RKNN AND RKNN_INCLUDE_DIR)
    target_include_directories(smart_video_analysis PRIVATE ${RKNN_INCLUDE_DIR})
endif()

# ============================================================================
# 链接库
# ============================================================================

# 使用 CMake 标准线程库，跨平台兼容（Windows 不需要 -lpthread）
find_package(Threads REQUIRED)

target_link_libraries(smart_video_analysis PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    Threads::Threads
)

# dl 仅在 Unix-like 系统上需要
if(UNIX)
    target_link_libraries(smart_video_analysis PRIVATE dl)
endif()

# Windows：将 onnxruntime.dll 拷贝到可执行文件目录，确保运行时可以找到
if(WIN32 AND ONNXRUNTIME_DIR)
    add_custom_command(TARGET smart_video_analysis POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:smart_video_analysis>
        COMMENT "Copying onnxruntime.dll to output directory"
    )
endif()

# 添加Protobuf库（如果找到）
if(Protobuf_FOUND)
    target_link_libraries(smart_video_analysis PRIVATE ${Protobuf_LIBRARIES})
elseif(PROTOBUF_LIB)
    target_link_libraries(smart_video_analysis PRIVATE ${PROTOBUF_LIB})
endif()

# 链接 RKNN Runtime（如果启用）
if(USE_RKNN AND RKNN_LIB)
    target_link_libraries(smart_video_analysis PRIVATE ${RKNN_LIB})
endif()

# ============================================================================
# 编译定义
# ============================================================================

target_compile_definitions(smart_video_analysis PRIVATE
    PROJECT_VERSION="${PROJECT_VERSION}"
    PROJECT_NAME="${PROJECT_NAME}"
)

# ============================================================================
# 安装规则
# ============================================================================

install(TARGETS smart_video_analysis RUNTIME DESTINATION bin)
install(DIRECTORY config/ DESTINATION etc/smart_video_analysis)
# models/ 目录为运行时产物（不纳入源码仓库），仅在目录存在时安装
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
    install(DIRECTORY models/ DESTINATION share/smart_video_analysis/models)
endif()

# ============================================================================
# 测试 (可选)
# ============================================================================

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# 打印配置摘要
# ============================================================================

message(STATUS "")
message(STATUS "========== Smart Video Analysis Build Configuration ==========")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")
if(Protobuf_FOUND)
    message(STATUS "Protobuf: ${Protobuf_VERSION}")
endif()
message(STATUS "RKNN backend: ${USE_RKNN} (lib: ${RKNN_LIB})")
message(STATUS "ARM build: ${ARM_BUILD}")
message(STATUS "==============================================================")
message(STATUS "")
