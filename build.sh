#!/bin/bash
#
# 智能视频分析系统构建脚本
# Build script for Smart Video Analysis System
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认参数
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL_PREFIX=""
ONNXRUNTIME_DIR=""
ARM_BUILD=false
VERBOSE=false
CLEAN=false

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -b|--build-dir)
                BUILD_DIR="$2"
                shift 2
                ;;
            -i|--install-prefix)
                INSTALL_PREFIX="$2"
                shift 2
                ;;
            -o|--onnxruntime)
                ONNXRUNTIME_DIR="$2"
                shift 2
                ;;
            --arm)
                ARM_BUILD=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -c|--clean)
                CLEAN=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 显示帮助信息
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -t, --type <type>       Build type (Debug/Release, default: Release)"
    echo "  -b, --build-dir <dir>   Build directory (default: build)"
    echo "  -i, --install-prefix    Installation prefix"
    echo "  -o, --onnxruntime <dir> ONNX Runtime installation directory"
    echo "  --arm                   Build for ARM platform"
    echo "  -v, --verbose           Verbose output"
    echo "  -c, --clean             Clean build directory before building"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default build"
    echo "  $0 -t Debug -v                        # Debug build with verbose output"
    echo "  $0 -o /path/to/onnxruntime --arm      # ARM build with custom ONNX Runtime"
}

# 检查依赖
check_dependencies() {
    print_info "Checking dependencies..."
    
    # 检查CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake not found. Please install CMake 3.14 or later."
        exit 1
    fi
    print_success "CMake found: $(cmake --version | head -n1)"
    
    # 检查编译器
    if command -v g++ &> /dev/null; then
        print_success "g++ found: $(g++ --version | head -n1)"
    elif command -v clang++ &> /dev/null; then
        print_success "clang++ found: $(clang++ --version | head -n1)"
    else
        print_error "No C++ compiler found. Please install g++ or clang++."
        exit 1
    fi
    
    # 检查OpenCV
    if pkg-config --exists opencv4 2>/dev/null; then
        print_success "OpenCV found: $(pkg-config --modversion opencv4)"
    elif pkg-config --exists opencv 2>/dev/null; then
        print_success "OpenCV found: $(pkg-config --modversion opencv)"
    else
        print_warning "OpenCV not found via pkg-config. CMake will try to find it."
    fi
    
    # 检查ONNX Runtime
    if [ -n "$ONNXRUNTIME_DIR" ]; then
        if [ -d "$ONNXRUNTIME_DIR" ]; then
            print_success "ONNX Runtime directory: $ONNXRUNTIME_DIR"
        else
            print_error "ONNX Runtime directory not found: $ONNXRUNTIME_DIR"
            exit 1
        fi
    else
        print_warning "ONNX Runtime directory not specified. Using system default."
    fi
}

# 清理构建目录
clean_build() {
    if [ "$CLEAN" = true ] && [ -d "$BUILD_DIR" ]; then
        print_info "Cleaning build directory: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
}

# 配置CMake
configure_cmake() {
    print_info "Configuring CMake..."
    
    mkdir -p "$BUILD_DIR"
    
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    )
    
    if [ -n "$INSTALL_PREFIX" ]; then
        CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX")
    fi
    
    if [ -n "$ONNXRUNTIME_DIR" ]; then
        CMAKE_ARGS+=(-DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR")
    fi
    
    if [ "$ARM_BUILD" = true ]; then
        CMAKE_ARGS+=(-DARM_BUILD=ON)
    fi
    
    if [ "$VERBOSE" = true ]; then
        CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
    fi
    
    cd "$BUILD_DIR"
    cmake "${CMAKE_ARGS[@]}" ..
    
    if [ $? -eq 0 ]; then
        print_success "CMake configuration completed"
    else
        print_error "CMake configuration failed"
        exit 1
    fi
}

# 构建项目
build_project() {
    print_info "Building project..."
    
    local PARALLEL_JOBS=$(nproc 2>/dev/null || echo 4)
    
    if [ "$VERBOSE" = true ]; then
        cmake --build . --config "$BUILD_TYPE" -j "$PARALLEL_JOBS" -- VERBOSE=1
    else
        cmake --build . --config "$BUILD_TYPE" -j "$PARALLEL_JOBS"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# 安装项目
install_project() {
    if [ -n "$INSTALL_PREFIX" ]; then
        print_info "Installing to: $INSTALL_PREFIX"
        cmake --install . --config "$BUILD_TYPE"
        
        if [ $? -eq 0 ]; then
            print_success "Installation completed"
        else
            print_error "Installation failed"
            exit 1
        fi
    fi
}

# 打印构建摘要
print_summary() {
    echo ""
    echo "=========================================="
    echo "         Build Summary"
    echo "=========================================="
    echo "Build Type:      $BUILD_TYPE"
    echo "Build Directory: $BUILD_DIR"
    echo "ARM Build:       $ARM_BUILD"
    if [ -n "$INSTALL_PREFIX" ]; then
        echo "Install Prefix:  $INSTALL_PREFIX"
    fi
    if [ -n "$ONNXRUNTIME_DIR" ]; then
        echo "ONNX Runtime:    $ONNXRUNTIME_DIR"
    fi
    echo "=========================================="
    echo ""
}

# 主函数
main() {
    parse_args "$@"
    
    print_info "Starting build process..."
    
    check_dependencies
    clean_build
    configure_cmake
    build_project
    install_project
    print_summary
    
    print_success "All done! The executable is in: $BUILD_DIR/smart_video_analysis"
}

# 运行主函数
main "$@"
