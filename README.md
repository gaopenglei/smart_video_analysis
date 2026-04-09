# 智能视频分析系统 (Smart Video Analysis System)

基于PyTorch、ONNX与ONNX Runtime的智能视频分析系统，支持在ARM Linux环境下进行实时目标检测。

## 项目特性

- **多模型支持**: 支持YOLOv5/v8/v11、ResNet、MobileNet等主流目标检测模型
- **多输入源**: 支持摄像头、视频文件、RTSP网络流等多种输入源
- **高效推理**: 基于ONNX Runtime的高效推理引擎，支持多种执行提供者
- **NPU适配**: 针对RK3588等NPU进行算子适配优化
- **实时可视化**: 实时显示检测结果，支持多种可视化选项

## 目录结构

```
smart_video_analysis/
├── CMakeLists.txt              # CMake构建配置
├── config/                     # 配置文件目录
│   ├── config.yaml            # 主配置文件
│   └── coco_classes.txt       # COCO类别名称
├── include/                    # 头文件目录
│   ├── core/                  # 核心模块头文件
│   │   ├── Logger.hpp         # 日志模块
│   │   ├── Config.hpp         # 配置管理
│   │   └── ErrorHandling.hpp  # 错误处理
│   └── modules/               # 功能模块头文件
│       ├── video_input/       # 视频输入模块
│       ├── preprocessing/     # 预处理模块
│       ├── inference/         # 推理引擎模块
│       ├── postprocessing/    # 后处理模块
│       ├── visualization/     # 可视化模块
│       └── operator_adapter/  # 算子适配模块
├── src/                        # 源文件目录
│   ├── main.cpp               # 主程序入口
│   ├── SmartVideoAnalysisSystem.cpp  # 系统集成
│   ├── core/                  # 核心模块实现
│   └── modules/               # 功能模块实现
├── models/                     # 模型文件目录
├── logs/                       # 日志文件目录
└── output/                     # 输出文件目录
```

## 依赖项

### 必需依赖

- **CMake** >= 3.14
- **C++17** 兼容编译器 (GCC 8+, Clang 7+)
- **OpenCV** >= 4.0
- **ONNX Runtime** >= 1.12.0

### 可选依赖

- **CUDA** >= 11.0 (用于GPU加速)
- **TensorRT** >= 8.0 (用于TensorRT加速)
- **OpenVINO** >= 2022.1 (用于OpenVINO加速)

## 编译说明

### 1. 安装依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake git
sudo apt-get install -y libopencv-dev

# 安装ONNX Runtime
# 从 https://github.com/microsoft/onnxruntime/releases 下载对应版本
# 解压到指定目录，如 /opt/onnxruntime
```

### 2. 编译项目

```bash
cd smart_video_analysis
mkdir build && cd build

# 基本编译
cmake ..
make -j$(nproc)

# 指定ONNX Runtime路径
cmake -DONNXRUNTIME_DIR=/path/to/onnxruntime ..
make -j$(nproc)

# ARM Linux交叉编译
cmake -DARM_BUILD=ON -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux.cmake ..
make -j$(nproc)
```

### 3. 安装

```bash
sudo make install
```

## 使用说明

### 命令行参数

```bash
./smart_video_analysis [options]

选项:
  -h, --help              显示帮助信息
  -c, --config <file>     配置文件路径
  -m, --model <file>      ONNX模型文件路径
  -i, --input <source>    输入源 (摄像头索引、视频文件或RTSP URL)
  -o, --output <file>     输出视频文件路径
  -t, --type <type>       模型类型 (yolov8, yolov5, resnet, mobilenet)
  --confidence <value>    置信度阈值 (0.0-1.0)
  --nms <value>           NMS阈值 (0.0-1.0)
  --no-display            禁用显示窗口
  --verbose               启用详细输出
```

### 使用示例

```bash
# 使用配置文件运行
./smart_video_analysis -c config/config.yaml

# 处理视频文件
./smart_video_analysis -m models/yolov8n.onnx -i video.mp4

# 使用摄像头
./smart_video_analysis -m models/yolov8n.onnx -i 0 -t yolov8

# 处理RTSP流
./smart_video_analysis -m models/yolov8n.onnx -i rtsp://localhost:8554/stream

# 保存输出视频
./smart_video_analysis -m models/yolov8n.onnx -i video.mp4 -o output/result.mp4

# 调整检测阈值
./smart_video_analysis -m models/yolov8n.onnx -i video.mp4 --confidence 0.5 --nms 0.4
```

## 模型准备

### YOLOv8模型导出

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")

# 导出为ONNX格式
model.export(format="onnx", opset=12, simplify=True)
```

### YOLOv5模型导出

```python
import torch
from models.experimental import attempt_load

# 加载模型
model = attempt_load("yolov5s.pt", map_location="cpu")

# 导出为ONNX格式
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "yolov5s.onnx",
    opset_version=12,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}}
)
```

## 配置说明

### 视频输入配置

```yaml
video_input:
  source_type: "file"        # 输入源类型: camera, file, rtsp
  source_path: "./test.mp4"  # 输入源路径
  frame_width: 1920          # 帧宽度
  frame_height: 1080         # 帧高度
  fps: 30                    # 帧率
```

### 推理配置

```yaml
inference:
  num_threads: 4             # 推理线程数
  execution_provider: "CPU"  # 执行提供者: CPU, CUDA, TensorRT, OpenVINO
  enable_fp16: false         # 是否启用FP16推理
```

### 后处理配置

```yaml
postprocess:
  confidence_threshold: 0.25  # 置信度阈值
  nms_threshold: 0.45         # NMS阈值
  max_detections: 100         # 最大检测数量
```

## 性能优化

### 1. 模型优化

- 使用ONNX Simplifier简化模型
- 固定动态维度为静态尺寸
- 针对目标NPU进行算子适配

### 2. 推理优化

- 启用多线程推理
- 使用GPU/NPU加速
- 启用FP16/INT8量化

### 3. 系统优化

- 异步视频帧获取
- 多线程处理流水线
- 内存池管理

## RK3588 NPU部署

### 1. 编译ONNX Runtime for RK3588

```bash
# 克隆ONNX Runtime源码
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# 交叉编译
./build.sh --config Release --build_shared_lib --parallel \
    --cross_compile --arm64 --with_nnapi
```

### 2. 模型转换

```bash
# 使用RKNN-Toolkit2转换模型
python convert_to_rknn.py --model yolov8n.onnx --target rk3588
```

### 3. 部署运行

```bash
# 复制程序和模型到RK3588
scp smart_video_analysis root@rk3588:/opt/sva/
scp models/yolov8n.rknn root@rk3588:/opt/sva/models/

# 运行
ssh root@rk3588
cd /opt/sva
./smart_video_analysis -m models/yolov8n.rknn -i 0
```

## 错误码说明

| 错误码 | 说明 |
|--------|------|
| 0 | 成功 |
| 1-99 | 通用错误 |
| 100-199 | 视频输入模块错误 |
| 200-299 | 预处理模块错误 |
| 300-399 | 模型处理模块错误 |
| 400-499 | 算子适配模块错误 |
| 500-599 | 推理执行模块错误 |
| 600-699 | 后处理模块错误 |

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。
