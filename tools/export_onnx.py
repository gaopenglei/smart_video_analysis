#!/usr/bin/env python3
"""
模型转换工具脚本
用于将PyTorch模型导出为ONNX格式，并进行优化处理
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def export_yolov8_to_onnx(model_path: str, output_path: str, imgsz: int = 640, 
                          simplify: bool = True, opset: int = 12):
    """
    将YOLOv8模型导出为ONNX格式
    
    Args:
        model_path: PyTorch模型路径
        output_path: ONNX模型输出路径
        imgsz: 输入图像尺寸
        simplify: 是否使用onnx-simplifier简化模型
        opset: ONNX opset版本
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found. Install with: pip install ultralytics")
        sys.exit(1)
    
    print(f"Loading YOLOv8 model from: {model_path}")
    model = YOLO(model_path)
    
    # 导出为ONNX
    print(f"Exporting to ONNX with imgsz={imgsz}, opset={opset}")
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,  # 固定batch size为1
    )
    
    # 重命名输出文件
    default_output = model_path.replace(".pt", ".onnx")
    if default_output != output_path and os.path.exists(default_output):
        os.rename(default_output, output_path)
    
    print(f"ONNX model saved to: {output_path}")
    return output_path


def export_yolov5_to_onnx(model_path: str, output_path: str, imgsz: int = 640,
                          simplify: bool = True, opset: int = 12):
    """
    将YOLOv5模型导出为ONNX格式
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not found. Install with: pip install torch")
        sys.exit(1)
    
    print(f"Loading YOLOv5 model from: {model_path}")
    
    # 尝试加载模型
    try:
        # 使用torch.hub加载
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    
    # 导出ONNX
    print(f"Exporting to ONNX with imgsz={imgsz}, opset={opset}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=None,  # 静态形状
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    # 简化模型
    if simplify:
        simplify_onnx(output_path)
    
    return output_path


def export_resnet_to_onnx(model_name: str, output_path: str, imgsz: int = 224,
                          simplify: bool = True, opset: int = 12):
    """
    将ResNet模型导出为ONNX格式
    """
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("Error: PyTorch/torchvision not found. Install with: pip install torch torchvision")
        sys.exit(1)
    
    print(f"Loading ResNet model: {model_name}")
    
    # 获取模型
    model_fn = getattr(models, model_name)
    model = model_fn(pretrained=True)
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    
    # 导出ONNX
    print(f"Exporting to ONNX with imgsz={imgsz}, opset={opset}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    if simplify:
        simplify_onnx(output_path)
    
    return output_path


def export_mobilenet_to_onnx(model_name: str, output_path: str, imgsz: int = 224,
                             simplify: bool = True, opset: int = 12):
    """
    将MobileNet模型导出为ONNX格式
    """
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("Error: PyTorch/torchvision not found. Install with: pip install torch torchvision")
        sys.exit(1)
    
    print(f"Loading MobileNet model: {model_name}")
    
    # 获取模型
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
    else:
        print(f"Unknown MobileNet variant: {model_name}")
        sys.exit(1)
    
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    
    # 导出ONNX
    print(f"Exporting to ONNX with imgsz={imgsz}, opset={opset}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    if simplify:
        simplify_onnx(output_path)
    
    return output_path


def simplify_onnx(model_path: str):
    """
    使用onnx-simplifier简化ONNX模型
    """
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("Warning: onnx-simplifier not found. Install with: pip install onnx-simplifier")
        return model_path
    
    print(f"Simplifying ONNX model: {model_path}")
    
    # 加载模型
    model = onnx.load(model_path)
    
    # 简化模型
    model_simplified, check = simplify(model)
    
    # 保存简化后的模型
    onnx.save(model_simplified, model_path)
    
    print(f"Model simplified and saved to: {model_path}")
    return model_path


def check_onnx_model(model_path: str):
    """
    检查ONNX模型有效性
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print("Warning: onnx not found. Install with: pip install onnx")
        return
    
    print(f"\nChecking ONNX model: {model_path}")
    
    # 加载模型
    model = onnx.load(model_path)
    
    # 检查模型有效性
    onnx.checker.check_model(model)
    print("✓ Model is valid")
    
    # 打印模型信息
    print("\nModel Info:")
    print(f"  IR Version: {model.ir_version}")
    print(f"  Opset Version: {model.opset_import[0].version}")
    print(f"  Producer: {model.producer_name} {model.producer_version}")
    
    # 打印输入信息
    print("\nInputs:")
    for input in model.graph.input:
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f"  {input.name}: {shape}")
    
    # 打印输出信息
    print("\nOutputs:")
    for output in model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"  {output.name}: {shape}")
    
    # 统计算子
    op_types = {}
    for node in model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    
    print("\nOperators:")
    for op_type, count in sorted(op_types.items()):
        print(f"  {op_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Model conversion tool for Smart Video Analysis System")
    parser.add_argument("--model", "-m", required=True, help="Model path or name")
    parser.add_argument("--output", "-o", required=True, help="Output ONNX file path")
    parser.add_argument("--type", "-t", choices=["yolov8", "yolov5", "resnet", "mobilenet"],
                        required=True, help="Model type")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Skip ONNX simplification")
    parser.add_argument("--check", action="store_true", help="Check ONNX model after export")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 根据模型类型导出
    simplify = not args.no_simplify
    
    if args.type == "yolov8":
        export_yolov8_to_onnx(args.model, args.output, args.imgsz, simplify, args.opset)
    elif args.type == "yolov5":
        export_yolov5_to_onnx(args.model, args.output, args.imgsz, simplify, args.opset)
    elif args.type == "resnet":
        export_resnet_to_onnx(args.model, args.output, args.imgsz, simplify, args.opset)
    elif args.type == "mobilenet":
        export_mobilenet_to_onnx(args.model, args.output, args.imgsz, simplify, args.opset)
    
    # 检查模型
    if args.check:
        check_onnx_model(args.output)
    
    print("\n✓ Model conversion completed!")


if __name__ == "__main__":
    main()
