# YOLOv8 实例分割 · 全链路部署

从数据采集到 Android 端推理的完整 YOLOv8 实例分割管线。

## 项目概述

- **目标**：键盘/鼠标实例分割模型，全链路自主完成
- **数据集**：自行采集并标注
- **模型**：YOLOv8-seg（实例分割）、YOLOv8n（目标检测）

## 技术栈

PyTorch · YOLOv8 · ultralytics · ONNX · NCNN · Android JNI · CUDA

## 核心工作

1. **模型训练**：YOLOv8-seg 实例分割，自定义数据集 YAML 配置
2. **ONNX 导出**：PyTorch → ONNX 转换，处理分割头多输出结构
3. **NCNN 转换**：ONNX → NCNN，定位并修复 mask prototypes 输出丢失问题
4. **param 级修复**：手动修补 NCNN param 文件，恢复双输出结构
5. **Android 部署**：JNI 集成，移动端实时分割推理

## 目录结构

```
yolov8-seg/
├── ncnn/                    # NCNN 工具链（需自行下载）
├── ncnn_final/              # 最终可用的 NCNN 模型
├── ncnn-android-yolov8-seg/ # Android 项目（原始模板）
├── ncnn-android-yolov8-seg-mymodel/  # Android 项目（自定义模型）
├── weights/                 # 模型权重
├── split_merge.py           # 数据集分割/合并工具
└── YOLOv8部署完整指南.md    # 部署流程文档
```

## 运行环境

- CUDA 13.3 + cuDNN 9.23
- PyTorch 2.6.0+cu124
- Python 3.13

## 关键突破

NCNN 转换后 mask prototypes 输出丢失 → ONNX 导出调试 → 手动修补 NCNN param 恢复双输出。
详见 `YOLOv8部署完整指南.md`。


