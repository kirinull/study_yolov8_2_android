# YOLOv8 实例分割 Android 手机部署完整指南

> 基于课程《YOLOv8实例分割实战-Android手机部署》的实战记录  
> 环境：Windows 11, CUDA 13.3, RTX 4080 SUPER, Android 14 (realme)

---

## 一、环境搭建踩坑

### 1.1 CUDA 与 cuDNN

课程要求的 CUDA 11.8 + cuDNN 8.9，但实际使用了 CUDA 13.3 + cuDNN 9.23。

| 组件 | 实际安装 | 备注 |
|---|---|---|
| GPU | RTX 4080 SUPER (16GB) | |
| 驱动 | 610.47 | `nvidia-smi` 可查 |
| CUDA Toolkit | 13.3 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3` |
| cuDNN | 9.23 | 官方安装器安装到 `C:\Program Files\NVIDIA\CUDNN\v9.23` |

**必做**：手动添加环境变量
```
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
CUDA_PATH_V13_3 = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
PATH 追加 CUDA\bin 和 CUDNN\bin
```

CUDA 安装器不会自动设环境变量，必须手动配。

### 1.2 Python 环境

课程使用 Python 3.9 + PyTorch 2.0 + CUDA 11.8。实测 Python 3.13 不兼容课程的 ultralytics 8.0.109（缺 `pkg_resources`）。

**最终方案**：Python 3.9.10 + PyTorch 2.0.1+cu118
```powershell
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy<2  # 课程版本必须 numpy 1.x
```

### 1.3 Git 认证

GitHub 已禁用密码认证，必须用 Personal Access Token。
```
Settings → Developer settings → Personal access tokens → Generate (classic)
勾选 repo 全部权限
```

配置：
```bash
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
git config --global credential.helper manager
git clone https://你的token@github.com/.../repo.git
```

---

## 二、YOLOv8 源码版本

**这是本次最关键的教训。**

课程提供了专门的 `ultralytics.zip`（版本 8.0.109），不能直接用 GitHub 最新版（8.4.x），原因：

1. 8.0.109 的代码结构（`block.py`、`head.py`）和 PDF 完全对应
2. 8.4.x 的 `Detect`、`Segment` 类结构完全不同，PDF 的修改无法套用
3. 8.0.109 需要 Python 3.9 + numpy<2，8.4.x 需要 Python 3.13 + numpy 2.x

```
F:\openhanako\yolov8-seg\
├── ultralytics/          ← 课程 8.0.109 源码（editable install）
│   └── ultralytics/
│       └── nn/modules/
│           ├── block.py   ← C2f 类
│           └── head.py    ← Detect / Segment 类
├── datasets/
│   └── kb_final/          ← 键盘鼠标数据集（399张）
├── weights/
│   ├── yolov8n-seg.pt     ← 预训练权重
│   └── yolov8s-seg.pt
├── ncnn/                  ← ncnn Windows 转换工具
└── ncnn-android-yolov8-seg/ ← Android Studio 项目
```

---

## 三、数据集

### 3.1 COCO category ID 陷阱

COCO 官方 annotation 文件中，category ID 和 ultralytics YAML 里的 class index 不同：

| 类别 | COCO annotation 中的 ID | ultralytics YAML 中的 index |
|---|---|---|
| mouse | **74** | 64 |
| keyboard | **76** | 66 |

**用了错误的 ID（64/66）会导致标注全部漏掉。** 必须用 COCO annotation 原始 ID 去匹配。

### 3.2 标注格式

COCO 的 segmentation 有两种格式：
- **Polygon**（list of points）：可以直接用
- **RLE**（dict with counts + size）：需要 `pycocotools.mask.decode()` 解码为二值 mask，再用 `cv2.findContours` 提取轮廓

课程预训练模型已覆盖 COCO 80 类（含 keyboard/mouse），如果不需要自定义类，直接用预训练权重即可，跳过数据集步骤。

---

## 四、源码修改（ONNX 导出适配）

**注意：训练时不需要改，导出 ONNX 前才改。**

### 4.1 block.py - C2f 类

**位置**：`ultralytics/nn/modules/block.py`

**原代码**：
```python
def forward(self, x):
    y = list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))
```

**改为**：
```python
def forward(self, x):
    #y = list(self.cv1(x).chunk(2, 1))
    #y.extend(m(y[-1]) for m in self.m)
    #return self.cv2(torch.cat(y, 1))
    x = self.cv1(x)
    x = [x, x[:, self.c:, ...]]
    x.extend(m(x[-1]) for m in self.m)
    x.pop(1)
    return self.cv2(torch.cat(x, 1))
```

原因：`chunk(2, 1)` 在 ONNX 导出时会生成不兼容的算子，改用切片。

### 4.2 head.py - Detect 类

**位置**：`ultralytics/nn/modules/head.py`

**原代码**（含 DFL 解码 + dist2bbox 后处理）：
```python
def forward(self, x):
    shape = x[0].shape
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    # ... DFL decode, dist2bbox, sigmoid ...
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y if self.export else (y, x)
```

**改为**：
```python
def forward(self, x):
    shape = x[0].shape
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape
    return torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
```

原因：去掉 DFL（Distribution Focal Loss）解码和 dist2bbox 后处理，让 ONNX 输出原始的 bbox 分布（64 维）+ 类别分数（80 维），由 ncnn 端（yolo.cpp）做后处理。

**如果不改 head.py，ONNX 输出维度是 116（= 4 bbox + 80 cls + 32 mask），yolo.cpp 期望 176（= 64 raw bbox + 80 cls + 32 mask），维度不匹配就会 SIGSEGV 闪退。**

---

## 五、导出 ONNX

```python
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')  # 或自己的 best.pt
model.export(format='onnx', simplify=True, opset=12, imgsz=640)
```

验证输出维度：
```python
import onnx
m = onnx.load('yolov8n-seg.onnx')
print(m.graph.output[0].type.tensor_type.shape)  # 应为 [1, 176, 8400]
```

---

## 六、ONNX → NCNN

### 6.1 版本匹配

**NCNN 转换工具版本必须和 Android 项目中 ncnn 库版本一致。**

课程 Android 项目使用 `ncnn-20230223-android-vulkan`，所以 onnx2ncnn 也要用 20230223 版本。

下载地址：https://github.com/Tencent/ncnn/releases/tag/20230223

```
ncnn-20230223-windows-vs2022.zip → 解压 → x64\bin\onnx2ncnn.exe
```

**用错版本（如 20240410 转出来的模型在 20230223 的 ncnn 库上运行）会导致运行时崩溃。**

### 6.2 转换命令

```powershell
# 转换
onnx2ncnn.exe yolov8n-seg.onnx yolov8n-seg.param yolov8n-seg.bin

# 优化（可选，但建议做）
ncnnoptimize.exe yolov8n-seg.param yolov8n-seg.bin yolov8n-seg-opt.param yolov8n-seg-opt.bin 0
```

完成后将 param 和 bin 文件放入 `app/src/main/assets/`。

---

## 七、Android 项目配置

### 7.1 必要的 Gradle 升级

原项目 Gradle 5.4.1 不兼容 Java 17。修改：

**`gradle-wrapper.properties`**：
```properties
distributionUrl=https\://services.gradle.org/distributions/gradle-7.3.3-all.zip
```

**`build.gradle`**（项目根）：
```groovy
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:7.2.0'
    }
}
```

**`app/build.gradle`**：
```groovy
compileSdkVersion 32
targetSdkVersion 33
namespace 'com.tencent.yolov8ncnn'
ndkVersion '24.0.8215888'
```

### 7.2 AndroidX 迁移

原项目用 `android.support.v4`（已废弃）。改为 AndroidX：

**`gradle.properties`**（新建）：
```properties
android.useAndroidX=true
android.enableJetifier=true
```

**`app/build.gradle`** 依赖：
```groovy
implementation 'androidx.core:core:1.6.0'
```

**MainActivity.java** 导入：
```java
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
```

### 7.3 Android 14 适配

**`AndroidManifest.xml`**：
```xml
<!-- camera2.full 改为 camera（项目用 Camera1 NDK API） -->
<uses-feature android:name="android.hardware.camera" android:required="true" />
<!-- targetSdk 33+ 必须加 exported -->
<activity ... android:exported="true">
```

**`ndkcamera.cpp`** 修复：主线程 `ALooper_prepare` 在 Android 14 会崩溃：
```cpp
// 原代码：
sensor_event_queue = ASensorManager_createEventQueue(..., ALooper_prepare(...), ...);

// 改为：
ALooper* looper = ALooper_forThread();
if (looper == NULL) {
    looper = ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
}
sensor_event_queue = ASensorManager_createEventQueue(..., looper, ...);
```

### 7.4 NDK / CMake

- NDK：`24.0.8215888`（SDK Manager → SDK Tools → Show Package Details → 勾选安装）
- CMake：`3.10.2.4988404`
- `local.properties`：
```properties
sdk.dir=C\:\\Users\\29267\\AppData\\Local\\Android\\Sdk
ndk.dir=C\:\\Users\\29267\\AppData\\Local\\Android\\Sdk\\ndk\\24.0.8215888
cmake.dir=C\:\\Users\\29267\\AppData\\Local\\Android\\Sdk\\cmake\\3.10.2.4988404
```

---

## 八、闪退排查流程

### 8.1 常见闪退原因（按概率排序）

| 原因 | 现象 | 解决 |
|---|---|---|
| **head.py 没改** | ONNX 输出 116 维，yolo.cpp 读 176 维，SIGSEGV | 改 head.py Detect forward |
| **ncnn 版本不匹配** | onnx2ncnn 版本 ≠ Android ncnn 库版本 | 用 20230223 版本工具 |
| **num_class 不匹配** | yolo.cpp num_class ≠ 模型类别数 | 用 80 类模型 + 80 类 yolo.cpp |
| **ALooper_prepare 冲突** | Android 14 主线程 ALooper 崩溃 | 先 ALooper_forThread 检查 |
| **权限未授权** | 摄像头打开失败 | 添加 onRequestPermissionsResult |
| **模型加载失败** | 屏幕显示 "unsupported" | 检查 assets 中 param/bin 文件名 |
| **android.support.v4** | Java 编译失败 | 改为 AndroidX |
| **Gradle/AGP 版本** | 项目无法 sync | Gradle 7.3.3 + AGP 7.2.0 |

### 8.2 抓崩溃日志

```powershell
adb logcat -b crash -d -t 100
```

关键信息：
- `Fatal signal 11 (SIGSEGV)` → 内存越界，大概率是模型输出维度不匹配
- `backtrace` → 定位到具体函数（如 `Yolo::detect`、`NdkCameraWindow::on_image`）
- `BuildId` → 确认是否是刚编译的版本（如果 BuildId 没变说明 APK 没更新）

### 8.3 终极回退方案

如果自己导出的模型一直闪退，**先用课程自带模型验证基础功能**：

1. 从 `ncnn-android-yolov8-seg.zip` 恢复原始 `assets/yolov8n-seg.*`
2. 从 zip 恢复原始 `yolo.cpp`
3. 编译跑通 → 确认摄像头、推理都正常
4. 再逐步替换为自己的模型

---

## 九、yolo.cpp 关键参数

```cpp
const int num_class = 80;           // 必须和模型 nc 一致
const int reg_max_1 = 16;           // DFL 参数，固定
```

**输出张量布局**（每 anchor）：
```
[ 64 floats: bbox DFL分布 | num_class floats: 分数 | 32 floats: mask系数 ]
  总长度 = 64 + num_class + 32 = 64 + 80 + 32 = 176
```

**如果是自定义 2 类模型**，需要确保：
1. ONNX 输出维度 = 64 + 2 + 32 = 98
2. yolo.cpp `num_class = 2`
3. ncnn param 文件内容能正常被 ncnn 库解析

但从实际测试来看，2 类模型的 ONNX 导出在 ncnn 上不稳定（经过 DFL 后的输出维度变了），**建议直接用 80 类预训练模型 + 在 yolo.cpp 里只显示 keyboard/mouse**。

---

## 十、完整操作流程（速查）

```
1. 环境
   安装 CUDA + cuDNN → 配环境变量 → Python 3.9 + PyTorch 2.0

2. 源码
   解压课程 ultralytics.zip → pip install -e .

3. 训练（可选，预训练模型已有 COCO 80 类含 keyboard/mouse）
   构建数据集 → python train.py

4. 源码修改（导出前）
   改 block.py C2f forward → 改 head.py Detect forward

5. 导出
   yolo export format=onnx simplify=True opset=12
   验证输出维度：onnx.checker 或 python onnx 查看

6. 转 NCNN
   下载 ncnn-20230223-windows-vs2022.zip
   onnx2ncnn.exe model.onnx model.param model.bin
   ncnnoptimize.exe model.param model.bin out.param out.bin 0

7. Android
   复制 param + bin 到 app/src/main/assets/
   恢复原始 yolo.cpp（80 类）
   Sync → Build APK → 安装到手机
```

---

## 十一、训练结果（参考）

用 COCO train2017 筛选 399 张含 keyboard+mouse 的图片训练 YOLOv8n-seg：

| 类别 | Box mAP50 | Mask mAP50 |
|---|---|---|
| mouse | 0.679 | 0.674 |
| keyboard | 0.564 | 0.544 |

键盘效果偏低（COCO 中键盘常被手遮挡、角度偏斜、样本偏少），但预训练模型本身足以在手机上跑通。

---

## 十二、关键教训

1. **COCO category ID 混淆**：annotation 用 74/76，ultralytics YAML 用 64/66，不要搞混
2. **ncnn 版本匹配**：转换工具版本 = Android 库版本 = 20230223
3. **head.py 修改不可省略**：不改则输出维度 116 ≠ yolo.cpp 期望的 176，必崩
4. **课程指定版本**：ultralytics 8.0.109，不能用 GitHub 最新版
5. **Android 14 特殊处理**：ALooper 检查、exported 属性、AndroidX 迁移
6. **先跑通基础再改**：用原始模型验证全链路，再替换自己的模型
