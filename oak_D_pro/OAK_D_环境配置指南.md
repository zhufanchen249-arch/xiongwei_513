# OAK-D 相机标定环境配置指南

## 依赖库清单

### 核心依赖（必须安装）

| 包名 | 版本 | 说明 | 安装方式 |
|------|------|------|----------|
| python | 3.10.x | 推荐版本 | conda |
| opencv-python | 4.6.0.66 | OpenCV 主包 | pip |
| opencv-contrib-python | 4.6.0.66 | **必须安装**，包含 aruco 模块 | pip |
| depthai | 2.28.0.0 | OAK 相机官方 Python SDK | pip |
| numpy | - | 数组操作 | pip |

### 可选依赖

| 包名 | 用途 | 安装方式 |
|------|------|----------|
| pyyaml | 读写 YAML 标定文件 | pip |

### 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 22.04 |
| USB 版本 | USB 3.0 |
| Conda | miniconda 或 anaconda |

---

## 详细操作步骤

### 步骤 1：创建 conda 环境

```bash
conda create -n oak_test python=3.10 -y
```

### 步骤 2：激活环境

```bash
conda activate oak_test
```

### 步骤 3：安装核心依赖

```bash
# 必须同时安装两个 OpenCV 包
pip install opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66

# 安装 DepthAI SDK
pip install depthai==2.28.0.0

# 安装 YAML 处理库（可选）
pip install pyyaml
```

### 步骤 4：验证安装

```bash
python -c "
import cv2
from cv2 import aruco
import depthai as dai
import numpy as np
print('OpenCV:', cv2.__version__)
print('DepthAI:', dai.__version__)
print('NumPy:', np.__version__)
print('aruco 模块可用')
"
```

### 步骤 5：运行标定程序

```bash
# 检查相机连接
python check_oak.py

# 运行标定
python charuco_calib.py
```

---

## 环境导出与复现

### 导出配置

```bash
conda activate oak_test
conda env export > oak_test_environment.yml
```

### 他人复现

```bash
conda env create -f oak_test_environment.yml
conda activate oak_test
```

---

## 常见问题

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `No module named 'aruco'` | 缺少 opencv-contrib-python | 同时安装两个 OpenCV 包 |
| `Segmentation fault` | OpenCV 4.6.0 新 API 不兼容 | 使用旧 API（带 `_create` 后缀） |
| 相机无法连接 | USB 2.0 或供电不足 | 使用 USB 3.0 或 Y 型线供电 |
