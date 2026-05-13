# Hand-Eye Calibration Module for LeRobot

手眼标定模块，用于计算相机与机械臂末端之间的变换关系（Eye-in-Hand配置）。

## 功能特性

### 核心功能
- **相机内参标定**：使用ChArUco标定板计算相机内参（焦距、畸变系数）
- **手眼标定**：求解相机到机械臂末端的变换矩阵（AX=XB方程）
- **标定精度验证**：实时验证标定结果精度

### 技术特点
- 支持多种ChArUco标定板配置（4x4, 5x5, 6x6, 7x7字典）
- 支持5种手眼标定算法：Tsai、Park、Horaud、Andreff、Daniilidis
- 亚像素级角点检测（CORNER_REFINE_CONTOUR）
- 基于URDF的正运动学计算（支持placo或内置FK）
- 机械臂位姿变化检测（防止采集冗余数据）
- 自动评估标定误差（平移误差、旋转误差）
- 支持MJPG格式和V4L2后端（优化Linux/Jetson性能）

## 文件结构

```
src/lerobot/hand_eye_calibration/
├── __init__.py                  # 模块导出接口
├── configs.py                   # ChArUco配置、内参/外参读写
├── charuco_detector.py          # ChArUco检测 + 6D位姿估计
├── camera_calibrator.py         # 相机内参标定
├── data_collector.py            # 手眼标定数据采集（含防呆机制）
├── hand_eye_calibrator.py       # 手眼标定计算（AX=XB求解）
├── robot_interface.py           # 机械臂连接 + 正运动学
├── verify_hand_eye.py          # 标定精度验证工具
├── scripts/
│   └── lerobot_hand_eye_calibrate.py  # CLI入口
└── outputs/                     # 标定输出目录
    ├── camera_intrinsics.json   # 相机内参
    ├── hand_eye_extrinsics.json # 手眼外参
    └── hand_eye_data.json       # 完整标定数据
```

## 依赖安装

```bash
# 基础依赖（LeRobot环境）
uv sync --locked

# 额外依赖（如使用placo进行正运动学）
pip install placo
```

## 使用流程

### 准备工作

1. **打印ChArUco标定板**
   - 推荐使用5×7的ChArUco板（DICT_6X6_250）
   - 方格边长：25mm，标记边长：18mm
   - 将标定板固定在桌面上

2. **硬件连接**
   - 相机安装在机械臂末端
   - 连接机械臂（默认端口：/dev/ttyACM0）
   - 连接相机（默认设备：/dev/video0）

### 第1步：相机内参标定

```bash
# 启动内参标定
python -m lerobot.hand_eye_calibration.scripts.lerobot_hand_eye_calibrate \
    --mode intrinsic \
    --camera /dev/video0 \
    --camera_width 1280 \
    --camera_height 720
```

**交互按键**：
- `c` - 采集标定帧（画面出现绿色角点表示检测成功）
- `s` - 计算并保存内参到 `outputs/camera_intrinsics.json`
- `q` - 退出

**要求**：至少采集5帧有效数据（不同角度）

### 第2步：手眼标定

```bash
# 启动手眼标定
python -m lerobot.scripts.lerobot_hand_eye_calibrate \
    --mode hand_eye \
    --camera /dev/video0 \
    --camera_width 1280 \
    --camera_height 720 \
    --intrinsics ./outputs/camera_intrinsics.json \
    --robot_type so100_follower \
    --robot_port /dev/ttyACM0 \
    --urdf_path ./SO-ARM100-main/so100_new_calib.urdf
```

**交互按键**：
- `c` - 采集数据对（读取关节角度 + 检测标定板位姿）
- `u` - 撤销上一对数据
- `s` - 计算外参（需≥15对）
- `q` - 退出（≥15对时自动计算）

**要求**：
- 最少15对数据，推荐25-30对
- 机械臂需变换不同位姿（平移>1mm，旋转>1°）
- 达到30对自动停止计算

### 第3步：验证标定精度

```bash
# 运行验证工具
python -m lerobot.hand_eye_calibration.verify_hand_eye
```

**交互按键**：
- `c` - 采集当前标定板在基座坐标系下的坐标
- `s` - 执行精度分析（计算极差和标准差）
- `r` - 重置数据
- `q` - 退出

**精度评级**：
- **S级**：最大误差 < 3mm（完美）
- **A级**：最大误差 < 8mm（优秀，胜任常规抓取）
- **B级**：最大误差 < 15mm（一般，建议检查机械晃动）
- **F级**：最大误差 ≥ 15mm（标定失败，需重新标定）

## 输出文件说明

### camera_intrinsics.json
```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [k1, k2, p1, p2, k3],
  "image_width": 1280,
  "image_height": 720,
  "_checksum": "abc12345"
}
```

### hand_eye_extrinsics.json
```json
{
  "R_cam_to_ee": [[...], [...], [...]],  // 3x3旋转矩阵
  "t_cam_to_ee": [[...]],                // 3x1平移向量（米）
  "method": "tsai",
  "num_pairs": 25,
  "_checksum": "def67890"
}
```

### hand_eye_data.json
完整的标定数据，包含所有采集对，可用于重新计算或分析。

## 标定原理

手眼标定求解经典方程：**AX = XB**

- **A**：机械臂末端位姿变化（从关节角度计算）
- **B**：标定板相对于相机的位姿变化（从图像检测）
- **X**：待求解的相机到末端变换矩阵（T_cam_to_ee）

变换链：Base → EE → Cam → Marker

## 命令行参数

### 通用参数
- `--camera`：相机设备路径（默认：/dev/video0）
- `--camera_width`：相机宽度（默认：1280）
- `--camera_height`：相机高度（默认：720）
- `--output_dir`：输出目录（默认：./outputs）

### 标定板参数
- `--squares_x`：标定板列数（默认：5）
- `--squares_y`：标定板行数（默认：7）
- `--square_length`：方格边长（米，默认：0.030）
- `--marker_length`：标记边长（米，默认：0.024）
- `--dictionary`：字典类型（默认：DICT_6X6_250）

### 手眼标定参数
- `--intrinsics`：内参文件路径
- `--robot_type`：机械臂类型（默认：so100_follower）
- `--robot_port`：机械臂端口（默认：/dev/ttyACM0）
- `--urdf_path`：URDF文件路径
- `--method`：标定算法（tsai/park/horaud/andreff/daniilidis）

## 注意事项

1. **标定板固定**：手眼标定过程中，标定板必须固定在桌面上不动
2. **充分运动**：机械臂需在不同位置和姿态采集数据（避免共面）
3. **扭矩禁用**：标定前禁用机械臂扭矩，便于手动移动
4. **光照充足**：确保标定板清晰可见，角点检测稳定
5. **URDF匹配**：使用的URDF文件需与实际机械臂型号一致

## 故障排查

### 检测不到标定板
- 检查光照条件
- 调整相机焦距和距离
- 确认使用正确的字典类型

### 标定误差过大
- 重新进行相机内参标定
- 增加采集数据对数（推荐30对）
- 确保机械臂运动充分（大范围位姿变化）
- 检查标定板是否固定牢固

### 机械臂连接失败
- 检查端口权限：`sudo chmod 666 /dev/ttyACM0`
- 确认机械臂型号和端口配置
- 检查URDF文件路径是否正确
