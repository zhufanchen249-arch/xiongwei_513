# SO-100/SO-101 机械臂配置与操作指南

本文档详细说明如何配置和操作 SO-100 或 SO-101 机械臂。

---

## 一、硬件识别

### 1.1 如何区分 SO-100 和 SO-101？

SO-100 和 SO-101 使用**相同的软件配置**，命令完全兼容。

主要区别是外观设计：
- SO-101：电机接头在组装后仍可轻松访问
- SO-100：电机接头在组装后不易访问，必须在组装前完成配置

### 1.2 所需设备

| 设备 | 数量 | 说明 |
|------|------|------|
| 机械臂 (follower) | 1 | 被控制的从臂 |
| 教臂 (leader) | 1 | 用于遥控操作 |
| 摄像头 (可选) | 1-2 | 用于视觉反馈 |

---

## 二、环境配置

### 2.1 安装 Python 依赖

```bash
# 确保 Python 3.12+ 已安装
python --version

# 安装 lerobot（含 feetech 电机驱动）
pip install 'lerobot[feetech]'

# 或安装完整版本
pip install 'lerobot[all]'
```

### 2.2 Linux 权限配置

```bash
# 将当前用户加入 dialout 组（需要重新登录生效）
sudo usermod -aG dialout $USER

# 或者直接修改端口权限（每次重新插拔后需重新执行）
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

# 验证权限
ls -la /dev/ttyACM*
```

### 2.3 Windows 驱动安装

1. 下载 CH340 USB 驱动：https://www.cnblogs.com/ning88/p/15792859.html
2. 安装驱动后，检查设备管理器确认 COM 端口号

### 2.4 macOS 驱动安装

```bash
# 使用 Homebrew 安装驱动
brew install --cask CH34x-USB-Serial-Driver

# 检查可用端口
ls /dev/tty.*
```

---

## 三、查找端口

### 3.1 自动查找

```bash
lerobot-find-port
```

这会显示当前可用的 USB 端口列表。按照提示，拔掉一个设备后按 Enter，再插上后会高亮显示新出现的端口。

### 3.2 手动识别

在 Linux 上运行：
```bash
ls -la /dev/ttyACM*
```

通常：
- `/dev/ttyACM0` → 第一个连接的设备（可能是 follower）
- `/dev/ttyACM1` → 第二个连接的设备（可能是 leader）

**注意**：USB 端口顺序可能因插拔顺序而变化，建议每次先运行 `lerobot-find-port` 确认。

---

## 四、配置电机 ID

这一步骤**只需要做一次**，首次设置或更换电机后才需要重新执行。

### 4.1 设置机械臂 (follower)

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

按照终端提示操作，配置完成后会显示成功信息。

### 4.2 设置教臂 (leader)

```bash
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/ttyACM1
```

---

## 五、校准

校准是**首次使用时必须执行**的步骤，用于确定每个关节的角度范围。

### 5.1 校准机械臂 (follower)

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
```

校准步骤：
1. 将机械臂移动到所有关节的**中间位置**
2. 按 **Enter** 键确认
3. 依次手动移动每个关节到其**最大**和**最小**位置
4. 重复步骤 3 直到所有关节校准完成
5. 记录校准 ID（这里是 `my_follower`）

### 5.2 校准教臂 (leader)

```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
```

校准步骤同上。

### 5.3 校准文件位置

校准文件保存在：
```bash
~/.cache/huggingface/lerobot/calibration/my_follower.yaml
~/.cache/huggingface/lerobot/calibration/my_leader.yaml
```

---

## 六、遥控操作

### 6.1 基本命令

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
```

### 6.2 带摄像头

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

### 6.3 快捷键说明

| 按键 | 功能 |
|------|------|
| → | 录制下一帧 |
| ← | 重复上一帧 |
| ESC | 结束录制 |

---

## 七、录制数据集

### 7.1 录制命令

```bash
# 设置 Hugging Face 用户名
HF_USER=$(hf auth whoami | awk -F': *' 'NR==1 {print $2}')

# 开始录制
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/my_task \
  --dataset.single_task="描述你要机器人完成的任务" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --display_data=true
```

### 7.2 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `num_episodes` | 录制 episode 数量 | 50 |
| `episode_time_s` | 每个 episode 时长（秒） | 20-45 |
| `reset_time_s` | 重置时间（秒） | 10 |
| `fps` | 帧率 | 30 |

---

## 八、回放检查

录制完成后，先回放检查数据质量：

```bash
lerobot-replay \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --dataset.repo_id=${HF_USER}/my_task \
  --dataset.episode=0
```

---

## 九、训练策略

### 9.1 使用 ACT（推荐新手）

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/my_task \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/act_my_task \
  --job_name=act_my_task \
  --batch_size=8 \
  --steps=30000 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_my_task
```

### 9.2 使用 SmolVLA（需要较大 GPU）

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/my_task \
  --policy.type=smolvla \
  --policy.device=cuda \
  --output_dir=outputs/train/smolvla_my_task \
  --job_name=smolvla_my_task \
  --batch_size=4 \
  --policy.freeze_vision_encoder=false \
  --steps=30000 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/smolvla_my_task
```

---

## 十、常见问题

### 10.1 找不到端口

```bash
# 检查 USB 设备是否被识别
lsusb

# 检查串口权限
ls -la /dev/ttyACM*
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
```

### 10.2 电机通信超时

常见原因：
1. **USB 端口顺序变化** → 重新运行 `lerobot-find-port`
2. **电机接线松动** → 检查电机链的 3-pin 电缆
3. **供电不足** → 使用 12V 2A 电源适配器
4. **电机错误状态** → 红灯闪烁表示过载或错误

### 10.3 校准失败

- 确保机械臂在通电状态
- 检查端口是否正确
- 确认电机 ID 是否冲突

### 10.4 训练时 GPU 内存不足

减小 batch size：
```bash
--batch_size=4
# 或更小
--batch_size=2
```

---

## 十一、完整工作流

```
┌─────────────────────────────────────────────────────────────┐
│  1. 安装环境                                            │
│     pip install 'lerobot[feetech]'                         │
│                                                         │
│  2. 查找端口                                            ���
│     lerobot-find-port                                    │
│                                                         │
│  3. 配置电机（仅首次）                                   │
│     lerobot-setup-motors --robot.port=...                  │
│     lerobot-setup-motors --teleop.port=...               │
│                                                         │
│  4. 校准（首次）                                      │
│     lerobot-calibrate --robot.port=... --robot.id=...          │
│     lerobot-calibrate --teleop.port=... --teleop.id=...    │
│                                                         │
│  5. 遥控测试                                            │
│     lerobot-teleoperate ...                             │
│                                                         │
│  6. 录制数据                                            │
│     lerobot-record ...                                │
│                                                         │
│  7. 回放检查                                            │
│     lerobot-replay ...                                │
│                                                         │
│  8. 训练                                                │
│     lerobot-train ...                                 │
│                                                         │
│  9. 评估                                                │
│     lerobot-eval ...                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 参考资料

- [官方文档](https://huggingface.co/docs/lerobot/index)
- [SO-101 文档](https://huggingface.co/docs/lerobot/so101)
- [SO-100 文档](https://huggingface.co/docs/lerobot/so100)
- [Hugging Face Discord](https://discord.gg/q8Dzzpym3f)