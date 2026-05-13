#!/bin/bash
# SO-100 环境设置脚本
conda activate lerobot  # 假设使用lerobot环境，根据实际情况调整

# 设置USB串口设备权限
USB_PORT="/dev/ttyACM0"
if [ -e "$USB_PORT" ]; then
    sudo chmod 666 $USB_PORT
else
    echo "警告: $USB_PORT 不存在，请检查SO-100是否已连接"
fi

# 如果使用其他端口，可以取消注释以下行
# USB_PORT_ALT="/dev/ttyUSB0"
# if [ -e "$USB_PORT_ALT" ]; then
#     sudo chmod 666 $USB_PORT_ALT
# fi