#test_oakd_standard.py
# Luxonis OAK-D Pro 相机测试 (标准全速模式)

import cv2
import depthai as dai
import os

# 针对 SSH 远程终端运行 OpenCV 弹窗的常见报错处理
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0" 

# 1. 创建数据管道 (Pipeline)
pipeline = dai.Pipeline()

# 2. 创建相机节点 (RGB和左右双目)
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

# 3. 配置相机参数 (标准全速模式，不限制 FPS)
# RGB 彩色相机配置 (1080P)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)

# 左目灰度相机配置 (400P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# 右目灰度相机配置 (400P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# 4. 创建数据输出通道 (XLinkOut)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")
monoLeft.out.link(xoutLeft.input)

xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")
monoRight.out.link(xoutRight.input)

# 5. 连接设备并抓取画面显示
print("正在以标准模式连接 OAK-D Pro 相机 (需要 5000M USB 3.0 带宽)...")
try:
    with dai.Device(pipeline) as device:
        print("连接成功！正在全速接收画面... (按小写 'q' 键退出)")
        
        # 获取输出队列
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

        while True:
            # 尝试从队列中获取最新的画面帧
            inRgb = qRgb.tryGet()
            inLeft = qLeft.tryGet()
            inRight = qRight.tryGet()

            # 使用 OpenCV 显示画面
            if inRgb is not None:
                cv2.imshow("RGB Color", inRgb.getCvFrame())
            
            if inLeft is not None:
                cv2.imshow("Left Mono", inLeft.getCvFrame())
                
            if inRight is not None:
                cv2.imshow("Right Mono", inRight.getCvFrame())

            # 监听键盘，按 'q' 退出循环
            if cv2.waitKey(1) == ord('q'):
                break
                
except Exception as e:
    print(f"\n运行出错啦: {e}")
    print("排查建议：如果是满血测试失败，请再次用 `lsusb -t` 确认是否跑在 5000M。或者检查是否需要用 Y 型线辅助供电。")

# 清理并关闭所有 OpenCV 窗口
cv2.destroyAllWindows()
