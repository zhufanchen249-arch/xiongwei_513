import depthai as dai

# 创建管道
pipeline = dai.Pipeline()
# 启动设备
with dai.Device(pipeline) as device:
    # 1. 查看 USB 速度
    print(f"USB 连接速度: {device.getUsbSpeed()}")
    
    # 2. 获取所有已连接的摄像头信息
    calibData = device.readCalibration()
    cameras = device.getConnectedCameras()
    
    print(f"\n当前相机共发现 {len(cameras)} 个传感器模块:")
    
    features = device.getConnectedCameraFeatures()
    for f in features:
        print("-" * 30)
        print(f"位置: {f.socket.name}")
        print(f"型号: {f.sensorName}")
        print(f"原生分辨率: {f.width} x {f.height}")
        print(f"支持类型: {f.supportedTypes}")

    # 3. 查看出厂内参（如果你想对比一下你标定的准不准）
    print("\n--- 正在读取出厂自带内参 (K 矩阵) ---")
    try:
        M_rgb = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB)
        print("出厂 RGB 内参矩阵:\n", np.array(M_rgb))
    except:
        print("无法读取出厂内参，可能未烧录。")