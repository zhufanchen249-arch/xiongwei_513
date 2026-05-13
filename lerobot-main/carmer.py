import cv2

# 1. 强制使用 V4L2 驱动，这在 Linux（特别是 Jetson）下最稳定
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

if not cap.isOpened():
    print("无法打开相机，请检查连接或权限！")
    exit()

# ==========================================
# 核心魔法：强制相机参数 (注意顺序：先设格式，再设尺寸)
# ==========================================
# 强制设定为 MJPG 编码 (对应终端里的 [0]: 'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# 设定分辨率 (以 1280x720 为例，你可以根据需要改成 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 强制要求 30 帧
cap.set(cv2.CAP_PROP_FPS, 30)

# 清理底层缓存，告别拖影延迟
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# ==========================================

# 打印实际生效的参数，用来验证是否设置成功
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"相机已启动！实际生效参数: {actual_w}x{actual_h} @ {actual_fps}fps")
print("按键盘上的 'q' 键退出画面。")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("无法获取画面！")
        break
        
    # 逆时针旋转 90 度
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    cv2.imshow('Camera Live Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
