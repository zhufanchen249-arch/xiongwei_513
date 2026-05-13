#!/usr/bin/env python3
import cv2
import time
from ultralytics import YOLO

def main():
    print("⏳ 正在加载 YOLOv8 模型...")
    # 注意：等我们下面转好模型后，这里要换成 yolov8n.engine
    model = YOLO("yolov8n.pt") 
    print("✅ 模型加载成功！")

    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 极度关键：实时模式下，必须把缓存设为 1，否则会有严重的画面延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_time = time.time()

    print("🚀 实时视觉启动！请在镜头前晃动物体。按 [q] 退出。")

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # 强制清空旧帧，保证拿到的绝对是当前毫秒的画面
        cap.grab()

        # 实时预测！去掉了 verbose=False，终端就不会一直刷屏了
        # stream=True 是实时视频流加速的终极秘籍，它使用生成器，极大降低内存占用
        results = model.predict(frame, conf=0.4, imgsz=320, verbose=False, stream=True)
        
        # 实时解析结果
        for result in results:
            annotated_frame = result.plot()
            
            # 提取坐标逻辑（供你后续联动机械臂使用）
            # boxes = result.boxes
            # if len(boxes) > 0:
            #     box = boxes[0].xyxy[0]
            #     u, v = (box[0] + box[2])/2, (box[1] + box[3])/2

            # 计算并显示 FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Real-Time", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
