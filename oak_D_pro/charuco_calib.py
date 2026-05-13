#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OAK-D Pro 动态组合标定工具 (可任意开关 RGB / Left / Right)
"""

import cv2
from cv2 import aruco
import numpy as np
import depthai as dai
import datetime

# ================= 1. 总控制台 (开关区) =================
# 想要标定哪个镜头，就把它改成 True，不用的改成 False！
# 比如只测双目，就把 RGB 改成 False。
ENABLE_LEFT  = True   # 开关：左黑白相机
ENABLE_RIGHT = False   # 开关：右黑白相机
ENABLE_RGB   = True  # 开关：中间彩色相机
# ========================================================

# ================= 2. 物理参数配置区 =================
SQUARE_LENGTH = 0.030    # 黑白大格尺寸 (米)
MARKER_LENGTH = 0.024    # ArUco 二维码尺寸 (米)
BOARD_SIZE = (5, 7)     # 真实格子数 (列, 行)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
board = aruco.CharucoBoard_create(BOARD_SIZE[0], BOARD_SIZE[1], SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

# 数据存储池
data_pool = {
    'rgb_corners': [], 'rgb_ids': [],
    'left_corners': [], 'left_ids': [],
    'right_corners': [], 'right_ids': []
}
img_sizes = {'rgb': None, 'left': None, 'right': None}

# ================= 辅助函数 (保持不变) =================
def detect_charuco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    if ids is not None and len(ids) > 0:
        res, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if res > 0:
            return True, ch_corners, ch_ids
    return False, None, None

def get_shared_points(corners1_list, ids1_list, corners2_list, ids2_list):
    obj_pts, img_pts1, img_pts2 = [], [], []
    for c1, id1, c2, id2 in zip(corners1_list, ids1_list, corners2_list, ids2_list):
        id1_list, id2_list = id1.flatten().tolist(), id2.flatten().tolist()
        common_ids = set(id1_list).intersection(set(id2_list))
        if len(common_ids) < 6: continue 
        frame_obj, frame_img1, frame_img2 = [], [], []
        for cid in common_ids:
            idx1, idx2 = id1_list.index(cid), id2_list.index(cid)
            frame_obj.append(board.chessboardCorners[cid])
            frame_img1.append(c1[idx1][0])
            frame_img2.append(c2[idx2][0])
        obj_pts.append(np.array(frame_obj, dtype=np.float32))
        img_pts1.append(np.array(frame_img1, dtype=np.float32))
        img_pts2.append(np.array(frame_img2, dtype=np.float32))
    return obj_pts, img_pts1, img_pts2

# ================= 初始化 OAK 管道 (动态铺水管) =================
print("正在初始化 OAK-D 相机管线...")
pipeline = dai.Pipeline()

if ENABLE_RGB:
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.video.link(xoutRgb.input)

if ENABLE_LEFT:
    camLeft = pipeline.create(dai.node.MonoCamera)
    camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutLeft.setStreamName("left")
    camLeft.out.link(xoutLeft.input)

if ENABLE_RIGHT:
    camRight = pipeline.create(dai.node.MonoCamera)
    camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    xoutRight = pipeline.create(dai.node.XLinkOut)
    xoutRight.setStreamName("right")
    camRight.out.link(xoutRight.input)

# ================= 图像采集与检测 =================
with dai.Device(pipeline) as device:
    print("OAK 相机连接成功！操作指南：[C]采集 [S]解算 [Q]退出")
    
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False) if ENABLE_RGB else None
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False) if ENABLE_LEFT else None
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False) if ENABLE_RIGHT else None

    sample_count = 0

    while True:
        display_list = [] # 用来把开启的画面拼在一起
        ready_to_capture = True # 默认是可以采集的，下面会用严格条件去卡它
        
        # --- 处理左目 ---
        if ENABLE_LEFT:
            frame_left = qLeft.get().getCvFrame()
            if img_sizes['left'] is None: img_sizes['left'] = (frame_left.shape[1], frame_left.shape[0])
            ok_l, c_l, id_l = detect_charuco(frame_left)
            disp_left = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2BGR)
            if ok_l: aruco.drawDetectedCornersCharuco(disp_left, c_l, id_l, (0, 255, 0))
            display_list.append(disp_left)
            # 左目必须看清 10 个以上角点才能采集
            ready_to_capture = ready_to_capture and (ok_l and len(id_l) > 10)

        # --- 处理 RGB ---
        if ENABLE_RGB:
            frame_rgb = qRgb.get().getCvFrame()
            if img_sizes['rgb'] is None: img_sizes['rgb'] = (frame_rgb.shape[1], frame_rgb.shape[0])
            ok_rgb, c_rgb, id_rgb = detect_charuco(frame_rgb)
            disp_rgb = frame_rgb.copy()
            if ok_rgb: aruco.drawDetectedCornersCharuco(disp_rgb, c_rgb, id_rgb, (0, 255, 0))
            # 缩放一下，不然和黑白图拼不起来
            if ENABLE_LEFT: 
                disp_rgb = cv2.resize(disp_rgb, (disp_left.shape[1], disp_left.shape[0]))
            display_list.append(disp_rgb)
            # RGB 必须看清 10 个以上角点才能采集
            ready_to_capture = ready_to_capture and (ok_rgb and len(id_rgb) > 10)

        # --- 处理右目 ---
        if ENABLE_RIGHT:
            frame_right = qRight.get().getCvFrame()
            if img_sizes['right'] is None: img_sizes['right'] = (frame_right.shape[1], frame_right.shape[0])
            ok_r, c_r, id_r = detect_charuco(frame_right)
            disp_right = cv2.cvtColor(frame_right, cv2.COLOR_GRAY2BGR)
            if ok_r: aruco.drawDetectedCornersCharuco(disp_right, c_r, id_r, (0, 255, 0))
            display_list.append(disp_right)
            # 右目必须看清 10 个以上角点才能采集
            ready_to_capture = ready_to_capture and (ok_r and len(id_r) > 10)

        # 动态拼接屏幕
        if len(display_list) > 0:
            combined_disp = np.hstack(display_list)
            cv2.putText(combined_disp, f"Valid Samples: {sample_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.imshow("OAK Dynamic Calibration", combined_disp)

        key = cv2.waitKey(1)
        if key == ord('c'):
            if ready_to_capture:
                if ENABLE_LEFT: data_pool['left_corners'].append(c_l); data_pool['left_ids'].append(id_l)
                if ENABLE_RGB: data_pool['rgb_corners'].append(c_rgb); data_pool['rgb_ids'].append(id_rgb)
                if ENABLE_RIGHT: data_pool['right_corners'].append(c_r); data_pool['right_ids'].append(id_r)
                sample_count += 1
                print(f"✅ 成功采集第 {sample_count} 组数据！")
            else:
                print("❌ 视野不完整！必须所有已开启的相机都看清标定板！")

        elif key == ord('s'):
            if sample_count < 10:
                print(f"⚠️ 样本太少 (当前:{sample_count}), 建议至少 10 组！")
            else:
                print("\n🚀 停止采集，开始矩阵运算...")
                break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # ================= 核心计算环节 (动态解算) =================
    # 准备一个大字典，把算出来的结果都存起来
    results = {}
    flags = cv2.CALIB_FIX_INTRINSIC

    print("\n--- 解算单目内参 ---")
    if ENABLE_LEFT:
        ret_l, K_l, D_l, _, _ = aruco.calibrateCameraCharuco(data_pool['left_corners'], data_pool['left_ids'], board, img_sizes['left'], None, None)
        results['K_left'] = K_l; results['D_left'] = D_l
        print(f"Left 内参 RMSE: {ret_l:.3f}")

    if ENABLE_RIGHT:
        ret_r, K_r, D_r, _, _ = aruco.calibrateCameraCharuco(data_pool['right_corners'], data_pool['right_ids'], board, img_sizes['right'], None, None)
        results['K_right'] = K_r; results['D_right'] = D_r
        print(f"Right 内参 RMSE: {ret_r:.3f}")

    if ENABLE_RGB:
        ret_rgb, K_rgb, D_rgb, _, _ = aruco.calibrateCameraCharuco(data_pool['rgb_corners'], data_pool['rgb_ids'], board, img_sizes['rgb'], None, None)
        results['K_rgb'] = K_rgb; results['D_rgb'] = D_rgb
        print(f"RGB 内参 RMSE: {ret_rgb:.3f}")

    print("\n--- 解算相机外参 ---")
    # 只有左边和右边都开了，才算左右双目外参
    if ENABLE_LEFT and ENABLE_RIGHT:
        objPts_lr, imgPts_l, imgPts_r = get_shared_points(data_pool['left_corners'], data_pool['left_ids'], data_pool['right_corners'], data_pool['right_ids'])
        ret_stereo_lr, _, _, _, _, R_lr, T_lr, _, _ = cv2.stereoCalibrate(objPts_lr, imgPts_l, imgPts_r, K_l, D_l, K_r, D_r, img_sizes['left'], flags=flags)
        results['R_Left_to_Right'] = R_lr; results['T_Left_to_Right'] = T_lr
        print(f"Left-Right 外参 RMSE: {ret_stereo_lr:.3f}")

    # 只有左边和 RGB 都开了，才算 Left-RGB 外参
    if ENABLE_LEFT and ENABLE_RGB:
        objPts_lrgb, imgPts_l2, imgPts_rgb = get_shared_points(data_pool['left_corners'], data_pool['left_ids'], data_pool['rgb_corners'], data_pool['rgb_ids'])
        ret_stereo_lrgb, _, _, _, _, R_lrgb, T_lrgb, _, _ = cv2.stereoCalibrate(objPts_lrgb, imgPts_l2, imgPts_rgb, K_l, D_l, K_rgb, D_rgb, img_sizes['rgb'], flags=flags)
        results['R_Left_to_RGB'] = R_lrgb; results['T_Left_to_RGB'] = T_lrgb
        print(f"Left-RGB 外参 RMSE: {ret_stereo_lrgb:.3f}")

    # ================= 保存文件 =================
    filename = f"oak_calibration_custom_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    for key, value in results.items():
        fs.write(key, value)
    fs.release()

    print("\n" + "="*50)
    print(f"🎉 标定完成！你开启的项已全部计算并保存至: {filename}")
    print("="*50)