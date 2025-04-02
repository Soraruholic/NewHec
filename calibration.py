#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
# 导入标定评估模块
import calibration_evaluation as ce

def read_tcp_poses(file_path):
    """
    读取TCP位姿数据
    格式为: [x, y, z, rx, ry, rz]
    返回: 位置和欧拉角列表
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            # 删除可能存在的括号和多余空格
            line = line.strip().replace('[', '').replace(']', '')
            values = [float(val) for val in line.split(',')]
            
            if len(values) != 6:
                print(f"警告: 无效的TCP位姿数据行: {line}")
                continue
                
            # 提取位置和欧拉角
            position = values[0:3]
            euler_angles = values[3:6]  # [rx, ry, rz]
            
            poses.append((position, euler_angles))
    return poses


def pose_to_transform_matrix(position, euler_angles):
    """
    将位置和欧拉角转换为4x4变换矩阵
    """
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler('xyz', euler_angles).as_matrix()
    transform[:3, 3] = position
    return transform


def find_chessboard_corners(image_path, pattern_size=(9, 6)):
    """
    检测棋盘格角点
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # 亚像素级角点检测
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 可视化检测结果
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
        
        # 将文件名保存在角点数据中以便后续匹配
        filename = os.path.basename(image_path)
        
        return ret, corners, image, filename
    
    return False, None, image, os.path.basename(image_path)


def calibrate_camera(images_pattern, pattern_size=(9, 6), square_size=0.015):
    """
    相机标定，计算内参和畸变系数
    """
    # 准备棋盘格的三维坐标，z=0
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    
    # 存储棋盘格角点的三维坐标和二维图像坐标
    objpoints = []
    imgpoints = []
    successful_images = []
    
    # 存储文件名与角点的对应关系
    filename_to_points = {}
    
    for image_path in sorted(glob.glob(images_pattern)):
        ret, corners, image, filename = find_chessboard_corners(image_path, pattern_size)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            successful_images.append(image)
            
            # 保存文件名与角点的对应关系
            filename_to_points[filename] = corners
            
            print(f"成功检测 {filename} 中的棋盘格角点")
        else:
            print(f"未能在 {filename} 中检测到棋盘格角点")
    
    if not objpoints:
        raise Exception("未能在任何图像中检测到棋盘格角点")
    
    # 进行相机标定
    gray = cv2.cvtColor(successful_images[0], cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # 返回相机内参、畸变系数以及每张图像中棋盘格的位姿
    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, filename_to_points


def match_tcp_and_camera_poses(tcp_poses, filename_to_points, images_directory, intrinsics, dist_coeffs, objpoints, pattern_size=(9, 6), square_size=0.015):
    """
    匹配TCP位姿和相机中棋盘格的位姿
    """
    # 获取图像文件列表并排序
    image_files = sorted(glob.glob(os.path.join(images_directory, '*.png')))
    
    # 确保TCP位姿数量与图像数量匹配
    if len(tcp_poses) != len(image_files):
        print(f"警告: TCP位姿数量({len(tcp_poses)})与图像数量({len(image_files)})不匹配")
    
    # 准备棋盘格的三维坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    
    # 存储匹配的变换矩阵对
    transformations_A = []  # TCP位姿变换矩阵
    transformations_B = []  # 相机到棋盘格变换矩阵
    
    for i, image_file in enumerate(image_files):
        if i >= len(tcp_poses):
            break
            
        filename = os.path.basename(image_file)
        if filename not in filename_to_points:
            print(f"跳过 {filename} - 未检测到角点")
            continue
        
        # 获取TCP位姿
        position, euler_angles = tcp_poses[i]
        tcp_transform = pose_to_transform_matrix(position, euler_angles)
        
        # 获取角点
        corners = filename_to_points[filename]
        
        # 使用PnP算法计算棋盘格相对于相机的位姿
        _, rvec, tvec = cv2.solvePnP(objp, corners, intrinsics, dist_coeffs)
        
        # 将旋转向量转换为旋转矩阵
        rot_mat, _ = cv2.Rodrigues(rvec)
        
        # 构建相机到棋盘格的变换矩阵
        board_to_cam = np.eye(4)
        board_to_cam[:3, :3] = rot_mat
        board_to_cam[:3, 3] = tvec.flatten()
        
        # 计算基座到末端的逆变换矩阵
        base_to_end = np.linalg.inv(tcp_transform)
        assert np.allclose(np.dot(base_to_end, tcp_transform), np.eye(4))
        
        # print(f"TCP位姿: {tcp_transform}")
        # print(f"基座到末端的逆变换矩阵: {base_to_end}")
        # print(f"相机到棋盘格位姿: {board_to_cam}")
        
        transformations_A.append(base_to_end)
        transformations_B.append(board_to_cam)
        
        # print(f"匹配成功: {filename} 与 TCP位姿 #{i}")
    
    return transformations_A, transformations_B


def solve_hand_eye_calibration(transformations_A, transformations_B):
    """
    解决手眼标定问题 AX = XB
    使用Tsai的方法求解
    """
    if len(transformations_A) < 2 or len(transformations_B) < 2:
        raise Exception("至少需要2组变换矩阵对进行手眼标定")
    
    # 使用OpenCV的手眼标定函数
    try:
        # 转换为OpenCV支持的格式
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        for A, B in zip(transformations_A, transformations_B):
            R_gripper2base.append(A[:3, :3])
            t_gripper2base.append(A[:3, 3].reshape(3, 1))
            R_target2cam.append(B[:3, :3])
            t_target2cam.append(B[:3, 3].reshape(3, 1))
        
        # 使用OpenCV的calibrateHandEye函数求解
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        # 构建变换矩阵
        X = np.eye(4)
        X[:3, :3] = R_cam2gripper
        X[:3, 3] = t_cam2gripper.reshape(3)
        
        return X
        
    except Exception as e:
        print(f"使用OpenCV的手眼标定函数失败: {e}")

def visualize_calibration(X, tcp_poses, camera_intrinsics, dist_coeffs, objpoints, imgpoints, image_files):
    """
    可视化标定结果
    """
    # 为简单起见，这里只显示第一张图的重投影结果
    if not image_files or len(image_files) == 0:
        print("没有图像可供可视化")
        return
    
    image = cv2.imread(image_files[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用PnP获取棋盘格位姿
    _, rvec, tvec = cv2.solvePnP(objpoints[0], imgpoints[0], camera_intrinsics, dist_coeffs)
    
    # 投影三维点到图像平面
    projected_points, _ = cv2.projectPoints(objpoints[0], rvec, tvec, camera_intrinsics, dist_coeffs)
    
    # 在原始图像上绘制投影点
    for point in projected_points.reshape(-1, 2):
        cv2.circle(image, tuple(point.astype(int)), 3, (0, 255, 0), -1)
    
    # 在原始图像上绘制检测点
    for point in imgpoints[0].reshape(-1, 2):
        cv2.circle(image, tuple(point.astype(int)), 3, (0, 0, 255), -1)
    
    # 显示图像
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("corner points")
    plt.savefig('calibration_result.png')
    plt.close()
    
    # 打印变换矩阵
    print("\n手眼标定结果 - 相机到机器人末端的变换矩阵:")
    print(X)
    
    # 提取旋转矩阵和平移向量
    rot = X[:3, :3]
    trans = X[:3, 3]
    
    # 将旋转矩阵转换为更直观的表示形式
    r = R.from_matrix(rot)
    euler_angles = r.as_euler('xyz', degrees=True)
    quat = r.as_quat()  # [x, y, z, w]
    
    print("\n旋转矩阵:")
    print(rot)
    print("\n欧拉角 (xyz, 度):")
    print(euler_angles)
    print("\n四元数 [x, y, z, w]:")
    print(quat)
    print("\n平移向量:")
    print(trans)
    print("\n逆变换:")
    print(np.linalg.inv(X))
    print("\n逆变换旋转角度:")
    r = R.from_matrix(np.linalg.inv(X)[:3, :3])
    print(r.as_euler('xyz', degrees=True))
    print("\n逆变换平移向量:")
    print(np.linalg.inv(X)[:3, 3])
    
    


def main():
    # 设置参数
    data_dir = './data'
    images_dir = os.path.join(data_dir, 'imgs')
    tcp_file = os.path.join(data_dir, 'tcp_poses.txt')
    pattern_size = (11, 8)  # 棋盘格内角点数量 (宽, 高)
    square_size = 0.015  # 棋盘格方格尺寸，单位米
    
    # 读取TCP位姿
    tcp_poses = read_tcp_poses(tcp_file)
    print(f"读取到 {len(tcp_poses)} 个TCP位姿")
    
    # 加载所有PNG图像
    images_pattern = os.path.join(images_dir, '*.png')
    image_files = sorted(glob.glob(images_pattern))
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 相机标定
    print("\n执行相机标定...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs, objpoints, imgpoints, filename_to_points = calibrate_camera(
        images_pattern, pattern_size, square_size
    )
    
    print(f"相机标定完成，重投影误差: {ret}")
    print("\n相机内参矩阵:")
    print(camera_matrix)
    print("\n畸变系数:")
    print(dist_coeffs)
    
    # 匹配TCP位姿和相机中棋盘格的位姿
    print("\n匹配TCP位姿和相机位姿...")
    transformations_A, transformations_B = match_tcp_and_camera_poses(
        tcp_poses, filename_to_points, images_dir, camera_matrix, dist_coeffs, objpoints[0], pattern_size, square_size
    )
    
    print(f"成功匹配 {len(transformations_A)} 组位姿")
    
    # 手眼标定
    print("\n执行手眼标定...")
    X = solve_hand_eye_calibration(transformations_A, transformations_B)
    
    # 分别保存 transformations_A 和 transformations_B
    np.savez('board2camera.npz', transformations_A=transformations_A)
    np.savez('base2end.npz', transformations_B=transformations_B)
    
    # 可视化标定结果
    visualize_calibration(X, tcp_poses, camera_matrix, dist_coeffs, objpoints, imgpoints, image_files)
    
    # 评估标定结果
    # print("\n执行标定质量评估...")
    # evaluation_results = ce.evaluate_calibration(
    #     X, transformations_A, transformations_B, 
    #     objpoints, imgpoints, camera_matrix, dist_coeffs,
    #     visualization=True, cross_val=True
    # )
    
    # 保存标定结果
    np.savez('hand_eye_calibration_result.npz',
            camera_matrix=camera_matrix, 
            dist_coeffs=dist_coeffs,
            hand_eye_transform=X,
            transformations_A=transformations_A,
            transformations_B=transformations_B)
    
    print("\n标定结果已保存至 hand_eye_calibration_result.npz")


if __name__ == "__main__":
    main()
