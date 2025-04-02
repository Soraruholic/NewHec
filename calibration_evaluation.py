#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D


def compute_reprojection_error(X, transformations_A, transformations_B, objpoints, imgpoints, camera_matrix, dist_coeffs):
    """
    通过计算标定棋盘格角点的重投影误差来评估手眼标定的质量
    
    参数:
    X - 手眼标定矩阵 (4x4)
    transformations_A - 机器人基座到末端的变换矩阵列表
    transformations_B - 相机到棋盘格的变换矩阵列表
    objpoints - 棋盘格的3D点列表
    imgpoints - 对应的2D图像点列表
    camera_matrix - 相机内参矩阵
    dist_coeffs - 畸变系数
    
    返回:
    平均重投影误差和每个姿态的误差列表
    """
    if len(transformations_A) != len(transformations_B) or len(objpoints) != len(imgpoints):
        raise ValueError("输入的变换矩阵和点数据数量不匹配")
        
    errors = []
    
    # 遍历所有位姿
    for i in range(len(transformations_A)):
        # 从基座坐标系中获取相机位姿
        # 对于手眼标定，相机在机器人末端上的变换为X
        # 基座到相机的变换为: 基座到末端的变换 * 末端到相机的变换
        T_base_to_cam = transformations_A[i] @ X
        
        # 将棋盘格角点从棋盘格坐标系转换到相机坐标系
        T_board_to_cam = transformations_B[i]
        
        # 投影3D点到图像平面
        obj_points = objpoints[i]
        img_points = imgpoints[i]
        
        # 计算理论上的棋盘格角点在相机坐标系中的位置
        points_3d_cam = []
        for point in obj_points:
            # 将点从棋盘格坐标系转换到相机坐标系
            p_board = np.append(point, 1.0)  # 齐次坐标
            p_cam = T_board_to_cam @ p_board
            points_3d_cam.append(p_cam[:3])  # 转回非齐次坐标
            
        points_3d_cam = np.array(points_3d_cam, dtype=np.float32)
        
        # 使用相机内参和畸变系数投影到图像平面
        projected_points, _ = cv2.projectPoints(
            points_3d_cam, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
        )
        
        # 计算投影点与检测到的图像点之间的误差
        print(f"img_points.shape: {img_points.shape}, projected_points.shape: {projected_points.shape}")
        
        # 将两组点数据展平为一维数组，方便计算误差
        img_points_flat = img_points.reshape(-1, 2).astype(np.float32)
        projected_points_flat = projected_points.reshape(-1, 2).astype(np.float32)
        
        # 使用reshape后的数据计算误差
        error = cv2.norm(img_points_flat, projected_points_flat, cv2.NORM_L2) / len(projected_points)
        errors.append(error)
    
    # 计算平均误差
    mean_error = np.mean(errors)
    
    return mean_error, errors


def compute_pose_consistency(X, transformations_A, transformations_B):
    """
    通过验证AX=XB方程的一致性来评估手眼标定的质量
    
    参数:
    X - 手眼标定矩阵 (4x4)
    transformations_A - 机器人基座到末端的变换矩阵列表
    transformations_B - 相机到棋盘格的变换矩阵列表
    
    返回:
    平均一致性误差（旋转和平移分量）和每个位姿对的误差列表
    """
    if len(transformations_A) != len(transformations_B):
        raise ValueError("输入的变换矩阵数量不匹配")
        
    rot_errors = []
    trans_errors = []
    
    # 对于所有可能的位姿对，检查AX=XB方程的一致性
    for i in range(len(transformations_A) - 1):
        for j in range(i + 1, len(transformations_A)):
            # 构建相对运动
            A = np.linalg.inv(transformations_A[i]) @ transformations_A[j]  # A1^-1 * A2
            B = np.linalg.inv(transformations_B[i]) @ transformations_B[j]  # B1^-1 * B2
            
            # 计算误差: AX - XB = 0
            left_side = A @ X
            right_side = X @ B
            error_matrix = left_side - right_side
            
            # 分别计算旋转和平移部分的误差
            rot_error = np.linalg.norm(error_matrix[:3, :3], 'fro')  # Frobenius范数
            trans_error = np.linalg.norm(error_matrix[:3, 3])
            
            rot_errors.append(rot_error)
            trans_errors.append(trans_error)
    
    # 计算平均误差
    mean_rot_error = np.mean(rot_errors)
    mean_trans_error = np.mean(trans_errors)
    
    return (mean_rot_error, mean_trans_error), (rot_errors, trans_errors)


def cross_validation(transformations_A, transformations_B, validation_method='leave-one-out'):
    """
    使用交叉验证方法评估手眼标定的稳定性
    
    参数:
    transformations_A - 机器人基座到末端的变换矩阵列表
    transformations_B - 相机到棋盘格的变换矩阵列表
    validation_method - 交叉验证方法，目前支持'leave-one-out'
    
    返回:
    交叉验证的结果，包括不同子集训练得到的X矩阵和它们之间的差异
    """
    import cv2
    
    if len(transformations_A) != len(transformations_B):
        raise ValueError("输入的变换矩阵数量不匹配")
        
    n = len(transformations_A)
    results = []
    
    if validation_method == 'leave-one-out':
        # 留一法交叉验证
        for i in range(n):
            # 选择除第i个样本外的所有样本进行训练
            train_A = [transformations_A[j] for j in range(n) if j != i]
            train_B = [transformations_B[j] for j in range(n) if j != i]
            
            # 提取旋转和平移部分用于OpenCV的calibrateHandEye
            R_gripper2base = []
            t_gripper2base = []
            R_target2cam = []
            t_target2cam = []
            
            for A, B in zip(train_A, train_B):
                R_gripper2base.append(A[:3, :3])
                t_gripper2base.append(A[:3, 3].reshape(3, 1))
                R_target2cam.append(B[:3, :3])
                t_target2cam.append(B[:3, 3].reshape(3, 1))
            
            # 使用OpenCV的calibrateHandEye函数
            try:
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=cv2.CALIB_HAND_EYE_TSAI
                )
                
                # 构建变换矩阵
                X_i = np.eye(4)
                X_i[:3, :3] = R_cam2gripper
                X_i[:3, 3] = t_cam2gripper.reshape(3)
                
                results.append(X_i)
            except Exception as e:
                print(f"交叉验证中第{i}轮失败: {e}")
    
    # 计算所有交叉验证结果之间的差异
    rot_diffs = []
    trans_diffs = []
    
    for i in range(len(results) - 1):
        for j in range(i + 1, len(results)):
            X_i = results[i]
            X_j = results[j]
            
            # 计算旋转差异（角度）
            R_i = X_i[:3, :3]
            R_j = X_j[:3, :3]
            
            # 计算两个旋转矩阵之间的相对旋转
            R_diff = R_i.T @ R_j
            r = R.from_matrix(R_diff)
            angle_diff = np.degrees(np.linalg.norm(r.as_rotvec()))
            
            # 计算平移差异（距离）
            t_i = X_i[:3, 3]
            t_j = X_j[:3, 3]
            trans_diff = np.linalg.norm(t_i - t_j)
            
            rot_diffs.append(angle_diff)
            trans_diffs.append(trans_diff)
    
    # 计算平均差异和标准差
    mean_rot_diff = np.mean(rot_diffs) if rot_diffs else 0
    std_rot_diff = np.std(rot_diffs) if rot_diffs else 0
    mean_trans_diff = np.mean(trans_diffs) if trans_diffs else 0
    std_trans_diff = np.std(trans_diffs) if trans_diffs else 0
    
    summary = {
        'mean_rotation_diff_degrees': mean_rot_diff,
        'std_rotation_diff_degrees': std_rot_diff,
        'mean_translation_diff_mm': mean_trans_diff * 1000,  # 转换为毫米
        'std_translation_diff_mm': std_trans_diff * 1000,  # 转换为毫米
        'validation_models': results
    }
    
    return summary


def visualize_reprojection_errors(X, transformations_A, transformations_B, objpoints, imgpoints, camera_matrix, dist_coeffs):
    """
    可视化每个标定位姿的重投影误差
    
    参数:
    X - 手眼标定矩阵 (4x4)
    transformations_A - 机器人基座到末端的变换矩阵列表
    transformations_B - 相机到棋盘格的变换矩阵列表
    objpoints - 棋盘格的3D点列表
    imgpoints - 对应的2D图像点列表
    camera_matrix - 相机内参矩阵
    dist_coeffs - 畸变系数
    """
    _, errors = compute_reprojection_error(X, transformations_A, transformations_B, objpoints, imgpoints, camera_matrix, dist_coeffs)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(errors)), errors)
    plt.xlabel('Position Index')
    plt.ylabel('Reprojection Error (pixels)')
    plt.title('Reprojection Error for Each Calibration Position')
    plt.grid(True)
    plt.savefig('reprojection_errors.png')
    plt.close()
    
    print(f"每个位姿的重投影误差:")
    for i, error in enumerate(errors):
        print(f"位姿 {i+1}: {error:.4f} 像素")
    print(f"平均重投影误差: {np.mean(errors):.4f} 像素")


def visualize_pose_errors(X, transformations_A, transformations_B):
    """
    可视化位姿一致性误差
    
    参数:
    X - 手眼标定矩阵 (4x4)
    transformations_A - 机器人基座到末端的变换矩阵列表
    transformations_B - 相机到棋盘格的变换矩阵列表
    """
    (mean_rot_error, mean_trans_error), (rot_errors, trans_errors) = \
        compute_pose_consistency(X, transformations_A, transformations_B)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.hist(rot_errors, bins=10, alpha=0.7, color='blue')
    plt.xlabel('Rotation Error (Frobenius Norm)')
    plt.ylabel('Frequency')
    plt.title(f'Rotation Errors (Mean: {mean_rot_error:.4f})')
    plt.grid(True)
    
    plt.subplot(122)
    plt.hist(trans_errors, bins=10, alpha=0.7, color='green')
    plt.xlabel('Translation Error (Euclidean Norm)')
    plt.ylabel('Frequency')
    plt.title(f'Translation Errors (Mean: {mean_trans_error:.4f})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pose_consistency_errors.png')
    plt.close()
    
    print(f"位姿一致性评估:")
    print(f"平均旋转误差: {mean_rot_error:.4f}")
    print(f"平均平移误差: {mean_trans_error:.4f} 米 ({mean_trans_error*1000:.4f} 毫米)")


def visualize_cross_validation(summary):
    """
    可视化交叉验证结果
    
    参数:
    summary - 交叉验证的结果摘要
    """
    mean_rot_diff = summary['mean_rotation_diff_degrees']
    std_rot_diff = summary['std_rotation_diff_degrees']
    mean_trans_diff = summary['mean_translation_diff_mm']
    std_trans_diff = summary['std_translation_diff_mm']
    
    print(f"交叉验证结果:")
    print(f"旋转角度差异: {mean_rot_diff:.4f} ± {std_rot_diff:.4f} 度")
    print(f"平移距离差异: {mean_trans_diff:.4f} ± {std_trans_diff:.4f} 毫米")
    
    # 创建一个3D图来可视化不同交叉验证结果的末端位置
    models = summary['validation_models']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取每个模型的平移部分
    translations = [model[:3, 3] for model in models]
    x = [t[0] for t in translations]
    y = [t[1] for t in translations]
    z = [t[2] for t in translations]
    
    # 绘制点和标签
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.scatter(xi, yi, zi, c='blue', marker='o', s=50)
        ax.text(xi, yi, zi, f'{i+1}', fontsize=12)
    
    # 计算所有点的平均位置
    mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
    ax.scatter(mean_x, mean_y, mean_z, c='red', marker='*', s=200, label='Mean Position')
    
    # 设置轴标签和标题
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Positions from Cross-Validation')
    ax.legend()
    
    plt.savefig('cross_validation_positions.png')
    plt.close()


def evaluate_calibration(X, transformations_A, transformations_B, objpoints=None, imgpoints=None, camera_matrix=None, dist_coeffs=None, 
                      visualization=True, cross_val=True):
    """
    综合评估手眼标定结果
    
    参数:
    X - 手眼标定矩阵 (4x4)
    transformations_A - 机器人基座到末端的变换矩阵列表
    transformations_B - 相机到棋盘格的变换矩阵列表
    objpoints - 棋盘格的3D点列表 (可选，用于重投影误差计算)
    imgpoints - 对应的2D图像点列表 (可选，用于重投影误差计算)
    camera_matrix - 相机内参矩阵 (可选，用于重投影误差计算)
    dist_coeffs - 畸变系数 (可选，用于重投影误差计算)
    visualization - 是否生成可视化图表
    cross_val - 是否执行交叉验证
    
    返回:
    包含各种评估指标的字典
    """
    results = {}
    
    # 1. 计算位姿一致性误差
    (mean_rot_error, mean_trans_error), (rot_errors, trans_errors) = \
        compute_pose_consistency(X, transformations_A, transformations_B)
    
    results['pose_consistency'] = {
        'mean_rotation_error': mean_rot_error,
        'mean_translation_error': mean_trans_error,
        'rotation_errors': rot_errors,
        'translation_errors': trans_errors
    }
    
    # 2. 如果提供了必要参数，计算重投影误差
    if objpoints is not None and imgpoints is not None and camera_matrix is not None and dist_coeffs is not None:
        mean_reproj_error, reproj_errors = compute_reprojection_error(
            X, transformations_A, transformations_B, objpoints, imgpoints, camera_matrix, dist_coeffs
        )
        
        results['reprojection_error'] = {
            'mean_error': mean_reproj_error,
            'errors': reproj_errors
        }
    
    # 3. 如果要求交叉验证，执行交叉验证
    if cross_val:
        cv_summary = cross_validation(transformations_A, transformations_B)
        results['cross_validation'] = cv_summary
    
    # 4. 生成可视化结果
    if visualization:
        # 可视化位姿一致性误差
        visualize_pose_errors(X, transformations_A, transformations_B)
        
        # 如果有重投影误差数据，可视化重投影误差
        if 'reprojection_error' in results:
            visualize_reprojection_errors(
                X, transformations_A, transformations_B, objpoints, imgpoints, camera_matrix, dist_coeffs
            )
        
        # 如果执行了交叉验证，可视化交叉验证结果
        if 'cross_validation' in results:
            visualize_cross_validation(results['cross_validation'])
    
    # 输出评估报告
    print("\n手眼标定评估报告:")
    print("=================\n")
    
    print("1. 位姿一致性评估:")
    print(f"   平均旋转误差: {mean_rot_error:.4f}")
    print(f"   平均平移误差: {mean_trans_error:.4f} 米 ({mean_trans_error*1000:.4f} 毫米)\n")
    
    if 'reprojection_error' in results:
        print("2. 重投影误差评估:")
        print(f"   平均重投影误差: {results['reprojection_error']['mean_error']:.4f} 像素\n")
    
    if 'cross_validation' in results:
        cv = results['cross_validation']
        print("3. 交叉验证评估:")
        print(f"   旋转稳定性: {cv['mean_rotation_diff_degrees']:.4f} ± {cv['std_rotation_diff_degrees']:.4f} 度")
        print(f"   平移稳定性: {cv['mean_translation_diff_mm']:.4f} ± {cv['std_translation_diff_mm']:.4f} 毫米\n")
    
    print("=================")
    
    return results


if __name__ == "__main__":
    # 测试示例 - 加载标定结果并评估
    try:
        # 加载标定结果
        calib_data = np.load('hand_eye_calibration_result.npz')
        hand_eye_transform = calib_data['hand_eye_transform']
        camera_matrix = calib_data['camera_matrix']
        dist_coeffs = calib_data['dist_coeffs']
        
        # 此处需要额外加载transformations_A, transformations_B, objpoints, imgpoints等数据
        # 通常这些数据会在标定过程中保存下来
        
        print("请先运行calibration.py生成评估所需的完整数据")
        
    except Exception as e:
        print(f"加载标定结果失败: {e}")
        print("请先运行标定程序生成hand_eye_calibration_result.npz文件")
