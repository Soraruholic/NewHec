#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import calibration_evaluation as ce


def main(result_file='hand_eye_calibration_result.npz', visualization=True, cross_validation=True):
    """
    从保存的标定结果中加载数据并评估手眼标定质量
    
    参数:
        result_file: 手眼标定结果文件路径
        visualization: 是否生成可视化结果
        cross_validation: 是否执行交叉验证
    """
    try:
        print(f"\n加载标定结果: {result_file}")
        data = np.load(result_file, allow_pickle=True)
        
        # 提取手眼标定结果中的内外参和畸变系数
        X = data['hand_eye_transform']
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        
        # 检查是否包含计算评估所需的变换数据
        has_transform_data = 'transformations_A' in data and 'transformations_B' in data
        
        if has_transform_data:
            transformations_A = data['transformations_A']
            transformations_B = data['transformations_B']
            
            print("\n标定结果:")
            print("相机内参:")
            print(camera_matrix)
            print("\n畸变系数:")
            print(dist_coeffs)
            print("\n手眼标定结果:")
            print(X)
            
            # 提取旋转矩阵和平移向量
            rot = X[:3, :3]
            trans = X[:3, 3]
            
            # 计算旋转矩阵的欧拉角和四元数表示
            from scipy.spatial.transform import Rotation as R
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
            print("\n逆变换矩阵:")
            print(np.linalg.inv(X))
            
            # 评估手眼标定质量
            print("\n评估手眼标定质量...")
            
            # 检查是否包含计算评估所需的点对应数据
            objpoints = None
            imgpoints = None
            if 'objpoints' in data and 'imgpoints' in data:
                objpoints = data['objpoints']
                imgpoints = data['imgpoints']
            
            # 评估标定质量
            evaluation_results = ce.evaluate_calibration(
                X, transformations_A, transformations_B,
                objpoints, imgpoints, camera_matrix, dist_coeffs,
                visualization=visualization, cross_val=cross_validation
            )
            
            # 输出评估结果
            pose_consistency = evaluation_results.get('pose_consistency', {})
            rot_error = pose_consistency.get('mean_rotation_error', 0)
            trans_error = pose_consistency.get('mean_translation_error', 0) * 1000  # mm
            
            # 评估标定质量等级
            # 注意：此处的质量等级评估仅供参考，实际应用中应根据具体需求调整
            if rot_error < 0.1 and trans_error < 1.0:
                quality = "优"
            elif rot_error < 0.3 and trans_error < 3.0:
                quality = "良"
            elif rot_error < 0.5 and trans_error < 5.0:
                quality = "中"
            else:
                quality = "差"
            
            print(f"\n标定质量等级: {quality}")
            recommendation = "可以直接使用该标定结果" if quality != "差" else "建议重新标定"
            print(f"建议: {recommendation}")
            
        else:
            print("\n警告: 标定结果文件不包含评估所需的变换数据。")
            print("请使用新版标定工具生成包含评估所需数据的标定结果文件。")
    
    except Exception as e:
        print(f"\n评估过程中出现错误: {e}")
        print("请检查标定结果文件是否存在且包含所需数据。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估手眼标定质量")
    parser.add_argument("--result_file", type=str, default="hand_eye_calibration_result.npz",
                        help="手眼标定结果文件路径")
    parser.add_argument("--no_visualization", action="store_false", dest="visualization",
                        help="禁用可视化结果输出")
    parser.add_argument("--no_cross_validation", action="store_false", dest="cross_validation",
                        help="禁用交叉验证")
    
    args = parser.parse_args()
    main(args.result_file, args.visualization, args.cross_validation)
