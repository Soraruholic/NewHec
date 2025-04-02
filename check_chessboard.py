#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import numpy as np

# 测试多种棋盘格尺寸
pattern_sizes = [
    (9, 6), (8, 6), (10, 7), (7, 5), (6, 9), (6, 8), (7, 7), (11, 8), (8, 5)
]

# 获取第一张图像进行测试
image_path = '/home/icrlab/NewHec/data/imgs/color_1920x1080_00_795997222.png'
image = cv2.imread(image_path)

if image is None:
    print(f"无法读取图像: {image_path}")
    exit(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 测试不同尺寸
success = False
for pattern_size in pattern_sizes:
    print(f"尝试棋盘格尺寸: {pattern_size}")
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        success = True
        print(f"成功! 检测到 {pattern_size} 棋盘格角点")
        
        # 亚像素级角点检测
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 在原始图像上绘制检测结果
        vis_img = image.copy()
        cv2.drawChessboardCorners(vis_img, pattern_size, corners, ret)
        
        # 保存可视化结果
        output_path = f"chessboard_detected_{pattern_size[0]}x{pattern_size[1]}.png"
        cv2.imwrite(output_path, vis_img)
        print(f"检测结果已保存至: {output_path}")
        
        # 只测试成功的第一个尺寸
        break

if not success:
    print("所有尺寸都无法检测到棋盘格角点!")
    print("尝试检测非标准样式的棋盘格 (CALIB_CB_ADAPTIVE_THRESH)...")
    
    for pattern_size in pattern_sizes:
        print(f"尝试棋盘格尺寸(自适应阈值): {pattern_size}")
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH)
        
        if ret:
            print(f"成功! 使用自适应阈值检测到 {pattern_size} 棋盘格角点")
            
            # 亚像素级角点检测
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 在原始图像上绘制检测结果
            vis_img = image.copy()
            cv2.drawChessboardCorners(vis_img, pattern_size, corners, ret)
            
            # 保存可视化结果
            output_path = f"chessboard_detected_adaptive_{pattern_size[0]}x{pattern_size[1]}.png"
            cv2.imwrite(output_path, vis_img)
            print(f"检测结果已保存至: {output_path}")
            break

# 如果仍然无法检测，尝试使用SimpleBlobDetector
if not success:
    print("尝试检测环形棋盘格 (CALIB_CB_ASYMMETRIC_GRID)...")
    for pattern_size in [(4, 11), (5, 7), (6, 4), (7, 5)]:
        print(f"尝试环形棋盘格尺寸: {pattern_size}")
        ret, corners = cv2.findCirclesGrid(gray, pattern_size, None, 
                                          cv2.CALIB_CB_ASYMMETRIC_GRID + 
                                          cv2.CALIB_CB_CLUSTERING)
        if ret:
            print(f"成功! 检测到 {pattern_size} 环形棋盘格角点")
            
            # 在原始图像上绘制检测结果
            vis_img = image.copy()
            cv2.drawChessboardCorners(vis_img, pattern_size, corners, ret)
            
            # 保存可视化结果
            output_path = f"circles_grid_detected_{pattern_size[0]}x{pattern_size[1]}.png"
            cv2.imwrite(output_path, vis_img)
            print(f"检测结果已保存至: {output_path}")
            break

# 将原始图像保存下来以便手动检查
resized_img = cv2.resize(image, (960, 540))  # 缩小以便查看
cv2.imwrite("original_chessboard.png", resized_img)
print("已保存原始图像以便手动检查")
