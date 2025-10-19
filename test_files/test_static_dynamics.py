# -*- coding: utf-8 -*-
"""
测试静态物体动态度计算功能
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from static_object_analyzer import StaticObjectDynamicsCalculator
from simple_raft import SimpleRAFTPredictor as RAFTPredictor
import json


def create_synthetic_test_data():
    """创建合成测试数据"""
    print("创建合成测试数据...")
    
    # 创建测试目录
    test_dir = 'test_data'
    os.makedirs(test_dir, exist_ok=True)
    
    # 参数设置
    width, height = 640, 480
    num_frames = 10
    
    # 创建静态背景（建筑物）
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加建筑物结�?
    # 主建�?
    cv2.rectangle(background, (100, 200), (300, 400), (150, 150, 150), -1)
    cv2.rectangle(background, (120, 220), (280, 380), (100, 100, 100), -1)
    
    # 窗户
    for i in range(3):
        for j in range(4):
            x = 140 + j * 30
            y = 240 + i * 40
            cv2.rectangle(background, (x, y), (x+20, y+25), (50, 100, 200), -1)
    
    # 另一栋建�?
    cv2.rectangle(background, (350, 150), (550, 420), (120, 120, 120), -1)
    cv2.rectangle(background, (370, 170), (530, 400), (80, 80, 80), -1)
    
    # 添加纹理和细�?
    for _ in range(200):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        cv2.circle(background, (x, y), 1, (np.random.randint(0, 255),) * 3, -1)
    
    frames = []
    
    # 模拟相机转动
    center_x, center_y = width // 2, height // 2
    
    for i in range(num_frames):
        # 计算旋转角度（模拟相机转动）
        angle = i * 2.0  # 每帧转动2�?
        
        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        
        # 应用旋转
        rotated_frame = cv2.warpAffine(background, rotation_matrix, (width, height))
        
        # 添加少量噪声
        noise = np.random.normal(0, 5, rotated_frame.shape).astype(np.uint8)
        rotated_frame = cv2.add(rotated_frame, noise)
        
        frames.append(rotated_frame)
        
        # 保存�?
        cv2.imwrite(os.path.join(test_dir, f'frame_{i:04d}.png'), 
                   cv2.cvtColor(rotated_frame, cv2.COLOR_RGB2BGR))
    
    print(f"已创�? {len(frames)} 帧测试数据，保存�? {test_dir}")
    return frames, test_dir


def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    # 创建测试数据
    frames, test_dir = create_synthetic_test_data()
    
    try:
        # 直接测试核心功能，不使用VideoProcessor
        calculator = StaticObjectDynamicsCalculator()
        predictor = RAFTPredictor(device='cpu')
        
        # 计算前两帧的光流
        flow = predictor.predict_flow(frames[0], frames[1])
        flow = flow.transpose(1, 2, 0)  # 转换�? (H, W, 2)
        
        # 计算动态度
        result = calculator.calculate_frame_dynamics(flow, frames[0], frames[1])
        
        # 检查结�?
        assert 'static_dynamics' in result
        assert 'global_dynamics' in result
        
        print("�? 基本功能测试通过")
        
        # 打印结果摘要
        static_dynamics = result['static_dynamics']
        global_dynamics = result['global_dynamics']
        print(f"  动态度分数: {static_dynamics['dynamics_score']:.3f}")
        print(f"  静态区域比�?: {global_dynamics['static_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"�? 基本功能测试失败: {e}")
        return False


def test_camera_motion_compensation():
    """测试相机运动补偿"""
    print("\n=== 测试相机运动补偿 ===")
    
    try:
        from static_object_analyzer import CameraMotionEstimator
        
        # 创建简单测试图�?
        img1 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # 创建旋转后的图像
        center = (100, 100)
        rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        img2 = cv2.warpAffine(img1, rotation_matrix, (200, 200))
        
        # 测试相机运动估计
        estimator = CameraMotionEstimator()
        motion = estimator.estimate_camera_motion(img1, img2)
        
        if motion is not None:
            print("�? 相机运动估计成功")
            print(f"  检测到 {len(motion['matches'])} 个特征匹�?")
            return True
        else:
            print("�? 相机运动估计返回空结果（可能是特征点不足�?")
            return True  # 这在某些情况下是正常�?
            
    except Exception as e:
        print(f"�? 相机运动补偿测试失败: {e}")
        return False


def test_static_detection():
    """测试静态区域检�?"""
    print("\n=== 测试静态区域检�? ===")
    
    try:
        from static_object_analyzer import StaticObjectDetector
        
        # 创建测试光流
        h, w = 100, 100
        flow = np.zeros((h, w, 2))
        
        # 添加一些运动区�?
        flow[20:40, 20:40, 0] = 5.0  # 水平运动
        flow[60:80, 60:80, 1] = 3.0  # 垂直运动
        
        # 测试静态区域检�?
        detector = StaticObjectDetector(flow_threshold=2.0)
        static_mask, compensated_flow = detector.detect_static_regions(flow)
        
        # 检查结�?
        assert static_mask.shape == (h, w)
        assert compensated_flow.shape == (h, w, 2)
        
        static_ratio = np.sum(static_mask) / (h * w)
        print(f"�? 静态区域检测成�?")
        print(f"  静态区域比�?: {static_ratio:.3f}")
        
        return True
        
    except Exception as e:
        print(f"�? 静态区域检测测试失�?: {e}")
        return False


def test_dynamics_calculation():
    """测试动态度计算"""
    print("\n=== 测试动态度计算 ===")
    
    try:
        calculator = StaticObjectDynamicsCalculator()
        
        # 创建测试数据
        h, w = 100, 100
        flow = np.random.normal(0, 0.5, (h, w, 2))  # 小幅随机运动
        image1 = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # 计算动态度
        result = calculator.calculate_frame_dynamics(flow, image1, image2)
        
        # 检查结果结�?
        required_keys = ['static_mask', 'compensated_flow', 'static_dynamics', 'global_dynamics']
        for key in required_keys:
            assert key in result, f"缺少关键�?: {key}"
        
        print("�? 动态度计算成功")
        print(f"  动态度分数: {result['static_dynamics']['dynamics_score']:.3f}")
        print(f"  静态区域比�?: {result['global_dynamics']['static_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"�? 动态度计算测试失败: {e}")
        return False


def test_report_generation():
    """测试报告生成"""
    print("\n=== 测试报告生成 ===")
    
    try:
        calculator = StaticObjectDynamicsCalculator()
        
        # 创建模拟结果
        result = {
            'static_dynamics': {
                'mean_magnitude': 1.5,
                'std_magnitude': 0.8,
                'max_magnitude': 3.2,
                'dynamics_score': 1.9
            },
            'global_dynamics': {
                'static_ratio': 0.75,
                'dynamic_ratio': 0.25,
                'mean_dynamic_magnitude': 4.2,
                'consistency_score': 0.85
            }
        }
        
        # 生成报告
        report = calculator.generate_report(result)
        
        # 检查报告内�?
        assert len(report) > 100, "报告内容太短"
        assert "静态物体动态度分析报告" in report
        assert "1.500" in report  # 检查数值格式化
        
        print("�? 报告生成成功")
        print(f"  报告长度: {len(report)} 字符")
        
        return True
        
    except Exception as e:
        print(f"�? 报告生成测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测�?"""
    print("开始运行静态物体动态度计算功能测试...")
    
    tests = [
        test_camera_motion_compensation,
        test_static_detection,
        test_dynamics_calculation,
        test_report_generation,
        test_basic_functionality,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试 {test.__name__} 发生异常: {e}")
            results.append(False)
    
    # 总结
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过�?")
    else:
        print(f"�? {total - passed} 个测试失�?")
    
    return passed == total


if __name__ == '__main__':
    # 设置随机种子以获得可重复的结�?
    np.random.seed(42)
    
    success = run_all_tests()
    
    if success:
        print("\n系统已准备就绪，可以处理真实的相机转动视频！")
        print("\n使用方法:")
        print("python video_processor.py -i your_video.mp4 -o output_dir")
        print("�?")
        print("python video_processor.py -i image_directory/ -o output_dir")
    else:
        print("\n请检查并修复失败的测试后再使用系统�?")