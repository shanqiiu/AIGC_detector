# -*- coding: utf-8 -*-
"""
测试相机补偿功能集成
"""

import os
import sys
import numpy as np
from video_processor import VideoProcessor

def test_camera_compensation_enabled():
    """测试启用相机补偿"""
    print("\n" + "="*70)
    print("测试1: 启用相机补偿")
    print("="*70)
    
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cpu',
        max_frames=5,
        enable_visualization=False,
        enable_camera_compensation=True
    )
    
    # 检查相机补偿器是否已初始化
    assert processor.camera_compensator is not None, "相机补偿器应该被初始化"
    assert processor.enable_camera_compensation == True, "相机补偿应该启用"
    
    print("? 相机补偿器初始化成功")
    print(f"  - 启用状态: {processor.enable_camera_compensation}")
    print(f"  - 补偿器实例: {type(processor.camera_compensator).__name__}")
    
    return True

def test_camera_compensation_disabled():
    """测试禁用相机补偿"""
    print("\n" + "="*70)
    print("测试2: 禁用相机补偿")
    print("="*70)
    
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cpu',
        max_frames=5,
        enable_visualization=False,
        enable_camera_compensation=False
    )
    
    # 检查相机补偿器是否未初始化
    assert processor.camera_compensator is None, "相机补偿器不应该被初始化"
    assert processor.enable_camera_compensation == False, "相机补偿应该禁用"
    
    print("? 相机补偿正确禁用")
    print(f"  - 启用状态: {processor.enable_camera_compensation}")
    print(f"  - 补偿器实例: {processor.camera_compensator}")
    
    return True

def test_camera_compensation_with_images():
    """测试使用图像序列进行相机补偿"""
    print("\n" + "="*70)
    print("测试3: 使用demo数据测试相机补偿")
    print("="*70)
    
    # 检查demo数据是否存在
    demo_dir = "demo_data"
    if not os.path.exists(demo_dir):
        print("? 跳过：demo_data目录不存在")
        return True
    
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cpu',
        max_frames=5,
        enable_visualization=False,
        enable_camera_compensation=True,
        camera_compensation_params={
            'ransac_thresh': 1.0,
            'max_features': 2000
        }
    )
    
    # 加载图像
    frames = processor.extract_frames_from_images(demo_dir)
    
    if len(frames) < 2:
        print("? 跳过：图像数量不足")
        return True
    
    print(f"  - 加载帧数: {len(frames)}")
    
    # 处理视频
    output_dir = "test_output_camera_comp"
    result = processor.process_video(frames[:5], output_dir=output_dir)
    
    # 验证结果
    assert 'camera_compensation_enabled' in result, "结果应包含相机补偿标志"
    assert result['camera_compensation_enabled'] == True, "相机补偿应该启用"
    assert 'camera_compensation_results' in result, "结果应包含相机补偿结果"
    
    # 检查相机补偿统计
    comp_results = result['camera_compensation_results']
    print(f"  - 补偿结果数量: {len(comp_results)}")
    
    # 统计成功的补偿
    successful = sum(1 for r in comp_results if r is not None and r['homography'] is not None)
    print(f"  - 成功补偿帧数: {successful}/{len(comp_results)}")
    
    if successful > 0:
        # 显示第一个成功补偿的详细信息
        for i, r in enumerate(comp_results):
            if r is not None and r['homography'] is not None:
                print(f"\n  第{i}帧补偿详情:")
                print(f"    - 内点数: {r['inliers']}")
                print(f"    - 总匹配数: {r['total_matches']}")
                print(f"    - 匹配率: {r['inliers']/max(r['total_matches'], 1):.1%}")
                
                # 检查光流形状
                print(f"    - 相机光流形状: {r['camera_flow'].shape}")
                print(f"    - 残差光流形状: {r['residual_flow'].shape}")
                
                # 比较幅度
                camera_mag = np.sqrt(r['camera_flow'][:,:,0]**2 + r['camera_flow'][:,:,1]**2)
                residual_mag = np.sqrt(r['residual_flow'][:,:,0]**2 + r['residual_flow'][:,:,1]**2)
                print(f"    - 相机光流平均幅度: {camera_mag.mean():.2f}")
                print(f"    - 残差光流平均幅度: {residual_mag.mean():.2f}")
                break
    
    print("\n? 相机补偿功能正常工作")
    
    # 清理测试输出
    # import shutil
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    
    return True

def test_custom_compensation_params():
    """测试自定义相机补偿参数"""
    print("\n" + "="*70)
    print("测试4: 自定义相机补偿参数")
    print("="*70)
    
    custom_params = {
        'ransac_thresh': 0.8,
        'max_features': 3000,
        'feature': 'ORB'
    }
    
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cpu',
        max_frames=5,
        enable_visualization=False,
        enable_camera_compensation=True,
        camera_compensation_params=custom_params
    )
    
    # 验证参数是否正确设置
    assert processor.camera_compensator is not None, "相机补偿器应该被初始化"
    assert processor.camera_compensator.ransac_thresh == 0.8, "RANSAC阈值应为0.8"
    assert processor.camera_compensator.max_features == 3000, "最大特征数应为3000"
    
    print("? 自定义参数设置成功")
    print(f"  - RANSAC阈值: {processor.camera_compensator.ransac_thresh}")
    print(f"  - 最大特征数: {processor.camera_compensator.max_features}")
    
    return True

def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("相机补偿功能集成测试")
    print("="*70)
    
    tests = [
        ("启用相机补偿", test_camera_compensation_enabled),
        ("禁用相机补偿", test_camera_compensation_disabled),
        ("自定义补偿参数", test_custom_compensation_params),
        ("图像序列补偿", test_camera_compensation_with_images),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"? {test_name} 失败")
        except Exception as e:
            failed += 1
            print(f"? {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n? 所有测试通过!")
        return 0
    else:
        print("\n? 部分测试失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())

