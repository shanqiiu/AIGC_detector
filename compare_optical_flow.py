# -*- coding: utf-8 -*-
"""
光流算法对比脚本
比较 Farneback vs TV-L1 在静态物体动态度检测任务中的表现
"""

import numpy as np
import cv2
import time
from pathlib import Path
from simple_raft import SimpleRAFTPredictor
from static_object_analyzer import StaticObjectDynamicsCalculator


def load_test_frames(data_dir='test_data', num_frames=5):
    """加载测试帧"""
    data_path = Path(data_dir)
    frames = []
    
    for i in range(num_frames):
        frame_path = data_path / f'frame_{i:04d}.png'
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        else:
            print(f"警告: 未找到 {frame_path}")
    
    return frames


def compare_optical_flow_methods(frames):
    """对比不同光流方法"""
    
    print("=" * 70)
    print("光流算法对比：Farneback vs TV-L1")
    print("=" * 70)
    
    methods = {
        'farneback': '快速（Farneback）',
        'tvl1': '高精度（TV-L1）'
    }
    
    results = {}
    
    for method_key, method_name in methods.items():
        print(f"\n{'='*70}")
        print(f"测试方法: {method_name}")
        print(f"{'='*70}")
        
        # 创建预测器
        predictor = SimpleRAFTPredictor(method=method_key)
        calculator = StaticObjectDynamicsCalculator()
        
        # 计算光流
        start_time = time.time()
        flows = []
        for i in range(len(frames) - 1):
            flow = predictor.predict_flow(frames[i], frames[i + 1])
            flows.append(flow.transpose(1, 2, 0))  # 转换为 (H, W, 2)
        
        flow_time = time.time() - start_time
        
        # 计算静态物体动态度
        start_time = time.time()
        dynamics_results = []
        for i, flow in enumerate(flows):
            result = calculator.calculate_frame_dynamics(
                flow, frames[i], frames[i + 1]
            )
            dynamics_results.append(result)
        
        analysis_time = time.time() - start_time
        total_time = flow_time + analysis_time
        
        # 统计结果
        static_dynamics_scores = [r['static_dynamics']['dynamics_score'] 
                                 for r in dynamics_results]
        avg_dynamics = np.mean(static_dynamics_scores)
        max_dynamics = np.max(static_dynamics_scores)
        
        # 保存结果
        results[method_key] = {
            'name': method_name,
            'flow_time': flow_time,
            'analysis_time': analysis_time,
            'total_time': total_time,
            'avg_dynamics': avg_dynamics,
            'max_dynamics': max_dynamics,
            'dynamics_scores': static_dynamics_scores
        }
        
        # 输出结果
        print(f"\n性能指标:")
        print(f"  光流计算时间: {flow_time:.3f}秒 ({flow_time/len(flows)*1000:.1f}ms/帧)")
        print(f"  动态度分析时间: {analysis_time:.3f}秒")
        print(f"  总时间: {total_time:.3f}秒")
        
        print(f"\n检测结果:")
        print(f"  平均动态度: {avg_dynamics:.4f}")
        print(f"  最大动态度: {max_dynamics:.4f}")
        print(f"  各帧动态度: {[f'{s:.4f}' for s in static_dynamics_scores]}")
    
    # 对比分析
    print(f"\n{'='*70}")
    print("对比分析")
    print(f"{'='*70}")
    
    farneback = results['farneback']
    tvl1 = results['tvl1']
    
    print(f"\n速度对比:")
    print(f"  Farneback: {farneback['total_time']:.3f}秒")
    print(f"  TV-L1:     {tvl1['total_time']:.3f}秒")
    print(f"  速度比:    TV-L1 慢 {tvl1['total_time']/farneback['total_time']:.2f}倍")
    
    print(f"\n精度对比 (平均动态度):")
    print(f"  Farneback: {farneback['avg_dynamics']:.4f}")
    print(f"  TV-L1:     {tvl1['avg_dynamics']:.4f}")
    print(f"  差异:      {abs(tvl1['avg_dynamics'] - farneback['avg_dynamics']):.4f}")
    
    print(f"\n精度对比 (最大动态度):")
    print(f"  Farneback: {farneback['max_dynamics']:.4f}")
    print(f"  TV-L1:     {tvl1['max_dynamics']:.4f}")
    print(f"  差异:      {abs(tvl1['max_dynamics'] - farneback['max_dynamics']):.4f}")
    
    # 判断差异显著性
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    
    diff_ratio = abs(tvl1['avg_dynamics'] - farneback['avg_dynamics']) / farneback['avg_dynamics']
    
    if diff_ratio < 0.1:
        print("? 两种方法检测结果相近 (<10% 差异)")
        print("  建议：使用Farneback（更快）")
    elif diff_ratio < 0.3:
        print("? 两种方法有一定差异 (10-30% 差异)")
        print("  建议：根据速度/精度需求选择")
    else:
        print("! 两种方法差异较大 (>30% 差异)")
        print("  建议：使用TV-L1（更准确）")
    
    print(f"\n速度/精度权衡:")
    print(f"  Farneback: 快速，适合实时处理和快速原型")
    print(f"  TV-L1:     精确，适合离线分析和高质量检测")
    
    return results


def main():
    """主函数"""
    print("\n光流算法对比工具")
    print("用于评估不同光流算法在静态物体动态度检测任务中的表现\n")
    
    # 加载测试帧
    print("加载测试数据...")
    frames = load_test_frames('test_data', num_frames=5)
    
    if len(frames) < 2:
        frames = load_test_frames('demo_data', num_frames=10)
    
    if len(frames) < 2:
        print("错误: 需要至少2帧图像")
        print("请确保 test_data/ 或 demo_data/ 目录中有测试图像")
        return
    
    print(f"? 加载了 {len(frames)} 帧图像")
    print(f"  分辨率: {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    # 运行对比
    results = compare_optical_flow_methods(frames)
    
    print(f"\n{'='*70}")
    print("测试完成!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

