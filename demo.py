# -*- coding: utf-8 -*-
"""
静态物体动态度分析系统演示
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from simple_raft import SimpleRAFTPredictor
from static_object_analyzer import StaticObjectDynamicsCalculator


def create_demo_video():
    """创建演示视频"""
    print("创建演示视频...")
    
    # 创建输出目录
    demo_dir = 'demo_data'
    os.makedirs(demo_dir, exist_ok=True)
    
    # 视频参数
    width, height = 640, 480
    num_frames = 20
    
    # 创建静态背景场景（建筑物）
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 绘制建筑�?
    # 主建�?
    cv2.rectangle(background, (150, 200), (350, 450), (120, 120, 120), -1)
    cv2.rectangle(background, (170, 220), (330, 430), (80, 80, 80), -1)
    
    # 窗户
    for i in range(3):
        for j in range(4):
            x = 190 + j * 25
            y = 250 + i * 35
            cv2.rectangle(background, (x, y), (x+15, y+25), (50, 100, 200), -1)
    
    # 另一栋建�?
    cv2.rectangle(background, (400, 150), (580, 420), (100, 100, 100), -1)
    cv2.rectangle(background, (420, 170), (560, 400), (60, 60, 60), -1)
    
    # 添加窗户
    for i in range(4):
        for j in range(2):
            x = 440 + j * 40
            y = 200 + i * 40
            cv2.rectangle(background, (x, y), (x+20, y+25), (100, 150, 50), -1)
    
    # 地面
    cv2.rectangle(background, (0, 420), (width, height), (40, 80, 40), -1)
    
    # 天空
    cv2.rectangle(background, (0, 0), (width, 150), (135, 206, 235), -1)
    
    # 添加一些纹�?
    for _ in range(500):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        color = np.random.randint(0, 50, 3)
        cv2.circle(background, (x, y), 1, color.tolist(), -1)
    
    frames = []
    
    # 模拟相机水平转动
    center_x, center_y = width // 2, height // 2
    
    for i in range(num_frames):
        # 计算水平平移（模拟相机转动）
        shift_x = int(i * 3)  # 每帧向右移动3像素
        
        # 创建平移矩阵
        M = np.float32([[1, 0, -shift_x], [0, 1, 0]])
        
        # 应用平移
        shifted_frame = cv2.warpAffine(background, M, (width, height))
        
        # 添加轻微的噪�?
        noise = np.random.normal(0, 3, shifted_frame.shape).astype(np.int16)
        shifted_frame = np.clip(shifted_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        frames.append(shifted_frame)
        
        # 保存�?
        cv2.imwrite(os.path.join(demo_dir, f'frame_{i:04d}.png'), 
                   cv2.cvtColor(shifted_frame, cv2.COLOR_RGB2BGR))
    
    print(f"已创�? {len(frames)} 帧演示数�?")
    return frames, demo_dir


def analyze_demo_video():
    """分析演示视频"""
    print("\n开始分析演示视�?...")
    
    # 创建演示数据
    frames, demo_dir = create_demo_video()
    
    # 初始化分析器
    predictor = SimpleRAFTPredictor(device='cpu')
    calculator = StaticObjectDynamicsCalculator()
    
    # 分析结果
    results = []
    flows = []
    
    print("计算光流和动态度...")
    for i in range(len(frames) - 1):
        # 计算光流
        flow = predictor.predict_flow(frames[i], frames[i + 1])
        flow = flow.transpose(1, 2, 0)  # 转换�? (H, W, 2)
        flows.append(flow)
        
        # 计算动态度
        result = calculator.calculate_frame_dynamics(flow, frames[i], frames[i + 1])
        results.append(result)
        
        print(f"  �? {i}: 动态度分数 = {result['static_dynamics']['dynamics_score']:.3f}, "
              f"静态区域比�? = {result['global_dynamics']['static_ratio']:.3f}")
    
    # 计算时序统计
    temporal_stats = calculator.calculate_temporal_statistics(results)
    
    print(f"\n时序统计结果:")
    print(f"  平均动态度分数: {temporal_stats['mean_dynamics_score']:.3f}")
    print(f"  动态度分数标准�?: {temporal_stats['std_dynamics_score']:.3f}")
    print(f"  平均静态区域比�?: {temporal_stats['mean_static_ratio']:.3f}")
    print(f"  时序稳定�?: {temporal_stats['temporal_stability']:.3f}")
    
    # 生成可视�?
    create_demo_visualizations(frames, flows, results, temporal_stats)
    
    return results, temporal_stats


def create_demo_visualizations(frames, flows, results, temporal_stats):
    """创建演示可视�?"""
    print("\n生成可视化结�?...")
    
    vis_dir = 'demo_output'
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. 显示关键帧分�?
    key_frame_idx = len(frames) // 2
    if key_frame_idx < len(results):
        fig = create_frame_analysis_plot(
            frames[key_frame_idx], 
            flows[key_frame_idx], 
            results[key_frame_idx]
        )
        fig.savefig(os.path.join(vis_dir, 'key_frame_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # 2. 时序动态度曲线
    fig = create_temporal_plot(results, temporal_stats)
    fig.savefig(os.path.join(vis_dir, 'temporal_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. 生成报告
    report = generate_demo_report(results, temporal_stats)
    with open(os.path.join(vis_dir, 'demo_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"可视化结果已保存�?: {vis_dir}")


def create_frame_analysis_plot(image, flow, result):
    """创建单帧分析�?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 光流幅度
    flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    im1 = axes[0, 1].imshow(flow_magnitude, cmap='jet')
    axes[0, 1].set_title(f'Flow Magnitude (Max: {flow_magnitude.max():.2f})')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 补偿后的光流
    compensated_flow = result['compensated_flow']
    compensated_magnitude = np.sqrt(compensated_flow[:, :, 0]**2 + compensated_flow[:, :, 1]**2)
    im2 = axes[1, 0].imshow(compensated_magnitude, cmap='jet')
    axes[1, 0].set_title(f'Compensated Flow (Max: {compensated_magnitude.max():.2f})')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 静态区域掩�?
    static_mask = result['static_mask']
    overlay = image.copy()
    overlay[static_mask] = [0, 255, 0]  # 绿色标记静态区�?
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f'Static Regions (Ratio: {result["global_dynamics"]["static_ratio"]:.3f})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def create_temporal_plot(results, temporal_stats):
    """创建时序分析�?"""
    dynamics_scores = [r['static_dynamics']['dynamics_score'] for r in results]
    static_ratios = [r['global_dynamics']['static_ratio'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 动态度分数
    ax1.plot(dynamics_scores, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=temporal_stats['mean_dynamics_score'], color='r', linestyle='--', 
               label=f'Mean: {temporal_stats["mean_dynamics_score"]:.3f}')
    ax1.set_ylabel('Dynamics Score')
    ax1.set_title('Static Object Dynamics Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 静态区域比�?
    ax2.plot(static_ratios, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.axhline(y=temporal_stats['mean_static_ratio'], color='r', linestyle='--',
               label=f'Mean: {temporal_stats["mean_static_ratio"]:.3f}')
    ax2.set_ylabel('Static Region Ratio')
    ax2.set_xlabel('Frame Index')
    ax2.set_title('Static Region Ratio Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_demo_report(results, temporal_stats):
    """生成演示报告"""
    report = f"""
相机转动拍摄静态建�? - 动态度分析演示报告
=====================================

演示场景描述:
- 场景类型: 静态建筑物
- 相机运动: 水平转动 (每帧3像素位移)
- 分析帧数: {len(results)}

分析结果:
--------
时序动态度统计:
- 平均动态度分数: {temporal_stats['mean_dynamics_score']:.3f}
- 动态度分数标准�?: {temporal_stats['std_dynamics_score']:.3f}
- 最大动态度分数: {temporal_stats['max_dynamics_score']:.3f}
- 最小动态度分数: {temporal_stats['min_dynamics_score']:.3f}

静态区域分�?:
- 平均静态区域比�?: {temporal_stats['mean_static_ratio']:.3f}
- 静态区域比例标准差: {temporal_stats['std_static_ratio']:.3f}

系统性能评估:
- 时序稳定�?: {temporal_stats['temporal_stability']:.3f}
- 平均一致性分�?: {temporal_stats['mean_consistency_score']:.3f}

结论:
----
"""
    
    # 添加分析结论
    mean_dynamics = temporal_stats['mean_dynamics_score']
    mean_static_ratio = temporal_stats['mean_static_ratio']
    temporal_stability = temporal_stats['temporal_stability']
    
    if mean_dynamics < 1.0:
        report += "�? 系统成功检测到静态物体，相机运动补偿效果良好\n"
    elif mean_dynamics < 2.0:
        report += "�? 检测到轻微的残余运动，可能需要进一步优化\n"
    else:
        report += "�? 动态度较高，建议检查相机运动估计准确性\n"
    
    if mean_static_ratio > 0.7:
        report += "�? 场景主要由静态物体组成，分析结果可靠\n"
    else:
        report += "�? 静态区域检测可能需要调整参数\n"
    
    if temporal_stability > 0.8:
        report += "�? 时序稳定性高，动态度计算结果一致\n"
    else:
        report += "�? 时序稳定性有待提高\n"
    
    report += f"\n本演示展示了系统如何有效区分相机运动和物体运动，\n"
    report += f"仅计算静态物体的真实动态度。\n"
    
    return report


def main():
    """主演示函�?"""
    print("=" * 60)
    print("静态物体动态度分析系统 - 演示")
    print("=" * 60)
    
    try:
        # 分析演示视频
        results, temporal_stats = analyze_demo_video()
        
        print("\n" + "=" * 60)
        print("演示完成�?")
        print("=" * 60)
        
        print(f"\n关键发现:")
        print(f"- 成功区分了相机运动和物体运动")
        print(f"- 平均静态物体动态度: {temporal_stats['mean_dynamics_score']:.3f}")
        print(f"- 系统稳定�?: {temporal_stats['temporal_stability']:.3f}")
        
        print(f"\n查看详细结果:")
        print(f"- 可视化图�?: demo_output/")
        print(f"- 分析报告: demo_output/demo_report.txt")
        
    except Exception as e:
        print(f"演示过程中发生错�?: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()