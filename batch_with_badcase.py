# -*- coding: utf-8 -*-
"""
批量视频BadCase检测 - 兼容性wrapper

本文件已整合到 video_processor.py 中。
为保持向后兼容，此文件作为wrapper重定向到统一入口。

推荐直接使用:
    python video_processor.py --batch --badcase-labels labels.json
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入统一入口
from video_processor import main as video_processor_main
from video_processor import load_expected_labels, VideoProcessor, BadCaseDetector, BadCaseAnalyzer
import argparse


def main():
    """兼容性main函数 - 转换参数并调用video_processor.py"""
    
    parser = argparse.ArgumentParser(
        description='批量视频处理与BadCase检测（兼容wrapper）',
        epilog='注意：本脚本已整合到 video_processor.py，建议使用: python video_processor.py --batch -l labels.json'
    )
    parser.add_argument('--input', '-i', required=True, help='输入视频目录')
    parser.add_argument('--output', '-o', default='badcase_output', help='输出目录')
    parser.add_argument('--labels', '-l', required=True, help='期望标签文件（JSON）')
    parser.add_argument('--raft_model', '-m', default="pretrained_models/raft-things.pth", help='RAFT模型路径')
    parser.add_argument('--device', default='cuda', help='计算设备 (cuda/cpu)')
    parser.add_argument('--fov', type=float, default=60.0, help='相机视场角（度）')
    parser.add_argument('--mismatch-threshold', type=float, default=0.3, help='不匹配阈值（默认0.3）')
    parser.add_argument('--no-camera-compensation', dest='camera_compensation', action='store_false', help='禁用相机补偿')
    parser.add_argument('--camera-ransac-thresh', type=float, default=1.0, help='相机补偿RANSAC阈值（像素）')
    parser.add_argument('--camera-max-features', type=int, default=2000, help='相机补偿最大特征点数')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果（会增加处理时间）')
    parser.add_argument('--normalize-by-resolution', dest='use_normalized_flow', action='store_true', 
                       help='按分辨率归一化光流（推荐开启以保证不同分辨率视频的公平性）')
    parser.add_argument('--flow-threshold-ratio', type=float, default=0.002, 
                       help='归一化后的静态阈值（相对于图像对角线，默认0.002）')
    parser.add_argument('--filter-badcase-only', action='store_true', help='只保留BadCase视频结果')
    parser.set_defaults(camera_compensation=True, visualize=False, use_normalized_flow=False)
    
    args = parser.parse_args()
    
    # 转换参数并调用统一入口
    print("=" * 70)
    print("兼容模式: 正在调用统一的 video_processor.py")
    print("建议: 下次直接使用 python video_processor.py --batch -l labels.json")
    print("=" * 70 + "\n")
    
    # 准备统一参数
    sys.argv = [
        'video_processor.py',
        '--input', args.input,
        '--output', args.output,
        '--batch',  # 批量模式
        '--badcase-labels', args.labels,  # 转换参数名
        '--raft_model', args.raft_model,
        '--device', args.device,
        '--fov', str(args.fov),
        '--mismatch-threshold', str(args.mismatch_threshold),
        '--camera-ransac-thresh', str(args.camera_ransac_thresh),
        '--camera-max-features', str(args.camera_max_features),
        '--flow-threshold-ratio', str(args.flow_threshold_ratio)
    ]
    
    if not args.camera_compensation:
        sys.argv.append('--no-camera-compensation')
    if args.visualize:
        sys.argv.append('--visualize')
    if args.use_normalized_flow:
        sys.argv.append('--normalize-by-resolution')
    
    # 调用统一入口
    video_processor_main()


if __name__ == '__main__':
    main()
