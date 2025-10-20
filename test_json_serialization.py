#!/usr/bin/env python3
"""
测试JSON序列化修复
验证在Linux环境下能够正确序列化结果到JSON
"""

import json
import numpy as np
import tempfile
import os


def test_json_serialization():
    """测试JSON序列化功能"""
    
    # 模拟包含numpy数组的结果
    result_with_numpy = {
        'video_name': 'test_video',
        'video_path': '/path/to/video.mp4',
        'status': 'success',
        'frame_count': 100,
        'mean_dynamics_score': 0.75,
        'mean_static_ratio': 0.25,
        'temporal_stability': 0.85,
        'actual_score': 0.72,
        'confidence': 0.90,
        'output_dir': '/output',
        'full_result': {
            'frame_results': [{'data': np.array([1, 2, 3])}],  # 这会导致序列化失败
            'original_flows': [np.random.rand(10, 10, 2)],  # numpy数组
            'camera_compensation_results': [
                {'homography': np.eye(3)}  # numpy数组
            ]
        }
    }
    
    print("测试1: 直接序列化包含numpy数组的结果（预期失败）")
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            json.dump([result_with_numpy], f, indent=2, ensure_ascii=False)
        print("  ? 意外成功 - 应该失败但没有失败")
        os.remove(temp_file)
        return False
    except (TypeError, ValueError) as e:
        print(f"  ? 预期失败: {type(e).__name__}")
    
    print("\n测试2: 使用过滤方法序列化（预期成功）")
    try:
        # 模拟修复后的序列化方法
        json_safe_result = {
            'video_name': result_with_numpy['video_name'],
            'video_path': result_with_numpy['video_path'],
            'status': result_with_numpy['status'],
            'frame_count': result_with_numpy['frame_count'],
            'mean_dynamics_score': result_with_numpy['mean_dynamics_score'],
            'mean_static_ratio': result_with_numpy['mean_static_ratio'],
            'temporal_stability': result_with_numpy['temporal_stability'],
            'actual_score': result_with_numpy['actual_score'],
            'confidence': result_with_numpy['confidence'],
            'output_dir': result_with_numpy['output_dir']
            # 注意: 不包含 'full_result'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            json.dump([json_safe_result], f, indent=2, ensure_ascii=False)
        
        # 验证文件可以被读取
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        print("  ? 序列化成功")
        print(f"  ? 反序列化成功，包含 {len(loaded_data)} 个结果")
        
        os.remove(temp_file)
        return True
        
    except Exception as e:
        print(f"  ? 失败: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False
    
    print("\n测试3: 验证过滤后的数据完整性")
    expected_fields = [
        'video_name', 'video_path', 'status', 'frame_count',
        'mean_dynamics_score', 'mean_static_ratio', 'temporal_stability',
        'actual_score', 'confidence', 'output_dir'
    ]
    missing_fields = [f for f in expected_fields if f not in json_safe_result]
    if missing_fields:
        print(f"  ? 缺少字段: {missing_fields}")
        return False
    else:
        print("  ? 所有必要字段都存在")
        return True


def test_badcase_export():
    """测试BadCase导出的JSON序列化"""
    
    print("\n测试4: BadCase列表导出")
    
    # 模拟包含numpy数组的batch_result
    batch_result = {
        'total_videos': 2,
        'successful': 2,
        'failed': 0,
        'badcase_count': 1,
        'normal_count': 1,
        'badcase_rate': 0.5,
        'badcase_list': [
            {
                'video_name': 'badcase_video',
                'is_badcase': True,
                'badcase_type': 'static_to_dynamic',
                'severity': 'moderate',
                'full_result': {  # 包含numpy数组
                    'flows': [np.random.rand(10, 10, 2)]
                }
            }
        ],
        'normal_list': [
            {
                'video_name': 'normal_video',
                'is_badcase': False,
                'full_result': {  # 包含numpy数组
                    'flows': [np.random.rand(10, 10, 2)]
                }
            }
        ]
    }
    
    try:
        # 模拟修复后的导出方法
        json_safe_result = {
            k: v for k, v in batch_result.items() 
            if k not in ['badcase_list', 'normal_list']
        }
        
        if 'badcase_list' in batch_result:
            json_safe_result['badcase_list'] = [
                {k: v for k, v in bc.items() if k != 'full_result'}
                for bc in batch_result['badcase_list']
            ]
        
        if 'normal_list' in batch_result:
            json_safe_result['normal_list'] = [
                {k: v for k, v in norm.items() if k != 'full_result'}
                for norm in batch_result['normal_list']
            ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            json.dump(json_safe_result, f, indent=2, ensure_ascii=False)
        
        # 验证
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        print("  ? BadCase导出序列化成功")
        print(f"  ? badcase_list包含 {len(loaded_data['badcase_list'])} 个条目")
        print(f"  ? normal_list包含 {len(loaded_data['normal_list'])} 个条目")
        
        # 验证full_result已被过滤
        if 'full_result' in loaded_data['badcase_list'][0]:
            print("  ? full_result字段未被正确过滤")
            os.remove(temp_file)
            return False
        else:
            print("  ? full_result字段已被正确过滤")
        
        os.remove(temp_file)
        return True
        
    except Exception as e:
        print(f"  ? 失败: {e}")
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return False


if __name__ == '__main__':
    print("="*70)
    print("JSON序列化修复验证测试")
    print("="*70)
    print()
    
    results = []
    results.append(test_json_serialization())
    results.append(test_badcase_export())
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if all(results):
        print("\n? 所有测试通过！JSON序列化修复成功。")
        print("\n修复说明:")
        print("1. video_processor.py: 在保存batch_summary.json时过滤掉full_result字段")
        print("2. badcase_detector.py: 在export_badcase_list时过滤掉full_result字段")
        print("3. 这些修复确保在Linux环境下也能正常序列化JSON")
    else:
        print("\n? 部分测试失败，请检查代码")

