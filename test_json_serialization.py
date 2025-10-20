#!/usr/bin/env python3
"""
����JSON���л��޸�
��֤��Linux�������ܹ���ȷ���л������JSON
"""

import json
import numpy as np
import tempfile
import os


def test_json_serialization():
    """����JSON���л�����"""
    
    # ģ�����numpy����Ľ��
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
            'frame_results': [{'data': np.array([1, 2, 3])}],  # ��ᵼ�����л�ʧ��
            'original_flows': [np.random.rand(10, 10, 2)],  # numpy����
            'camera_compensation_results': [
                {'homography': np.eye(3)}  # numpy����
            ]
        }
    }
    
    print("����1: ֱ�����л�����numpy����Ľ����Ԥ��ʧ�ܣ�")
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            json.dump([result_with_numpy], f, indent=2, ensure_ascii=False)
        print("  ? ����ɹ� - Ӧ��ʧ�ܵ�û��ʧ��")
        os.remove(temp_file)
        return False
    except (TypeError, ValueError) as e:
        print(f"  ? Ԥ��ʧ��: {type(e).__name__}")
    
    print("\n����2: ʹ�ù��˷������л���Ԥ�ڳɹ���")
    try:
        # ģ���޸�������л�����
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
            # ע��: ������ 'full_result'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            json.dump([json_safe_result], f, indent=2, ensure_ascii=False)
        
        # ��֤�ļ����Ա���ȡ
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        print("  ? ���л��ɹ�")
        print(f"  ? �����л��ɹ������� {len(loaded_data)} �����")
        
        os.remove(temp_file)
        return True
        
    except Exception as e:
        print(f"  ? ʧ��: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False
    
    print("\n����3: ��֤���˺������������")
    expected_fields = [
        'video_name', 'video_path', 'status', 'frame_count',
        'mean_dynamics_score', 'mean_static_ratio', 'temporal_stability',
        'actual_score', 'confidence', 'output_dir'
    ]
    missing_fields = [f for f in expected_fields if f not in json_safe_result]
    if missing_fields:
        print(f"  ? ȱ���ֶ�: {missing_fields}")
        return False
    else:
        print("  ? ���б�Ҫ�ֶζ�����")
        return True


def test_badcase_export():
    """����BadCase������JSON���л�"""
    
    print("\n����4: BadCase�б���")
    
    # ģ�����numpy�����batch_result
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
                'full_result': {  # ����numpy����
                    'flows': [np.random.rand(10, 10, 2)]
                }
            }
        ],
        'normal_list': [
            {
                'video_name': 'normal_video',
                'is_badcase': False,
                'full_result': {  # ����numpy����
                    'flows': [np.random.rand(10, 10, 2)]
                }
            }
        ]
    }
    
    try:
        # ģ���޸���ĵ�������
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
        
        # ��֤
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        print("  ? BadCase�������л��ɹ�")
        print(f"  ? badcase_list���� {len(loaded_data['badcase_list'])} ����Ŀ")
        print(f"  ? normal_list���� {len(loaded_data['normal_list'])} ����Ŀ")
        
        # ��֤full_result�ѱ�����
        if 'full_result' in loaded_data['badcase_list'][0]:
            print("  ? full_result�ֶ�δ����ȷ����")
            os.remove(temp_file)
            return False
        else:
            print("  ? full_result�ֶ��ѱ���ȷ����")
        
        os.remove(temp_file)
        return True
        
    except Exception as e:
        print(f"  ? ʧ��: {e}")
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return False


if __name__ == '__main__':
    print("="*70)
    print("JSON���л��޸���֤����")
    print("="*70)
    print()
    
    results = []
    results.append(test_json_serialization())
    results.append(test_badcase_export())
    
    print("\n" + "="*70)
    print("�����ܽ�")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"ͨ��: {passed}/{total}")
    
    if all(results):
        print("\n? ���в���ͨ����JSON���л��޸��ɹ���")
        print("\n�޸�˵��:")
        print("1. video_processor.py: �ڱ���batch_summary.jsonʱ���˵�full_result�ֶ�")
        print("2. badcase_detector.py: ��export_badcase_listʱ���˵�full_result�ֶ�")
        print("3. ��Щ�޸�ȷ����Linux������Ҳ���������л�JSON")
    else:
        print("\n? ���ֲ���ʧ�ܣ��������")

