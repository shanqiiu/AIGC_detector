# -*- coding: utf-8 -*-
"""
�����㷨�ԱȽű�
�Ƚ� Farneback vs TV-L1 �ھ�̬���嶯̬�ȼ�������еı���
"""

import numpy as np
import cv2
import time
from pathlib import Path
from simple_raft import SimpleRAFTPredictor
from static_object_analyzer import StaticObjectDynamicsCalculator


def load_test_frames(data_dir='test_data', num_frames=5):
    """���ز���֡"""
    data_path = Path(data_dir)
    frames = []
    
    for i in range(num_frames):
        frame_path = data_path / f'frame_{i:04d}.png'
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        else:
            print(f"����: δ�ҵ� {frame_path}")
    
    return frames


def compare_optical_flow_methods(frames):
    """�ԱȲ�ͬ��������"""
    
    print("=" * 70)
    print("�����㷨�Աȣ�Farneback vs TV-L1")
    print("=" * 70)
    
    methods = {
        'farneback': '���٣�Farneback��',
        'tvl1': '�߾��ȣ�TV-L1��'
    }
    
    results = {}
    
    for method_key, method_name in methods.items():
        print(f"\n{'='*70}")
        print(f"���Է���: {method_name}")
        print(f"{'='*70}")
        
        # ����Ԥ����
        predictor = SimpleRAFTPredictor(method=method_key)
        calculator = StaticObjectDynamicsCalculator()
        
        # �������
        start_time = time.time()
        flows = []
        for i in range(len(frames) - 1):
            flow = predictor.predict_flow(frames[i], frames[i + 1])
            flows.append(flow.transpose(1, 2, 0))  # ת��Ϊ (H, W, 2)
        
        flow_time = time.time() - start_time
        
        # ���㾲̬���嶯̬��
        start_time = time.time()
        dynamics_results = []
        for i, flow in enumerate(flows):
            result = calculator.calculate_frame_dynamics(
                flow, frames[i], frames[i + 1]
            )
            dynamics_results.append(result)
        
        analysis_time = time.time() - start_time
        total_time = flow_time + analysis_time
        
        # ͳ�ƽ��
        static_dynamics_scores = [r['static_dynamics']['dynamics_score'] 
                                 for r in dynamics_results]
        avg_dynamics = np.mean(static_dynamics_scores)
        max_dynamics = np.max(static_dynamics_scores)
        
        # ������
        results[method_key] = {
            'name': method_name,
            'flow_time': flow_time,
            'analysis_time': analysis_time,
            'total_time': total_time,
            'avg_dynamics': avg_dynamics,
            'max_dynamics': max_dynamics,
            'dynamics_scores': static_dynamics_scores
        }
        
        # ������
        print(f"\n����ָ��:")
        print(f"  ��������ʱ��: {flow_time:.3f}�� ({flow_time/len(flows)*1000:.1f}ms/֡)")
        print(f"  ��̬�ȷ���ʱ��: {analysis_time:.3f}��")
        print(f"  ��ʱ��: {total_time:.3f}��")
        
        print(f"\n�����:")
        print(f"  ƽ����̬��: {avg_dynamics:.4f}")
        print(f"  ���̬��: {max_dynamics:.4f}")
        print(f"  ��֡��̬��: {[f'{s:.4f}' for s in static_dynamics_scores]}")
    
    # �Աȷ���
    print(f"\n{'='*70}")
    print("�Աȷ���")
    print(f"{'='*70}")
    
    farneback = results['farneback']
    tvl1 = results['tvl1']
    
    print(f"\n�ٶȶԱ�:")
    print(f"  Farneback: {farneback['total_time']:.3f}��")
    print(f"  TV-L1:     {tvl1['total_time']:.3f}��")
    print(f"  �ٶȱ�:    TV-L1 �� {tvl1['total_time']/farneback['total_time']:.2f}��")
    
    print(f"\n���ȶԱ� (ƽ����̬��):")
    print(f"  Farneback: {farneback['avg_dynamics']:.4f}")
    print(f"  TV-L1:     {tvl1['avg_dynamics']:.4f}")
    print(f"  ����:      {abs(tvl1['avg_dynamics'] - farneback['avg_dynamics']):.4f}")
    
    print(f"\n���ȶԱ� (���̬��):")
    print(f"  Farneback: {farneback['max_dynamics']:.4f}")
    print(f"  TV-L1:     {tvl1['max_dynamics']:.4f}")
    print(f"  ����:      {abs(tvl1['max_dynamics'] - farneback['max_dynamics']):.4f}")
    
    # �жϲ���������
    print(f"\n{'='*70}")
    print("����")
    print(f"{'='*70}")
    
    diff_ratio = abs(tvl1['avg_dynamics'] - farneback['avg_dynamics']) / farneback['avg_dynamics']
    
    if diff_ratio < 0.1:
        print("? ���ַ����������� (<10% ����)")
        print("  ���飺ʹ��Farneback�����죩")
    elif diff_ratio < 0.3:
        print("? ���ַ�����һ������ (10-30% ����)")
        print("  ���飺�����ٶ�/��������ѡ��")
    else:
        print("! ���ַ�������ϴ� (>30% ����)")
        print("  ���飺ʹ��TV-L1����׼ȷ��")
    
    print(f"\n�ٶ�/����Ȩ��:")
    print(f"  Farneback: ���٣��ʺ�ʵʱ����Ϳ���ԭ��")
    print(f"  TV-L1:     ��ȷ���ʺ����߷����͸��������")
    
    return results


def main():
    """������"""
    print("\n�����㷨�Աȹ���")
    print("����������ͬ�����㷨�ھ�̬���嶯̬�ȼ�������еı���\n")
    
    # ���ز���֡
    print("���ز�������...")
    frames = load_test_frames('test_data', num_frames=5)
    
    if len(frames) < 2:
        frames = load_test_frames('demo_data', num_frames=10)
    
    if len(frames) < 2:
        print("����: ��Ҫ����2֡ͼ��")
        print("��ȷ�� test_data/ �� demo_data/ Ŀ¼���в���ͼ��")
        return
    
    print(f"? ������ {len(frames)} ֡ͼ��")
    print(f"  �ֱ���: {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    # ���жԱ�
    results = compare_optical_flow_methods(frames)
    
    print(f"\n{'='*70}")
    print("�������!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

