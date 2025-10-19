# -*- coding: utf-8 -*-
"""
��������������ܼ���
"""

import os
import sys
import numpy as np
from video_processor import VideoProcessor

def test_camera_compensation_enabled():
    """���������������"""
    print("\n" + "="*70)
    print("����1: �����������")
    print("="*70)
    
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cpu',
        max_frames=5,
        enable_visualization=False,
        enable_camera_compensation=True
    )
    
    # �������������Ƿ��ѳ�ʼ��
    assert processor.camera_compensator is not None, "���������Ӧ�ñ���ʼ��"
    assert processor.enable_camera_compensation == True, "�������Ӧ������"
    
    print("? �����������ʼ���ɹ�")
    print(f"  - ����״̬: {processor.enable_camera_compensation}")
    print(f"  - ������ʵ��: {type(processor.camera_compensator).__name__}")
    
    return True

def test_camera_compensation_disabled():
    """���Խ����������"""
    print("\n" + "="*70)
    print("����2: �����������")
    print("="*70)
    
    processor = VideoProcessor(
        raft_model_path="pretrained_models/raft-things.pth",
        device='cpu',
        max_frames=5,
        enable_visualization=False,
        enable_camera_compensation=False
    )
    
    # �������������Ƿ�δ��ʼ��
    assert processor.camera_compensator is None, "�����������Ӧ�ñ���ʼ��"
    assert processor.enable_camera_compensation == False, "�������Ӧ�ý���"
    
    print("? ���������ȷ����")
    print(f"  - ����״̬: {processor.enable_camera_compensation}")
    print(f"  - ������ʵ��: {processor.camera_compensator}")
    
    return True

def test_camera_compensation_with_images():
    """����ʹ��ͼ�����н����������"""
    print("\n" + "="*70)
    print("����3: ʹ��demo���ݲ����������")
    print("="*70)
    
    # ���demo�����Ƿ����
    demo_dir = "demo_data"
    if not os.path.exists(demo_dir):
        print("? ������demo_dataĿ¼������")
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
    
    # ����ͼ��
    frames = processor.extract_frames_from_images(demo_dir)
    
    if len(frames) < 2:
        print("? ������ͼ����������")
        return True
    
    print(f"  - ����֡��: {len(frames)}")
    
    # ������Ƶ
    output_dir = "test_output_camera_comp"
    result = processor.process_video(frames[:5], output_dir=output_dir)
    
    # ��֤���
    assert 'camera_compensation_enabled' in result, "���Ӧ�������������־"
    assert result['camera_compensation_enabled'] == True, "�������Ӧ������"
    assert 'camera_compensation_results' in result, "���Ӧ��������������"
    
    # ����������ͳ��
    comp_results = result['camera_compensation_results']
    print(f"  - �����������: {len(comp_results)}")
    
    # ͳ�Ƴɹ��Ĳ���
    successful = sum(1 for r in comp_results if r is not None and r['homography'] is not None)
    print(f"  - �ɹ�����֡��: {successful}/{len(comp_results)}")
    
    if successful > 0:
        # ��ʾ��һ���ɹ���������ϸ��Ϣ
        for i, r in enumerate(comp_results):
            if r is not None and r['homography'] is not None:
                print(f"\n  ��{i}֡��������:")
                print(f"    - �ڵ���: {r['inliers']}")
                print(f"    - ��ƥ����: {r['total_matches']}")
                print(f"    - ƥ����: {r['inliers']/max(r['total_matches'], 1):.1%}")
                
                # ��������״
                print(f"    - ���������״: {r['camera_flow'].shape}")
                print(f"    - �в������״: {r['residual_flow'].shape}")
                
                # �ȽϷ���
                camera_mag = np.sqrt(r['camera_flow'][:,:,0]**2 + r['camera_flow'][:,:,1]**2)
                residual_mag = np.sqrt(r['residual_flow'][:,:,0]**2 + r['residual_flow'][:,:,1]**2)
                print(f"    - �������ƽ������: {camera_mag.mean():.2f}")
                print(f"    - �в����ƽ������: {residual_mag.mean():.2f}")
                break
    
    print("\n? �������������������")
    
    # ����������
    # import shutil
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    
    return True

def test_custom_compensation_params():
    """�����Զ��������������"""
    print("\n" + "="*70)
    print("����4: �Զ��������������")
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
    
    # ��֤�����Ƿ���ȷ����
    assert processor.camera_compensator is not None, "���������Ӧ�ñ���ʼ��"
    assert processor.camera_compensator.ransac_thresh == 0.8, "RANSAC��ֵӦΪ0.8"
    assert processor.camera_compensator.max_features == 3000, "���������ӦΪ3000"
    
    print("? �Զ���������óɹ�")
    print(f"  - RANSAC��ֵ: {processor.camera_compensator.ransac_thresh}")
    print(f"  - ���������: {processor.camera_compensator.max_features}")
    
    return True

def main():
    """�������в���"""
    print("\n" + "="*70)
    print("����������ܼ��ɲ���")
    print("="*70)
    
    tests = [
        ("�����������", test_camera_compensation_enabled),
        ("�����������", test_camera_compensation_disabled),
        ("�Զ��岹������", test_custom_compensation_params),
        ("ͼ�����в���", test_camera_compensation_with_images),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"? {test_name} ʧ��")
        except Exception as e:
            failed += 1
            print(f"? {test_name} �쳣: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("�����ܽ�")
    print("="*70)
    print(f"ͨ��: {passed}/{len(tests)}")
    print(f"ʧ��: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n? ���в���ͨ��!")
        return 0
    else:
        print("\n? ���ֲ���ʧ��")
        return 1

if __name__ == '__main__':
    sys.exit(main())

