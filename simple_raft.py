"""
�򻯰�RAFTģ�ͣ�������ʾ��̬���嶯̬�ȼ���
֧�ֶ��ֹ����㷨��Farneback, TV-L1��
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple


class SimpleRAFT:
    """�򻯰�RAFT�������� - ֧�ֶ��ֹ����㷨"""
    
    def __init__(self, device='cpu', method='farneback'):
        """
        ��ʼ������������
        
        Args:
            device: �����豸 ('cpu' �� 'cuda')
            method: ��������
                - 'farneback': ���٣������е� (Ĭ��)
                - 'tvl1': ���������ȸߣ��߽�����
        """
        self.device = device
        self.method = method
        
        # ���ʹ��TV-L1��������������
        if method == 'tvl1':
            try:
                self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
                    tau=0.25,          # ʱ�䲽��
                    lambda_=0.15,      # ������Ȩ��
                    theta=0.3,         # ƽ����Ȩ��  
                    nscales=5,         # ����������
                    warps=5,           # Warp����
                    epsilon=0.01,      # ֹͣ��ֵ
                    innerIterations=30,
                    outerIterations=10,
                    scaleStep=0.8,
                    gamma=0.0,
                    useInitialFlow=False
                )
                print(f"? ʹ��TV-L1�����㷨���߾��ȣ�")
            except AttributeError:
                print("? opencv-contrib-pythonδ��װ��TV-L1������")
                print("  �Զ����˵�Farneback�㷨")
                print("  ��װ���pip install opencv-contrib-python")
                self.method = 'farneback'
        
        if method == 'farneback':
            print(f"? ʹ��Farneback�����㷨�����٣�")
    
    def estimate_flow_opencv(self, image1, image2):
        """ʹ��OpenCV�Ĺ���������ΪRAFT�����"""
        # ת��Ϊ�Ҷ�ͼ
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = image1
            
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = image2
        
        # ����ѡ��ķ����������
        if self.method == 'tvl1':
            # TV-L1��������ַ������߽籣�ֺã����ȸߣ�
            flow = self.tvl1.calc(gray1, gray2, None)
            
        else:  # 'farneback' ��Ĭ��
            # Farneback�����㷨���ٶȿ죬�����еȣ�
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 
                pyr_scale=0.5,     # ����������
                levels=5,          # �����������������Դ����λ�ƣ�
                winsize=15,        # ���ڴ�С
                iterations=3,      # ÿ���������
                poly_n=7,          # ����ʽ��չ��������ƽ���ԣ�
                poly_sigma=1.5,    # ��˹��׼��
                flags=0
            )
        
        return flow
    
    def predict_flow(self, image1, image2):
        """Ԥ�����"""
        # ʹ��OpenCV���ƹ���
        flow = self.estimate_flow_opencv(image1, image2)
        
        # ת��ΪPyTorch������ʽ (2, H, W)
        if isinstance(flow, np.ndarray):
            flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
        else:
            # ���OpenCV����None�����������
            h, w = image1.shape[:2]
            flow_tensor = torch.zeros(2, h, w, dtype=torch.float32)
        
        return flow_tensor.numpy()


class SimpleRAFTPredictor:
    """�򻯰�RAFTԤ����"""
    
    def __init__(self, model_path=None, device='cpu', method='farneback'):
        """
        ��ʼ��Ԥ����
        
        Args:
            model_path: ģ��·�������ݲ�������ʵ���в�ʹ�ã�
            device: �����豸
            method: ��������
                - 'farneback': ���٣������е� (Ĭ��)
                - 'tvl1': ���������ȸߣ��߽�����
        
        ʾ��:
            # ʹ��Farneback�����٣�
            predictor = SimpleRAFTPredictor(method='farneback')
            
            # ʹ��TV-L1���߾��ȣ�
            predictor = SimpleRAFTPredictor(method='tvl1')
        """
        self.device = device
        self.method = method
        self.model = SimpleRAFT(device, method)
    
    def predict_flow(self, image1, image2):
        """Ԥ�����"""
        return self.model.predict_flow(image1, image2)
    
    def predict_flow_sequence(self, images):
        """Ԥ��ͼ�����еĹ���"""
        flows = []
        for i in range(len(images) - 1):
            flow = self.predict_flow(images[i], images[i + 1])
            flows.append(flow)
        return flows


if __name__ == '__main__':
    print("=" * 60)
    print("�򻯰�RAFT����Ԥ���� - ʹ��ʾ��")
    print("=" * 60)
    
    # ����1: ʹ��Farneback��Ĭ�ϣ�
    print("\n����1: Farneback���������٣�")
    predictor_fast = SimpleRAFTPredictor(method='farneback')
    
    # ����2: ʹ��TV-L1���߾��ȣ�
    print("\n����2: TV-L1�������߾��ȣ�")
    predictor_accurate = SimpleRAFTPredictor(method='tvl1')
    
    print("\n" + "=" * 60)
    print("ѡ���飺")
    print("- ����ԭ��/��ʾ �� method='farneback'")
    print("- ��������/�߾��� �� method='tvl1'")
    print("=" * 60)
