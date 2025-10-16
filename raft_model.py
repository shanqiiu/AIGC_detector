"""
RAFT (Recurrent All-Pairs Field Transforms) 光流估计模型
用于计算视频帧间的光流
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import cv2


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        num_groups = planes // 8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            
        if stride == 1:
            self.downsample = None
        else:
            if norm_fn == 'group':
                norm_layer = nn.GroupNorm(num_groups=planes//8, num_channels=planes)
            elif norm_fn == 'batch':
                norm_layer = nn.BatchNorm2d(planes)
            elif norm_fn == 'instance':
                norm_layer = nn.InstanceNorm2d(planes)
            else:
                norm_layer = nn.Sequential()
            
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), 
                norm_layer)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class FeatureEncoder(nn.Module):
    """特征编码器"""
    def __init__(self, output_dim=256, norm_fn='batch', dropout=0.0):
        super(FeatureEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # 输出卷积层
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # 如果输入是灰度图，转换为3通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class CorrBlock:
    """相关性计算块"""
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # 所有对相关性
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """双线性采样"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    
    # 确保batch size匹配
    if img.dim() == 4 and grid.dim() == 4:
        if img.shape[0] != grid.shape[0]:
            # 如果batch size不匹配，调整grid的batch size
            grid = grid.view(img.shape[0], -1, grid.shape[-2], grid.shape[-1])
    
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


class UpdateBlock(nn.Module):
    """更新块"""
    def __init__(self, hidden_dim=128, input_dim=128):
        super(UpdateBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.gru = nn.Sequential(
            nn.Conv2d(hidden_dim+hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),
        )

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(torch.cat([corr, flow], dim=1))
        inp = self.encoder(inp)

        net = self.gru(torch.cat([net, inp, motion_features], dim=1))
        delta_flow = self.flow_head(net)

        return net, delta_flow


class RAFT(nn.Module):
    """RAFT光流估计网络"""
    def __init__(self, small=False, dropout=0.0):
        super(RAFT, self).__init__()

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        if small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64

        # 特征网络，上下文网络
        self.fnet = FeatureEncoder(output_dim=256, norm_fn='instance', dropout=dropout)        
        self.cnet = FeatureEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=dropout)
        self.update_block = UpdateBlock(hidden_dim=hdim, input_dim=cdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """初始化光流"""
        N, C, H, W = img.shape
        coordinates = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        coords0 = torch.stack(coordinates, dim=0).float()
        coords1 = torch.stack(coordinates, dim=0).float()
        coords0 = coords0[None].repeat(N, 1, 1, 1)
        coords1 = coords1[None].repeat(N, 1, 1, 1)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """上采样光流"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """前向传播"""
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # 运行特征网络
        fmap1 = self.fnet(image1)        
        fmap2 = self.fnet(image2)
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=4)

        # 运行上下文网络
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        coords0 = coords0.to(image1.device)
        coords1 = coords1.to(image1.device)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # 索引相关性体积

            flow = coords1 - coords0
            net, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(F)
            coords1 = coords1 + delta_flow

            # 上采样预测
            if upsample and (test_mode or itr == iters-1):
                # 简化版本，不使用mask上采样
                flow_up = 8 * F.interpolate(coords1 - coords0, scale_factor=8, mode='bilinear', align_corners=True)
                flow_predictions.append(flow_up)
            else:
                flow_predictions.append(coords1 - coords0)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions


def load_image(imfile):
    """加载图像"""
    img = cv2.imread(imfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_image(img, device='cuda'):
    """预处理图像"""
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


class RAFTPredictor:
    """RAFT预测器"""
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.model = RAFT(small=False)
        
        if model_path and torch.cuda.is_available():
            # 如果有预训练模型，加载它
            try:
                self.model.load_state_dict(torch.load(model_path))
            except:
                print("无法加载预训练模型，使用随机初始化")
        
        self.model.to(device)
        self.model.eval()

    def predict_flow(self, image1, image2):
        """预测光流"""
        with torch.no_grad():
            # 预处理图像
            if isinstance(image1, np.ndarray):
                image1 = preprocess_image(image1, self.device)
            if isinstance(image2, np.ndarray):
                image2 = preprocess_image(image2, self.device)
            
            # 预测光流
            _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            
            return flow_up.cpu().numpy()[0]

    def predict_flow_sequence(self, images):
        """预测图像序列的光流"""
        flows = []
        for i in range(len(images) - 1):
            flow = self.predict_flow(images[i], images[i + 1])
            flows.append(flow)
        return flows