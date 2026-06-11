import torch
from model.backbone import resnet
import numpy as np

class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding,
                                     dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim  # (num_gridding, num_cls_per_lane, num_of_lanes)
        self.use_aux = use_aux
        self.total_dim = int(np.prod(cls_dim))

        # 骨干网络
        self.model = resnet(backbone, pretrained=pretrained)

        # 池化层：将特征图压缩为固定维度
        # ResNet-18/34 输出 512 通道，其他输出 2048 通道
        pool_in_ch = 512 if backbone in ['34', '18'] else 2048
        self.pool = torch.nn.Conv2d(pool_in_ch, 8, 1)
        # 288x800 下采样 32 倍 -> 9x25, 8 通道 -> 1800 维
        # 224x224 下采样 32 倍 -> 7x7,  8 通道 -> 392 维
        self.feat_dim = 8 * (self.w // 32) * (self.h // 32)

        # 自适应全连接分类头：中间层维度根据总输出维度调整
        hidden_dim = min(2048, max(256, self.total_dim // 2))
        if hidden_dim >= self.total_dim:
            # 输出维度较小，直接单层映射
            self.cls = torch.nn.Linear(self.feat_dim, self.total_dim)
        else:
            self.cls = torch.nn.Sequential(
                torch.nn.Linear(self.feat_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, self.total_dim),
            )

        if self.use_aux:
            small_backbone = backbone in ['34', '18']
            ch2 = 128 if small_backbone else 512
            ch3 = 256 if small_backbone else 1024
            ch4 = 512 if small_backbone else 2048

            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(ch2, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(ch3, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(ch4, 128, 3, padding=1),
                conv_bn_relu(128, 128, 3, padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3, padding=2, dilation=2),
                conv_bn_relu(256, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=2, dilation=2),
                conv_bn_relu(128, 128, 3, padding=4, dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1, 1),
            )
            initialize_weights(self.aux_header2, self.aux_header3,
                             self.aux_header4, self.aux_combine)

        initialize_weights(self.cls)

    def forward(self, x):
        # nchw -> backbone features
        x2, x3, fea = self.model(x)

        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3, scale_factor=2, mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4, scale_factor=4, mode='bilinear')
            aux_seg = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        fea = self.pool(fea).view(-1, self.feat_dim)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)

        if self.use_aux:
            return group_cls, aux_seg
        return group_cls


def initialize_weights(*models):
    for model in models:
        _init_weights(model)


def _init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            _init_weights(mini_m)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0.0, std=0.01)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Module):
        for mini_m in m.children():
            _init_weights(mini_m)