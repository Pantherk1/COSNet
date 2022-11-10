# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    # t_min=t_min.float()
    # t_max=t_max.float()
    
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def create_mapping_kernel(kernel_size=7):
    # [kernel_size * kernel_size, kernel_size, kernel_size]
    kernel_arr = np.zeros((kernel_size * kernel_size, kernel_size, kernel_size), np.float32)
    for h in range(kernel_arr.shape[1]):
        for w in range(kernel_arr.shape[2]):
            kernel_arr[h * kernel_arr.shape[2] + w, h, w] = 1.0
    
    # [kernel_size * kernel_size, 1, kernel_size, kernel_size]
    kernel_tensor = torch.from_numpy(np.expand_dims(kernel_arr, axis=1))
    kernel_params = nn.Parameter(data=kernel_tensor.contiguous(), requires_grad=False)
    print(kernel_params.type())
    
    return kernel_params


def create_conv_kernel(in_channels, out_channels, kernel_size=3, avg=0.0, std=0.1):
    # [out_channels, in_channels, kernel_size, kernel_size]
    kernel_arr = np.random.normal(loc=avg, scale=std, size=(out_channels, in_channels, kernel_size, kernel_size))
    kernel_arr = kernel_arr.astype(np.float32)
    kernel_tensor = torch.from_numpy(kernel_arr)
    kernel_params = nn.Parameter(data=kernel_tensor.contiguous(), requires_grad=True)
    print(kernel_params.type())
    return kernel_params


def create_conv_bias(channels):
    # [channels, ]
    bias_arr = np.zeros(channels, np.float32)
    assert bias_arr.shape[0] % 2 == 1
    
    bias_arr[bias_arr.shape[0] // 2] = 1.0
    bias_tensor = torch.from_numpy(bias_arr)
    bias_params = nn.Parameter(data=bias_tensor.contiguous(), requires_grad=True)
    
    return bias_params


class base(nn.Module):
    def __init__(self, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1):
        """
        :param channels: the basic channels of feature maps.
        :param pn_size: the size of propagation neighbors.
        :param kernel_size: the size of kernel.
        :param avg: the mean of normal initialization.
        :param std: the standard deviation of normal initialization.
        """
        super(base, self).__init__()
        self.kernel_size = kernel_size
        
        self.conv1_kernel = create_conv_kernel(in_channels=3, out_channels=channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)  # ##
        # self.conv2_kernel = create_conv_kernel(in_channels=channels, out_channels=channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv3_kernel = create_conv_kernel(in_channels=channels, out_channels=channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv4_kernel = create_conv_kernel(in_channels=channels, out_channels=2*channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv5_kernel = create_conv_kernel(in_channels=2*channels, out_channels=2*channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv6_kernel = create_conv_kernel(in_channels=2*channels, out_channels=2*channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_kernel = create_conv_kernel(in_channels=2*channels, out_channels=pn_size*pn_size,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_bias = create_conv_bias(pn_size*pn_size)
        self.bn1 = nn.BatchNorm2d(channels)
        # self.bn2 = nn.BatchNorm2d(channels)
        # self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(2*channels)
        # self.bn5 = nn.BatchNorm2d(2*channels)
        # self.bn6 = nn.BatchNorm2d(2*channels)
        self.bn7 = nn.BatchNorm2d(pn_size*pn_size)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_src, input_thick, input_thin): #8,1,304,304
        # a, b,  c = input_src, input_thick, input_thin
        if isinstance(input_thick, tuple):
            input_thick = input_thick[0]
        if isinstance(input_thin, tuple):
            input_thin = input_thin[0]
        # print(b[0].shape)  # torch.size([8, 1, 304, 304]) = a.shape
        input_all = torch.cat((input_src, input_thick, input_thin), dim=1)  # [b, 3, h, w] 8,3,304,304
        print(input_all.shape)
        assert input_all.size()[1] == 3  # ##
        print(input_all.shape)

        input_all = torch.squeeze(input_all)
        print(input_all.shape)
        fm_1 = F.conv2d(input_all, self.conv1_kernel, padding=self.kernel_size//2)
        fm_1 = self.bn1(fm_1)  # 2, 64, 304, 304
        fm_1 = self.relu(fm_1)
        #fm_2 = F.conv2d(fm_1, self.conv2_kernel, padding=self.kernel_size//2)
        #fm_2 = self.bn2(fm_2)
        #fm_2 = self.relu(fm_2)
        #fm_3 = F.conv2d(fm_2, self.conv3_kernel, padding=self.kernel_size//2)
        #fm_3 = self.bn3(fm_3)
        #fm_3 = self.relu(fm_3)
        fm_4 = F.conv2d(fm_1, self.conv4_kernel, padding=self.kernel_size//2)
        fm_4 = self.bn4(fm_4)  # 2, 128, 304, 304
        fm_4 = self.relu(fm_4)  # with MEM 2, 512, 304, 304
        #fm_5 = F.conv2d(fm_4, self.conv5_kernel, padding=self.kernel_size//2)
        #fm_5 = self.bn5(fm_5)
        #fm_5 = self.relu(fm_5)
        #fm_6 = F.conv2d(fm_5, self.conv6_kernel, padding=self.kernel_size//2)
        #fm_6 = self.bn6(fm_6)
        #fm_6 = self.relu(fm_6)
        fm_7 = F.conv2d(fm_4, self.conv7_kernel, self.conv7_bias, padding=self.kernel_size//2)
        fm_7 = self.bn7(fm_7)  # 2, 9, 304, 304
        fm_7 = F.relu(fm_7)  # with MEM 2, 25, 304, 304

        # x = F.softmax(fm_7, dim=1)  # 2, 9, 304, 304

        return F.softmax(fm_7, dim=1)  # 2, 9, 304, 304 [b, pn_size * pn_size, h, w]


class adaptive_aggregation(nn.Module):
    def __init__(self, pn_size=5):
        """
        :param pn_size: the size of propagation neighbors.
        """
        super(adaptive_aggregation, self).__init__()
        self.kernel_size = pn_size
        self.weight = create_mapping_kernel(kernel_size=self.kernel_size)
    
    def forward(self, input_thick,input_thin, agg_coeff):
        assert input_thick.size()[1] == 1 and input_thin.size()[1] == 1
        input_sal = torch.max(input_thick, input_thin)
        map_sal = F.conv2d(input_sal, self.weight, padding=self.kernel_size//2)
        # map_sal_inv = 1.0 - map_sal
        assert agg_coeff.size() == map_sal.size()
        
        prod_sal = torch.sum(map_sal * agg_coeff, dim=1).unsqueeze(1)
        # prod_sal = F.sigmoid(prod_sal)
        # prod_sal_inv = torch.sum(map_sal_inv * agg_coeff, dim=1).unsqueeze(1)
        
        return prod_sal  # [b, 1, h, w]


class fusion(nn.Module):
    def __init__(self, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1):
        super(fusion, self).__init__()
        self.backbone = base(channels, pn_size, kernel_size, avg, std)
        self.adagg = adaptive_aggregation(pn_size)
    
    def forward(self, input_src, input_thick, input_thin):  # ##
        agg_coeff = self.backbone(input_src, input_thick, input_thin)  # ##
        prod_sal = self.adagg(input_thick, input_thin, agg_coeff)
        
        return prod_sal

# ----------------------------------------------------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Sequential(
                nn.Conv2d(1, dim_in, 3, 1, 1, bias=False),
                nn.BatchNorm2d(dim_in),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(1, dim_in, 3, 1, 1, bias=False),
                nn.BatchNorm2d(dim_in),
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class FusionContrast(nn.Module):
    def __init__(self, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1, proj_dim=256, num_classes=2):
        super(FusionContrast, self).__init__()
        self.backbone = base(channels, pn_size, kernel_size, avg, std)
        self.adagg = adaptive_aggregation(pn_size)
        self.proj_dim = proj_dim  # 256
        self.num_classes = num_classes  # 2

        self.cls_head = nn.Sequential(
            nn.Conv2d(1, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(0.10),
            nn.Conv2d(channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.proj_head = ProjectionHead(dim_in=channels, proj_dim=self.proj_dim)

    def forward(self, input_src, input_thick, input_thin, with_embed=True, is_eval=False):  # ##
        if len(input_src.shape) == 3:
            input_src, input_thick, input_thin = input_src.unsqueeze(1), input_thick.unsqueeze(1), input_thin.unsqueeze(1)
        agg_coeff = self.backbone(input_src, input_thick, input_thin)  # ##
        prod_sal = self.adagg(input_thick, input_thin, agg_coeff)
        # agg_coeff (2, 9, 304, 304) prod_sal (2, 1, 304, 304)
        out = self.cls_head(prod_sal)  # out(2, 2, 304, 304)
        emb = self.proj_head(prod_sal)  # emb (2, 256, 304, 304)

        return {'seg': out, 'embed': emb}


class FusionContrastMEM(nn.Module):
    def __init__(self, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1, proj_dim=256, m=0.999,
                 with_masked_ppm=False, memory_size=20, num_classes=2):
        super().__init__()
        self.backbone = base(channels, pn_size, kernel_size, avg, std)
        self.adagg = adaptive_aggregation(pn_size)

        self.m = m
        self.r = memory_size
        self.with_masked_ppm = with_masked_ppm

        self.encoder_q = FusionContrast()

        self.register_buffer("segment_queue", torch.randn(num_classes, self.r, channels))
        self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer("pixel_queue", torch.randn(num_classes, self.r, channels))
        self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
        self.register_buffer("pixel_queue_ptr", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, input_src, input_thick, input_thin, lb_q=None, with_embed=True, is_eval=False):  # TODO: im_q
        agg_coeff = self.backbone(input_src, input_thick, input_thin)
        prod_sal = self.adagg(input_thick, input_thin, agg_coeff)
        im_q = prod_sal
        # agg_coeff (2, 9, 304, 304)
        # im_q (2, 1, 304, 304), input_src (2, 1, 304, 304), input_thick (2, 1, 304, 304), input_thin (2, 1, 304, 304)
        if is_eval is True or lb_q is None:
            ret = self.encoder_q(im_q, im_q, im_q, with_embed=with_embed)
            return ret

        ret = self.encoder_q(im_q, im_q, im_q)

        q = ret['embed']
        out = ret['seg']

        return {'seg': out, 'embed': q, 'key': q.detach(), 'lb_key': lb_q.detach()}


# ----------------------------------------------------------------------------------------------------------------------

class FusionSegmenter(nn.Module):
    def __init__(self, input_nc=3, inter_nc=64):  # , num_classes=2
        super(FusionSegmenter, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc, inter_nc, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm2d(inter_nc),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_nc, inter_nc, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(inter_nc),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_nc, inter_nc, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(inter_nc),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bicubic")
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(inter_nc, inter_nc, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(inter_nc),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(inter_nc, inter_nc, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm2d(inter_nc),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(inter_nc, 1, kernel_size=1, stride=1, padding=0, bias=True)  # num_classes ,
            # nn.Softmax(dim=1)
        )
    
    def forward(self, input_src, input_thick, input_thin):
        x = torch.cat((input_src, input_thick, input_thin), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.up(x)
        
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final_conv(x)
        
        return F.sigmoid(x)


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, size_average=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self, pred, gt):
        gt_oh = torch.cat((gt, 1.0 - gt), dim=1)  # [b, 2, h, w]
        pt = (gt_oh * pred).sum(1)  # [b, h, w]
        focal_map = - self.alpha * torch.pow(1.0 - pt, self.gamma) * torch.log2(clip_by_tensor(pt, 1e-12, 1.0))  # [b, h, w]
        
        if self.size_average:
            loss = focal_map.mean()
        else:
            loss = focal_map.sum()
        
        return loss
