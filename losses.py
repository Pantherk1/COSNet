# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
from contrastive_loss import loss_contrast, loss_contrast_mem


# ---------------------------------------------------------------------------------------------
## contrastive losses part
# TODO: parser核对完善
def contrast_config():
    params = {"ce_ignore_index": [],   # ignore index
              "ce_weight": 11,
              "ce_reduction": 11,  # str, default reduction = "elementwise_mean"
              "use_sigmoid": True,
              "num_classes": 2,
              "rmi_radius": 1,
              "rmi_pool_way": 0,
              "rmi_pool_size": 11,
              "rmi_pool_stride": 11,
              "loss_weight_lambda": 11,
              "loss_weight": 11,
              "lambda_way": 11}
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", default=0.1, type=float, dest="contrast:temperature")
    parser.add_argument("--base_temperature", default=0.1, type=float, dest="contrast:base_temperature")
    parser.add_argument("--params", default=params, type=list, dest="loss:params")
    parser.add_argument("--max_sample", default=100, type=int, dest="contrast:max_sample")
    parser.add_argument("--max_views", default=100, type=int, dest="contrast:max_views")
    parser.add_argument("--loss_weight", default=0.04, type=float, dest="contrast:loss_weight")
    parser.add_argument("--use_rmi", default=False, type=bool, dest="contrast:use_rmi")
    parser.add_argument("--use_lovasz", default=False, type=bool, dest="contrast:use_lovasz")

    parse = parser.parse_args()
    return parse


def contrast_loss(config):
    loss_select = {"use_mem": False, "loss_name": "PixelContrastLoss"}
    contrast_losses_name = ['PixelContrastLoss', 'ContrastCELoss', 'ContrastAuxCELoss']
    if loss_select["use_mem"]:
        if loss_select["loss_name"] == "PixelContrastLoss":
            loss = loss_contrast_mem.PixelContrastLoss(configer=config)
        elif loss_select["loss_name"] == "ContrastCELoss":
            loss = loss_contrast_mem.ContrastCELoss(configer=config)
        elif loss_select["loss_name"] == "ContrastAuxCELoss":
            loss = loss_contrast_mem.ContrastAuxCELoss(configer=config)
        else:
            raise NameError(f"the {loss_select['loss_name']} not found.")
    else:
        if loss_select["loss_name"] == "PixelContrastLoss":
            loss = loss_contrast.PixelContrastLoss(configer=config)
        elif loss_select["loss_name"] == "ContrastCELoss":
            loss = loss_contrast.ContrastCELoss(configer=config)
        elif loss_select["loss_name"] == "ContrastAuxCELoss":
            loss = loss_contrast.ContrastAuxCELoss(configer=config)
        else:
            raise NameError(f"the {loss_select['loss_name']} not found.")
    return loss

# ---------------------------------------------------------------------------------------------


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: clipped tensor
    """
    t = t.float()
    
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    
    return result


class dice_loss(nn.Module):
    def __init__(self, eps=1e-12):
        super(dice_loss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, gt):
        assert pred.size() == gt.size() and pred.size()[1] == 1
        
        N = pred.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
        intersection = pred_flat * gt_flat
        dice = (2.0 * intersection.sum(1) + self.eps) / (pred_flat.sum(1) + gt_flat.sum(1) + self.eps)
        loss = 1.0 - dice.mean()
        
        return loss


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, size_average=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self, pred, gt):
        assert pred.size() == gt.size() and pred.size()[1] == 1
        
        pred_oh = torch.cat((pred, 1.0 - pred), dim=1)  # [b, 2, h, w]
        gt_oh = torch.cat((gt, 1.0 - gt), dim=1)  # [b, 2, h, w]
        pt = (gt_oh * pred_oh).sum(1)  # [b, h, w]
        focal_map = - self.alpha * torch.pow(1.0 - pt, self.gamma) * torch.log2(clip_by_tensor(pt, 1e-12, 1.0))  # [b, h, w]
        
        if self.size_average:
            loss = focal_map.mean()
        else:
            loss = focal_map.sum()
        
        return loss


# 构建损失函数，可扩展
def build_loss(loss):
    if loss == "mse":
        criterion = nn.MSELoss()
    elif loss == "l1":
        criterion = nn.L1Loss()
    elif loss == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif loss == "bce":
        criterion = focal_loss(alpha=1.0, gamma=0.0)
    elif loss == "focal":
        criterion = focal_loss(alpha=0.25, gamma=2.0)
    elif loss == "dice":
        criterion = dice_loss()
    else:
        raise NotImplementedError('loss [%s] is not implemented' % loss)
    
    return criterion
