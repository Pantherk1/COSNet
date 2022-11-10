from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.loss_helper import FSAuxCELoss, FSAuxRMILoss, FSCELoss
from .utils.logger import Logger as Log


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.temperature
        self.base_temperature = self.configer.base_temperature

        self.ignore_label = -1

        self.max_samples = self.configer.max_samples
        self.max_views = self.configer.max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero(as_tuple=False).shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple=False)
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(as_tuple=False)

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask


        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        # labels = labels.unsqueeze(1).float().clone()
        labels = labels.float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss


class ContrastMSELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastMSELoss, self).__init__()

        self.configer = configer

        self.loss_weight = self.configer.loss_weight
        self.use_rmi = self.configer.use_rmi

        if self.use_rmi:
            self.seg_criterion = nn.MSELoss()
        else:
            self.seg_criterion = nn.MSELoss()

        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target, with_embed=True):
        target_thick=target[0]
        target_thin = target[1]

        batch,h, w = target_thick.size(0),target_thick.size(2), target_thick.size(3)

        assert "seg_thick" in preds
        assert "embed_thick" in preds

        assert "seg_thin" in preds
        assert "embed_thin" in preds

        seg_thick = preds['seg_thick']
        embedding_thick = preds['embed_thick']

        seg_thin = preds['seg_thin']
        embedding_thin = preds['embed_thin']

        pred_thick =seg_thick
        pred_thin = seg_thin

        loss_thick = self.seg_criterion(pred_thick, target_thick)
        loss_thin = self.seg_criterion(pred_thin,target_thin)

        loss=loss_thick+loss_thin


        labels = torch.zeros((batch,1, h, w)).to(target_thick)

        index1 = torch.ones_like(labels).to(labels)
        labels_thick=torch.where(target_thick>0.5,index1,labels)
        labels_thin = torch.where(target_thin > 0.5, index1, labels)

        predict_thick=torch.zeros((batch,1, h, w)).to(target_thick)
        predict_thin = torch.zeros((batch, 1, h, w)).to(target_thick)

        threshold=[]

        for i in range(pred_thick.size(0)):
            pred_img_thick = np.array(pred_thick[i,0].cpu().detach().numpy() * 255, np.uint8)
            pred_img_thin = np.array(pred_thin[i,0].cpu().detach().numpy()  * 255, np.uint8)

            thresh_thick, _ = cv2.threshold(pred_img_thick, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_thin, _ = cv2.threshold(pred_img_thin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            threshold.append([thresh_thick/255,thresh_thin/255])

        for i in range(pred_thick.size(0)):
            predict_thick[i,0] = torch.where(pred_thick[i,0]>threshold[i][0],index1[i,0],labels[i,0])
            predict_thin[i,0] = torch.where(pred_thin[i,0]>threshold[i][1],index1[i,0],labels[i,0])


        loss_contrast_thick = self.contrast_criterion(embedding_thick, labels_thick, predict_thick)
        loss_contrast_thin = self.contrast_criterion(embedding_thin, labels_thin, predict_thin)

        loss_contrast=loss_contrast_thick+loss_contrast_thin
        if with_embed is True:
            # a = loss + self.loss_weight * loss_contrast
            return loss + self.loss_weight * loss_contrast
        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training



class ContrastAuxCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastAuxCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        # Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.loss_weight
        self.use_rmi = self.configer.use_rmi

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSAuxCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "seg_aux" in preds
        assert "embed" in preds

        seg = preds['seg']
        seg_aux = preds['seg_aux']
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        pred_aux = F.interpolate(input=seg_aux, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion([pred_aux, pred], target)

        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training
