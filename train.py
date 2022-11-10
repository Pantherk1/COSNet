# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from Utils import get_lr, adjust_lr
import numpy as np
import cv2

def train_first_stage(img_path, writer, dataloader, net, optimizer, base_lr, thin_criterion, thick_criterion, device, power, epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        thin_gt = sample[2].to(device)
        thick_gt = sample[3].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        thick_pred, thin_pred, _ = net(img)
        img_1=img[0, :, :, :].permute(1,2,0)
        img_2=thin_gt[0, :, :, :].permute(1,2,0).repeat(1,1,3)
        img_3=thick_gt[0, :, :, :].permute(1,2,0).repeat(1,1,3)
        img_4=thin_pred[0, :, :, :].permute(1,2,0).repeat(1,1,3)
        img_5=thick_pred[0, :, :, :].permute(1,2,0).repeat(1,1,3)
        img_save=torch.cat([img_1,img_2,img_4,img_3,img_5],dim=1)*255
        img_in=np.uint8(img_save.detach().cpu().numpy())
        cv2.imwrite(img_path+'/%d_%d.png'%(epoch,step),img_in)
        loss = thin_criterion(thin_pred, thin_gt) + thick_criterion(thick_pred, thick_gt)  # 可加权
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        # writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))
        # viz.plot("train loss", loss.item())
        
        # 写入当前lr
        current_lr = get_lr(optimizer)
        # viz.plot("learning rate", current_lr)
        # writer.add_scalars("learning_rate", {"lr": current_lr}, niter)
    
    print("epoch %d loss: %0.4f" % (epoch, epoch_loss/step))
    print("current learning rate: %f" % current_lr)
    writer.write('%d,%.4f,%f,'%(epoch,epoch_loss/step,current_lr))
    
    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)
    
    return net
#########################################################################
def dequeue_and_enqueue( args,keys, labels,segment_queue, segment_queue_ptr,pixel_queue, pixel_queue_ptr):
    batch_size = keys.shape[0]
    feat_dim = keys.shape[1]

    for bs in range(batch_size):
        this_feat = keys[bs].contiguous().view(feat_dim, -1)
        this_label = labels[bs].contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x > 0]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero(as_tuple=False)

            # segment enqueue and dequeue
            feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
            ptr = int(segment_queue_ptr[lb])
            segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
            segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % args.memory_size

            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, args.pixel_update_freq)
            feat = this_feat[:, perm[:K]]
            feat = torch.transpose(feat, 0, 1)
            ptr = int(pixel_queue_ptr[lb])

            if ptr + K >=args.memory_size:
                pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                pixel_queue_ptr[lb] = 0
            else:
                pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + 1) % args.memory_size


def train_first_stage_contrast(img_path,args,writer, dataloader, net, optimizer, base_lr, criterion, device,
                               power, epoch, num_epochs=100, with_memory=True, with_embed=True):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        gt = sample[1].to(device)
        thin_gt = sample[2].to(device)
        thick_gt = sample[3].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        if with_memory:
            outputs = net(img, [thick_gt,thin_gt],with_embed=with_embed)
            outputs['pixel_queue_thick'] = net.pixel_queue_thick
            outputs['pixel_queue_ptr_thick'] = net.pixel_queue_ptr_thick
            outputs['segment_queue_thick'] = net.segment_queue_thick
            outputs['segment_queue_ptr_thick'] = net.segment_queue_ptr_thick

            outputs['pixel_queue_thin'] = net.pixel_queue_thin
            outputs['pixel_queue_ptr_thin'] = net.pixel_queue_ptr_thin
            outputs['segment_queue_thin'] = net.segment_queue_thin
            outputs['segment_queue_ptr_thin'] = net.segment_queue_ptr_thin
        else:
            # img (2, 3, 304, 304)
            outputs = net(img, with_embed=with_embed)
        thick_pred, thin_pred = outputs['seg_thick'],outputs['seg_thin']
        img_1=img[0, :, :, :].permute(1,2,0)
        img_2=thin_gt[0, :, :, :].permute(1,2,0).repeat(1,1,3)
        img_3=thick_gt[0, :, :, :].permute(1,2,0).repeat(1,1,3)
        img_4=thin_pred[0, :, :, :].permute(1,2,0).repeat(1,1,3)
        img_5=thick_pred[0, :, :, :].permute(1,2,0).repeat(1,1,3)
        img_save=torch.cat([img_1,img_2,img_4,img_3,img_5],dim=1)*255
        img_in=np.uint8(img_save.detach().cpu().numpy())
        cv2.imwrite(img_path+'/%d_%d.png'%(epoch,step),img_in)

        # forward

        backward_loss = display_loss = criterion(outputs,[thick_gt,thin_gt])
        if with_memory and 'key' in outputs and 'lb_key' in outputs:
            dequeue_and_enqueue(args,outputs['key_thick'], outputs['lb_thick_key'],
                                      segment_queue=net.segment_queue_thick,
                                      segment_queue_ptr=net.segment_queue_ptr_thick,
                                      pixel_queue=net.pixel_queue_thick,
                                      pixel_queue_ptr=net.pixel_queue_ptr_thick)
            dequeue_and_enqueue(args,outputs['key_thin'], outputs['lb_thin_key'],
                                segment_queue=net.segment_queue_thin,
                                segment_queue_ptr=net.segment_queue_ptr_thin,
                                pixel_queue=net.pixel_queue_thin,
                                pixel_queue_ptr=net.pixel_queue_ptr_thin)
        backward_loss.backward()
        optimizer.step()
        epoch_loss += backward_loss.item()

        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
       
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, backward_loss.item()))
        # viz.plot("train loss", loss.item())

        # 写入当前lr
        current_lr = get_lr(optimizer)
        # viz.plot("learning rate", current_lr)
       
    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)
    writer.write('%d,%.4f,%f,'%(epoch,epoch_loss/step,current_lr))
    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)

    return net




def train_second_stage( writer, dataloader, front_net_thick, front_net_thin, fusion_net, optimizer, base_lr, criterion, device, power, epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        gt = sample[1].to(device)
        with torch.no_grad(): 
            thick_pred = front_net_thick(img)
            thin_pred = front_net_thin(img)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        fusion_pred = fusion_net(img[:, :1, :, :], thick_pred[0], thin_pred[0])
        # viz.img(name="images", img_=img[0, :, :, :])
        # viz.img(name="labels", img_=gt[0, :, :, :])
        # viz.img(name="prediction", img_=fusion_pred[0, :, :, :])
        loss = criterion(fusion_pred, gt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))
        # viz.plot("train loss", loss.item())
        
        # 写入当前lr
        current_lr = get_lr(optimizer)
        # viz.plot("learning rate", current_lr)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)
    
    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)
    
    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)
    
    return fusion_net

# ----------------------------------------------------------------------------------------------------------------------


def train_second_stage_contrast(writer, dataloader, front_net_thick, front_net_thin, fusion_net, optimizer, base_lr,
                       criterion, device, power, epoch, num_epochs=100, with_memory=True, with_embed=True):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)  # 2, 3, 304, 304
        gt = sample[1].to(device)  # 2, 1, 304, 304
        with torch.no_grad():
            thick_pred = front_net_thick(img)
            thin_pred = front_net_thin(img)
            if isinstance(thick_pred, tuple):
                thick_pred = thick_pred[0]
            if isinstance(thin_pred, tuple):
                thin_pred = thin_pred[0]
            # thick_pred thin_pred (2, 1, 304, 304)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # fusion_pred = fusion_net(img[:, :1, :, :], thick_pred, thin_pred)
        if with_memory:
            outputs = fusion_net(img[:, :1, :, :], thick_pred, thin_pred)
            outputs['pixel_queue'] = fusion_net.pixel_queue  # 2, 5000, 64
            outputs['pixel_queue_ptr'] = fusion_net.pixel_queue_ptr  # (2, )
            outputs['segment_queue'] = fusion_net.segment_queue  # 2, 5000, 64
            outputs['segment_queue_ptr'] = fusion_net.segment_queue_ptr  # (2, )
        else:
            # img (2, 3, 304, 304)
            outputs = fusion_net(img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :], with_embed=with_embed)  # TODO: inputs
        # TODO 原代码中的展示
        # viz.img(name="images", img_=img[0, :, :, :])
        # viz.img(name="labels", img_=gt[0, :, :, :])
        # viz.img(name="prediction", img_=outputs[0, :, :, :])  # TODO: 注释了
        loss = criterion(outputs, gt)  # without embed outputs:{'seg': (2, 2, 304, 304), 'embed': (2, 256, 304, 304)}  gt (2, 1, 304, 304)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if with_memory and 'key' in outputs and 'lb_key' in outputs:
            from fusion_main_contrast import _dequeue_and_enqueue
            _dequeue_and_enqueue(outputs['key'], outputs['lb_key'],
                                 segment_queue=fusion_net.module.segment_queue,
                                 segment_queue_ptr=fusion_net.module.segment_queue_ptr,
                                 pixel_queue=fusion_net.module.pixel_queue,
                                 pixel_queue_ptr=fusion_net.module.pixel_queue_ptr)

        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))
        # viz.plot("train loss", loss.item())

        # 写入当前lr
        current_lr = get_lr(optimizer)
        # viz.plot("learning rate", current_lr)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)

    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)

    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)

    return fusion_net

    
