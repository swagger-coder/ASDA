import time
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast

from model.model import *
from dataset.data_loader import *
from utils.losses import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.utils import dice_loss, sigmoid_focal_loss

use_cuda = torch.cuda.is_available()
print("use_cuda, ", use_cuda)


def train_epoch(rank, args, train_loader, model, optimizer, epoch, scaler, logger):
    print('train at epoch %d'%epoch)
    batch_time = AverageMeter()
    losses = AverageMeter()
    dice_losses = AverageMeter()
    sigmoid_focal_losses = AverageMeter()
    cos_losses = AverageMeter()
    model.train()
    end = time.time()

    for batch_idx, (imgs, word_id, word_mask, bbox, seg_map) in enumerate(train_loader):
        imgs = imgs.cuda(rank, non_blocking=True)
        word_id = word_id.cuda(rank, non_blocking=True)
        word_mask = word_mask.cuda(rank, non_blocking=True)
        seg_map = seg_map.cuda(rank, non_blocking=True)
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        seg_map = Variable(seg_map)

        with autocast():
            mask_out = model(image, word_id, word_mask)
            loss = 0.
            
            mask_out_np = mask_out.data.cpu().numpy() # [bs, 1, 208, 208]
            seg_map_np = seg_map.cpu().numpy() # [bs, 1, 208, 208]
            seg_iou = cal_seg_iou_loss(seg_map_np, mask_out_np, args.seg_thresh)
         
            dice_loss_ = dice_loss(mask_out, seg_map)
            sigmoid_focal_loss_ = sigmoid_focal_loss(mask_out, seg_map)

            loss += dice_loss_ + sigmoid_focal_loss_

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), imgs.size(0))
        dice_losses.update(dice_loss_.item(), imgs.size(0))
        sigmoid_focal_losses.update(sigmoid_focal_loss_.item(), imgs.size(0))
        cos_losses.update(seg_iou.mean().item(), imgs.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if rank == 0 and batch_idx % args.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'dice_losses {dice_losses.val:.4f} ({dice_losses.avg:.4f})\t' \
                'sigmoid_focal_losses {sigmoid_focal_losses.val:.4f} ({sigmoid_focal_losses.avg:.4f})\t' \
                'IoU {cos_loss.val:.4f} ({cos_loss.avg:.4f})\t' \
                .format(epoch, batch_idx, len(train_loader), batch_time=batch_time, loss=losses, dice_losses=dice_losses, sigmoid_focal_losses=sigmoid_focal_losses, cos_loss=cos_losses)
            print(print_str)
            logger.info(print_str)

    return losses.avg

def validate_epoch(args, val_loader, model, logger, mode='val'):
    print('begin test')
    batch_time = AverageMeter()
    miou = AverageMeter()
    miou_seg = AverageMeter()

    prec=dict()
    thresholds = np.arange(0.5, 1, 0.05)

    for thresh in thresholds:
        prec[thresh]= AverageMeter()

    model.eval()
    end = time.time()
    idx = 0

    t_all = []

    for batch_idx, (imgs, word_id, word_mask, bbox, seg_map, ratio, dw, dh, im_id, phrase, draw_img) in enumerate(val_loader):
        
        imgs = imgs.cuda(0)
        word_id = word_id.cuda(0)
        word_mask = word_mask.cuda(0)
        seg_map = seg_map.cuda(0)
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        seg_map = Variable(seg_map)

        t1 = time.time()
        with torch.no_grad():
            mask_out = model(image, word_id, word_mask)
            mask_out = mask_out.sigmoid()

        t2 = time.time()
        t_all.append(t2-t1)

        ## test: convert pred, gt box to original scale with meta-info
        ih = seg_map.shape[-2]
        iw = seg_map.shape[-1]
        nh = int(ih * ratio)
        nw = int(iw * ratio)
        top, bottom = int(dh[0]), nh + int(dh[0])
        left, right = int(dw[0]), nw + int(dw[0])
        ratio = float(ratio)
        new_shape = (iw, ih)
        
        ## revert image for visualization
        seg_map_np = seg_map[0,:,:,:].data.cpu().numpy().transpose(1,2,0)
        seg_map_np = cv2.resize(seg_map_np, new_shape, interpolation=cv2.INTER_CUBIC)
        img_np = imgs[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)

        img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))
        
        # seg
        mask_out = mask_out[0].data.cpu().numpy().transpose(1,2,0)
        mask_out = cv2.resize(mask_out, (args.size, args.size))
        mask_out_np = mask_out[top:bottom, left:right]
        mask_out_np = cv2.resize(mask_out_np, new_shape)
        seg_iou, seg_prec = cal_seg_iou(seg_map[0].cpu().numpy(), mask_out_np, args.seg_thresh)
        miou_seg.update(seg_iou, imgs.size(0))
        for thresh in thresholds:
            prec[thresh].update(seg_prec[thresh], imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 1000 == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'seg_iu {seg.val:.4f} ({seg.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, seg=miou_seg)
            print(print_str)
            logger.info(print_str)  
        idx = idx + 1
        
    print(miou_seg.avg)
    for thresh in thresholds:
            print("prec@%f: %f"%(thresh,float(prec[thresh].avg)))
            logger.info("prec@%f:%f"%(thresh,float(prec[thresh].avg)))
    logger.info("%f,%f"%(float(miou.avg), miou_seg.avg))
    return miou_seg.avg, prec

