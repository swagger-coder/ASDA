# -*- coding: utf-8 -*-

"""
Custom loss function definitions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import *

class IoULoss(nn.Module):
    """
    Creates a criterion that computes the Intersection over Union (IoU)
    between a segmentation mask and its ground truth.

    Rahman, M.A. and Wang, Y:
    Optimizing Intersection-Over-Union in Deep Neural Networks for
    Image Segmentation. International Symposium on Visual Computing (2016)
    http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
    """

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        input = F.sigmoid(input)
        intersection = (input * target).sum()
        union = ((input + target) - (input * target)).sum()
        iou = intersection / union
        iou_dual = input.size(0) - iou
        if self.size_average:
            iou_dual = iou_dual / input.size(0)
        return iou_dual


def yolo_loss(input, target, gi, gj, best_n_list, w_coord=5.):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    batch = input.size(0)

    pred_bbox = Variable(torch.zeros(batch,4).cuda())
    gt_bbox = Variable(torch.zeros(batch,4).cuda())
    for ii in range(batch):
        pred_bbox[ii, 0:2] = F.sigmoid(input[ii,best_n_list[ii],0:2,gj[ii],gi[ii]])
        pred_bbox[ii, 2:4] = input[ii,best_n_list[ii],2:4,gj[ii],gi[ii]]
        gt_bbox[ii, :] = target[ii,best_n_list[ii],:4,gj[ii],gi[ii]]
    loss_x = mseloss(pred_bbox[:,0], gt_bbox[:,0])
    loss_y = mseloss(pred_bbox[:,1], gt_bbox[:,1])
    loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
    loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])

    pred_conf_list, gt_conf_list = [], []
    pred_conf_list.append(input[:,:,4,:,:].contiguous().view(batch,-1))
    gt_conf_list.append(target[:,:,4,:,:].contiguous().view(batch,-1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x+loss_y+loss_w+loss_h)*w_coord + loss_conf

def build_target(raw_coord, anchors, args):
    coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
    batch, grid = raw_coord.size(0), args.size//args.gsize
    coord[:,0] = (raw_coord[:,0] + raw_coord[:,2])/(2*args.size) # x 相对原图归一化
    coord[:,1] = (raw_coord[:,1] + raw_coord[:,3])/(2*args.size) # y
    coord[:,2] = (raw_coord[:,2] - raw_coord[:,0])/(args.size) # w
    coord[:,3] = (raw_coord[:,3] - raw_coord[:,1])/(args.size) # h
    coord = coord * grid
    bbox=torch.zeros(coord.size(0),len(anchors),5,grid,grid)
    
    best_n_list, best_gi, best_gj = [],[],[]

    for ii in range(batch):
        gi = coord[ii,0].long()
        gj = coord[ii,1].long()
        tx = coord[ii,0] - gi.float()
        ty = coord[ii,1] - gj.float()
        gw = coord[ii,2]
        gh = coord[ii,3]

        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]

        ## Get shape of gt box
        gt_box = torch.FloatTensor(np.array([0, 0, gw, gh],dtype=np.float32)).unsqueeze(0) #[1,4]
        ## Get shape of anchor box
        anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
        ## Calculate iou between gt and anchor shapes
        anch_ious = list(bbox_iou(gt_box, anchor_shapes,x1y1x2y2=False))
        ## Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))

        tw = torch.log(gw / scaled_anchors[best_n][0] + 1e-16)
        th = torch.log(gh / scaled_anchors[best_n][1] + 1e-16)

        bbox[ii, best_n, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)
    bbox = Variable(bbox.cuda())
    return bbox, best_gi, best_gj, best_n_list

def adjust_learning_rate(args, optimizer, i_iter):
    # print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
    if i_iter in args.steps:
        #lr = args.lr * args.power
        lr = args.lr * args.power ** (args.steps.index(i_iter) + 1)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr / 10
        if len(optimizer.param_groups) > 2:
            optimizer.param_groups[2]['lr'] = lr / 10

def cem_loss(co_energy):
    loss = -1.0 * torch.log(co_energy+1e-6).sum()
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)