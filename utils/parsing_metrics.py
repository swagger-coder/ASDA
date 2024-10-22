import torch
import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class, bg_thre=200):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        # hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        hist += _fast_hist(lt[lt<bg_thre].flatten(), lp[lt<bg_thre].flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def label_confusion_matrix(label_trues, label_preds, n_class, bg_thre=200):
    # eps=1e-20
    hist=np.zeros((n_class,n_class),dtype=float)
    """ (8,256,256), (256,256) """
    for lt,lp in zip(label_trues, label_preds):
        # hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        hist += _fast_hist(lt[lt<bg_thre].flatten(), lp[lt<bg_thre].flatten(), n_class)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # for i in range(n_class):
    # 	hist[i,:]=(hist[i,:]+eps)/sum(hist[i,:]+eps)
    return hist, iu

def body_region_confusion_matrix(label_trues, label_preds, n_class, boxes, counter):
    ## pred: [bb,region_index,c,h,w] (pred score)
    ## gt: [bb,region_index,h,w] (0-nclass score)
    label_trues = label_trues.data.cpu().numpy()
    label_preds = label_preds.data.cpu().numpy()
    hist=np.zeros((label_trues.shape[1],n_class,n_class),dtype=float)
    for body_i in range(label_trues.shape[1]):
        for bb in range(label_trues.shape[0]):
            if body_i != label_trues.shape[1]-1 and \
                torch.equal(boxes[bb,body_i,:], torch.Tensor([0.,0.,1.,1.])):
                counter+=1
                continue
            else:
                hist[body_i,:,:] += label_confusion_matrix(label_trues[bb,body_i,:,:], \
                        np.argmax(label_preds[bb,body_i,:,:,:], axis=0), n_class)[0]
    return hist

def hist_based_accu_cal(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu

def cal_seg_iou_loss(gt,pred,trsh=0.5):
    t=np.array(pred>trsh)
    p=np.array(gt>0.)
    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = (np.sum(intersection > 0 , axis=(2,3)) + 1e-10 )/ (np.sum(union > 0, axis=(2,3)) + 1e-10)
    return iou

def cal_seg_iou(gt,pred,trsh=0.5):
    #(gt.shape) [1 428 640]
    #(pred.shape)  [428 640]
    t=np.array(pred>trsh)
    p=np.array(gt>0.)
    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
    
    prec=dict()
    thresholds = np.arange(0.5, 1, 0.05)
    for thresh in thresholds:
        prec[thresh]= float(iou > thresh)
    return iou,prec

def cal_seg_iou2(gt,pred,trsh=0.5):
    #(gt.shape) [1 428 640]
    #(pred.shape)  [428 640]
    # trsh 这个会影响精度吗？
    t=np.array(pred>trsh)
    p=np.array(gt>0.)
    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
    
    prec=dict()
    thresholds = np.arange(0.5, 1, 0.05)
    for thresh in thresholds:
        prec[thresh]= float(iou > thresh)
    return iou, prec, np.sum(intersection > 0), np.sum(union > 0)