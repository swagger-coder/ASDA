import os
import sys
import argparse
import random
import datetime
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed

from tensorboardX import SummaryWriter

#import apex.amp as amp
from torch.cuda.amp import autocast as autocast, GradScaler

from model.model import *
from engine.engine import *

from dataset.data_loader import *
from utils.losses import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume
from utils.logger import setup_logger

def get_args():
    parser = argparse.ArgumentParser(description='Dataloader test')
    parser.add_argument('--gpu', default='2', help='gpu id')
    parser.add_argument('--ngpu', default=2, type=int, help='gpu num')
    parser.add_argument('--workers', default=4, type=int, help='num workers for data loading')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    parser.add_argument('--clip_model', default='ViT-B/16', type=str, help='clip model RN50 RN101 ViT-B/32')
    parser.add_argument('--nb_epoch', default=32, type=int, help='training epoch')
    parser.add_argument('--lr', default=0.000025, type=float, help='batch size 16 learning rate')
    parser.add_argument('--power', default=0.1, type=float, help='lr poly power')
    parser.add_argument('--steps', default=[18, 28], type=list, help='in which step lr decay by power')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--size', default=416, type=int, help='image size')
    parser.add_argument('--dataset', default='grefcoco', type=str,
                        help='refcoco/refcoco+/refcocog/grefcoco')
    
    parser.add_argument('--num_query', default=16, type=int, help='the number of query')
    parser.add_argument('--w_seg', default=0.1, type=float, help='weight of the seg loss')
    parser.add_argument('--w_coord', default=5, type=float, help='weight of the reg loss')
    parser.add_argument('--tunelang', dest='tunelang', default=True, action='store_true', help='if finetune language model')
    parser.add_argument('--anchor_imsize', default=416, type=int,
                        help='scale used to calculate anchors defined in model cfg file')
    parser.add_argument('--data_root', type=str, default='./ln_data',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='./data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--time', default=17, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='path to ReferIt splits data folder')

    parser.add_argument('--fusion_dim', default=768, type=int,
                        help='fusion module embedding dimensions')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    
    parser.add_argument('--seg_thresh', default=0.35, type=float, help='seg score above this value means foreground')
    parser.add_argument('--seg_out_stride', default=2, type=int, help='the seg out stride')
    parser.add_argument('--best_iou', default=-float('Inf'), type=int, help='the best accu')

    global args, anchors_full, writer, logger
    args = parser.parse_args()
    args.gsize = 32 
    # 获取格式化日期
    args.date = datetime.datetime.now().strftime('%Y%m%d')
    if args.savename=='default':
        args.savename = 'model_v1_%s_batch%d_%s'%(args.dataset, args.batch_size, args.date)
    os.makedirs(args.log_dir, exist_ok=True)
    args.lr = round(args.lr * (args.batch_size *  args.ngpu / 16), 6)
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')

    return args

def main(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    if(torch.cuda.is_available()):
        n_gpus = torch.cuda.device_count()
        print("Running DDP with {} GPUs".format(n_gpus))
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args,))
    else:
        print("Please use GPU for training")

def run(rank, n_gpus, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)

    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    ## save logs
    logger = setup_logger(output=os.path.join(args.log_dir, args.savename), distributed_rank=rank, color=False, name="model-v1")
    
    logger.info(str(sys.argv))
    logger.info(str(args))
    if rank == 0:
        writer = SummaryWriter(comment=args.savename)

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    train_dataset = ReferDataset(data_root=args.data_root,
                         dataset=args.dataset,
                         split_root=args.split_root,
                         split='train',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         augment=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=n_gpus, rank=rank, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers, sampler=train_sampler)
    

    if rank == 0:
        val_dataset = ReferDataset(data_root=args.data_root,
                            dataset=args.dataset,
                            split_root=args.split_root,
                            split='val',
                            imsize = args.size,
                            transform=input_transform,
                            max_query_len=args.time)

    
    
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    pin_memory=True, drop_last=True, num_workers=args.workers)


    ## Model
    model = Model(clip_model=args.clip_model, tunelang=args.tunelang, num_query=args.num_query, fusion_dim=args.fusion_dim).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model_without_ddp = model.module

    args.start_epoch = 0
    if args.pretrain and os.path.isfile(args.pretrain):
        model=load_pretrain(model,args,logger, rank)
        model.to(rank)
    
    visu_param = [param for name, param in model_without_ddp.named_parameters() if 'visumodel' in name]
    text_param = [param for name, param in model_without_ddp.named_parameters() if 'textmodel' in name]
    rest_param = [param for name, param in model_without_ddp.named_parameters() if 'textmodel' not in name and 'visumodel' not in name]

    sum_visu = sum([param.nelement() for param in visu_param])
    sum_text = sum([param.nelement() for param in text_param])
    sum_fusion = sum([param.nelement() for param in rest_param])
    if rank == 0:
        print('Num of parameters:', sum([param.nelement() for param in model_without_ddp.parameters()]))
        logger.info('Num of parameters:%d'%int(sum([param.nelement() for param in model_without_ddp.parameters()])))
        print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    ## optimizer; adam default
    if args.tunelang:
        optimizer = torch.optim.Adam([{'params': rest_param, 'lr': args.lr},
                                      {'params': visu_param, 'lr': args.lr / 10.},
                                      {'params': text_param, 'lr': args.lr / 10.}])
    else:
        optimizer = torch.optim.Adam([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr / 10.}], lr=args.lr)

    # Initialization
    scaler = GradScaler()

    best_miou_seg = -float('Inf')
    if args.resume:
        model = load_resume(model, optimizer, args, logger, rank)
        model.to(rank)
        best_miou_seg = args.best_iou
        print(best_miou_seg)
    
    
    for epoch in range(args.start_epoch, args.nb_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        loss = train_epoch(rank, args, train_loader, model, optimizer, epoch, scaler, logger)
        if rank == 0:
            writer.add_scalar('loss', loss, global_step=epoch)
            miou_seg = 0
            if epoch == 0 or epoch > 10:
                miou_seg, prec = validate_epoch(args, val_loader, model, logger, 'Val')
                writer.add_scalar('miou_seg', miou_seg, global_step=epoch)
                thresholds = np.arange(0.5, 1, 0.05)
                for thresh in thresholds:
                    writer.add_scalar('prec@%f'%thresh, prec[thresh].avg, global_step=epoch)

            ## remember best accu and save checkpoint
            is_best = miou_seg > best_miou_seg
            best_miou_seg= max(miou_seg, best_miou_seg)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_iou': best_miou_seg,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args, filename=args.savename)
    print('\nBest Accu: %f\n'%best_miou_seg)
    logger.info('\nBest Accu: %f\n'%best_miou_seg)

if __name__ == "__main__":
    args = get_args()
    main(args)
