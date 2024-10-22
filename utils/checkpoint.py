import os
import shutil
import torch

def save_checkpoint(state, is_best, args, filename='default'):
    if filename=='default':
        filename = 'mcn_%s_batch%d'%(args.dataset,args.samples_per_gpu)
    print("=> saving checkpoint '{}'".format(filename))
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    checkpoint_name = './saved_models/%s_checkpoint.pth.tar'%(filename)
    best_name = './saved_models/%s_model_best.pth.tar'%(filename)
    torch.save(state, checkpoint_name)
    if is_best:
        print("=> saving best model '{}'".format(best_name))
        shutil.copyfile(checkpoint_name, best_name)

def load_pretrain(model, args, logging, rank):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain)
        pretrained_dict = checkpoint['state_dict']
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()])!=0)
        model_dict.update(pretrained_dict)
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)
        print("=> loaded pretrain model at {}"
              .format(args.pretrain))
        if rank == 0:
            logging.info("=> loaded pretrain model at {}"
                .format(args.pretrain))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no pretrained file found at '{}'".format(args.pretrain)))
        if rank == 0:
            logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    return model

def load_pretrain_ddp(model, args):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()])!=0)
        model_dict.update(pretrained_dict)
        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
            model.module.load_state_dict(model_dict)
        else:
            state_dict = model.state_dict()
            model.load_state_dict(model_dict)
        print("load ")
        print("=> loaded pretrain model at {}"
              .format(args.pretrain))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no pretrained file found at '{}'".format(args.pretrain)))
    return model


def load_resume(model, optimizer, args, logging, rank):
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        if rank == 0:
            logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        print("epoch: ", args.start_epoch)
        args.best_iou = checkpoint['best_iou']
        print("best iou: ", args.best_iou)
        state_dict = checkpoint['state_dict']

        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        new_state_dict = {k:v for k,v in state_dict.items() if k in model_dict}
        model_dict.update(new_state_dict)

        
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
        print("load successfully!")
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
        if rank == 0:
            logging.info(("=> no checkpoint found at '{}'".format(args.resume)))
    return model
