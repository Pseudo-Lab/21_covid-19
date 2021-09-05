import sys
sys.path.append('FixMatch-pytorch')
sys.path.append('pytorch-image-models')

import argparse
import logging
import math
import os
import random
import shutil
import time
from datetime import datetime
from collections import OrderedDict

import yaml
import numpy as np
import torch
from torch.cuda import amp
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from utils import AverageMeter, accuracy

# from timm.models import create_model
from geguri.model import MultiTaskResNet, MultiTaskEfficientNet

from fixmatch.datasets import (
    set_image_size as set_unlabeled_image_size,
    get_unlabeled_data,
    get_transform_unlabeled,
)
from misc.module import attach_multi_stage_dropout

from geguri.datasets import (
    set_image_size as set_labeled_image_size,
    get_siim_data,
    train_augment,
)

logger = logging.getLogger(__name__)
best_acc = 0
best_map = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(optimizer,
                                      num_warmup_steps,
                                      last_epoch=-1):

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 1.

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--model', default='wideresnet', type=str)
    parser.add_argument('--validation-fold', default=0, type=int)
    parser.add_argument('--datatype', type=str, default='px1280')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--msd', action='store_true', default=False)

    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--total-steps', default=2**17, type=int,
                        help='number of total steps to run')
    parser.add_argument('--num-epochs', default=-1, type=int)
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=1126, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()

    # hard code
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    project_root = os.path.abspath(project_root)
    data_root = os.path.join(project_root, 'data')
    args.dataroot = data_root

    set_labeled_image_size(args.img_size)
    set_unlabeled_image_size(args.img_size)

    global best_acc, best_map

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    args.out = os.path.join(args.out, f'fixmatch-aux-{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    # set num_classes
    args.num_classes = 4

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if 'efficient' in args.model:
        model = MultiTaskEfficientNet(args)
    elif 'res' in args.model:
        model = MultiTaskResNet(args)
    else:
        raise NotImplementedError
    # model = create_model(
    #     model_name=args.model,
    #     pretrained=True,
    #     num_classes=args.num_classes,
    #     checkpoint_path='',     # TODO;
    # )

    labeled_dataset = get_siim_data(args, down_ratio=model.down_ratio, validation=False, transform=train_augment)
    test_dataset = get_siim_data(args, down_ratio=model.down_ratio, validation=True, transform=None)
    unlabeled_dataset = get_unlabeled_data(args, transform=get_transform_unlabeled())

    if args.msd:
        model = attach_multi_stage_dropout(model, args.num_classes, use_mask=True)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # hard-code
    # TODO; make use-len-loader option
    args.eval_step = len(labeled_trainloader)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # TODO; optimizer factory
    # optimizer = optim.AdamW(grouped_parameters, lr=args.lr)
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    if args.num_epochs > 0:
        args.epochs = args.num_epochs
        args.total_steps = args.epochs * args.eval_step

    # TODO; scheduler factory
    scheduler = get_constant_schedule_with_warmup(
        optimizer, args.warmup,
    )
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer, args.warmup, args.total_steps)
    scaler = amp.GradScaler() if args.amp else None

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        best_map = checkpoint['best_map']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if checkpoint['scaler'] is not None and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])

    if args.local_rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    except_keys = ['device', 'writer']
    dump = {k: v for k, v in args.__dict__.items() if k not in except_keys}
    with open(os.path.join(args.out, 'args.yaml'), 'w') as fs:
        yaml.safe_dump(dump, fs, sort_keys=False)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, scaler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, scaler):
    global best_acc, best_map
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_m = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, masks_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, masks_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)

            masks_x = masks_x.to(args.device)
            targets_x = targets_x.to(args.device)

            if args.amp:
                with amp.autocast():
                    logits, masks_pred = model(inputs)
            else:
                logits, masks_pred = model(inputs)

            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            masks_pred = de_interleave(masks_pred, 2*args.mu+1)
            masks_pred_x = masks_pred[:batch_size]
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            Lm = F.binary_cross_entropy_with_logits(masks_pred_x, masks_x)

            pseudo_label = torch.softmax(logits_u_w.detach().float()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

            loss = Lx + Lm + args.lambda_u * Lu

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_m.update(Lm.item())
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_m: {loss_m:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_m=losses_m.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc, test_map = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_m', losses_m.avg, epoch)
            args.writer.add_scalar('train/5.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_map', test_map, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            args.writer.add_scalar('test/3.test_acc', test_acc, epoch)

            is_best = test_map > best_map
            best_map = max(test_map, best_map)
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'mean_ap': test_map,
                'best_map': best_map,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
            }, is_best, args.out)

            logger.info('Best mAP: {:.5f}'.format(best_map))
            logger.info('Best top-1 acc: {:.5f}'.format(best_acc))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def np_metric_map_curve_by_class(probability, truth):
    num_sample, num_label = probability.shape
    score = []
    for i in range(num_label):
        s = average_precision_score(truth == i, probability[:, i])
        score.append(s)
    score = np.array(score)
    return score


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    probabilities = []
    ground_truth = []
    with torch.no_grad():
        for batch_idx, (inputs, masks, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            masks = masks.to(args.device)
            targets = targets.to(args.device)
            outputs, masks_pred = model(inputs)
            loss_class = F.cross_entropy(outputs, targets)
            loss_mask = F.binary_cross_entropy_with_logits(masks_pred, masks)
            loss = loss_class + loss_mask
            probs = torch.softmax(outputs, -1)

            prec1, = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            probabilities.append(probs.cpu().numpy())
            ground_truth.append(targets.cpu().numpy())
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}.".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    ground_truth = np.concatenate(ground_truth, 0)
    probabilities = np.concatenate(probabilities, 0)
    mean_ap = np_metric_map_curve_by_class(probabilities, ground_truth)

    logger.info("top-1 acc: {:.5f}".format(top1.avg))
    logger.info("mAP: {:.5f}".format(mean_ap.mean()))
    return losses.avg, top1.avg, mean_ap.mean()


if __name__ == '__main__':
    main()
