# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import json
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from multiprocessing import cpu_count
import torch
import torchvision
from dataset.image_pose_dataset import AdhocImageDataset_che
from tqdm import tqdm
from config.config import get_args
from torch.cuda.amp import GradScaler, autocast
from dataset.img_pose_transform import Stack, ToTorchFormatTensor, GroupMultiScaleCrop,\
    GroupRandomHorizontalFlip, GroupNormalize, GroupScale, GroupCenterCrop
from models.build_model import build_model

torchvision.disable_beta_transforms_warning()
timings = {}
BATCH_SIZE = 4


def make_img_path(file_path, input):
    # read txt
    img_path = []
    label = []

    with open(file_path, 'r') as file:
        for line in file:
            # 移除行尾的换行符并分割字符串
            parts = line.strip().split()
            if parts:
                # 添加第一列的数据
                img_path.append(os.path.join(input, parts[0]))
                # 添加最后一列的数据
                label.append(int(parts[-1]))

    return img_path, label

def make_img_path_autsl(file_path, input):
    # read txt
    img_path = []
    label = []
    if 'train' in file_path:
        input = os.path.join(input, 'train')
    else:
        input = os.path.join(input, 'test')

    with open(file_path, 'r') as file:
        for line in file:
            # 移除行尾的换行符并分割字符串
            parts = line.strip().split()
            if parts:
                # 添加第一列的数据
                img_path.append(os.path.join(input, parts[0]))
                # 添加最后一列的数据
                label.append(int(parts[-1]))

    return img_path, label


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_matched_state_dict(model, checkpoint, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict:
            if curr_state_dict[key].shape == state_dict[key].shape:
                curr_state_dict[key] = state_dict[key]
                num_matched += 1
            else:
                print(key)
        else:
            print(key)
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

def feat_save(feature, output_path):

    np.save(output_path, feature)

def get_transform(is_train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    scale_range = [256, 320]
    augments = []

    if is_train:
        augments += [
            GroupMultiScaleCrop(224, [1, .875, .75, .66])
            ]
        augments += [GroupRandomHorizontalFlip(224)]
        # augments += [RandAugment(input_size=224, auto_augment='rand-m7-n4-mstd0.5-inc1', interpolation='bicubic')]

    else:
        scaled_size = 224
        augments += [
            GroupScale(scaled_size),
            GroupCenterCrop(scaled_size)
        ]
    augments += [
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std)
    ]
    # if is_train:
    #     augments += [RandomErasing(prob=0.25)]

    augmentor = torchvision.transforms.Compose(augments)
    return augmentor

def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = get_args()

    assert args.output_root != ""
    assert args.input != ""

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)


    # build the model from a checkpoint file
    model = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume_path)
        load_matched_state_dict(model, checkpoint)
        del checkpoint

    ## no precision conversion needed for torchscript. run at fp3
    dtype = torch.half if args.fp16 else torch.float32
    model.to(dtype).to(args.device)

    imgs_path_train, labels_train = make_img_path(args.file_train, args.file_test, args.input)
    imgs_path_val, labels_val = make_img_path(args.file_test, args.file_test, args.input)
    train_transform = get_transform(is_train=True)
    val_transform = get_transform(is_train=False)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    train_dataset = AdhocImageDataset_che(
        imgs_path_train,
        labels_train,
        args.keypoints_body_path,
        shape=(input_shape[1], input_shape[2]),
        transform=train_transform,
        dataset=args.dataset,
        num_frames=args.num_frames,
    )
    val_dataset = AdhocImageDataset_che(
        imgs_path_val,
        labels_val,
        args.keypoints_body_path,
        shape=(input_shape[1], input_shape[2]),
        transform=val_transform,
        dataset=args.dataset,
        num_frames=args.num_frames,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(min(args.batch_size, cpu_count()), 1),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()), 1),
    )

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay, eps=args.eps, betas=(args.beta1, args.beta2))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=args.minlr)
    scaler = GradScaler()

    best_acc1 = 0.
    best_acc5 = 0.

    for epoch in range(args.epoch):
        model.train()
        losses = AverageMeter()
        losses_main = AverageMeter()
        losses_center = AverageMeter()
        top1_train = AverageMeter()
        top5_train = AverageMeter()
        top1_val = AverageMeter()
        top5_val = AverageMeter()

        for batch_idx, (batch_imgs, keypoints, keypoint_scores, batch_labels) in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            with autocast():
                output = model(batch_imgs.to(dtype).to(args.device), keypoints, keypoint_scores, batch_labels.to(args.device))

            results = output['gloss_logits']
            loss_main = output['loss_main']
            loss_center = output['loss_center']

            loss = loss_main + 0.2 * loss_center
            acc1, acc5 = accuracy(results, batch_labels.to(args.device))
            top1_train.update(acc1[0], BATCH_SIZE)
            top5_train.update(acc5[0], BATCH_SIZE)
            losses.update(loss.item(), BATCH_SIZE)
            losses_main.update(loss_main.item(), BATCH_SIZE)
            losses_center.update(loss_center.item(), BATCH_SIZE)

            loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()

            if (batch_idx+1) % 100 == 0 or (batch_idx + 1) == len(train_dataloader):
                log_stats = {'epoch': epoch, 'loss': losses.avg,
                             'loss_main': losses_main.avg, 'loss_center': losses_center.avg,
                             'acc1_train': float(top1_train.avg.cpu().numpy()),
                             'acc5_train': float(top5_train.avg.cpu().numpy())}
                with open(os.path.join(args.output_root, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                print('epoch: {}, loss: {}, acc1_train: {}, acc5_train{}'\
                    .format(epoch, losses.avg, float(top1_train.avg.cpu().numpy()), float(top5_train.avg.cpu().numpy())))

        scheduler.step()

        model.eval()
        for batch_idx, (batch_imgs, keypoints, keypoint_scores, batch_labels) in tqdm(
                enumerate(val_dataloader), total=len(val_dataloader)
            ):
                with torch.no_grad():
                    output = model(batch_imgs.to(dtype).to(args.device), keypoints, keypoint_scores, batch_labels.to(args.device))

                results = output['gloss_logits']
                acc1, acc5 = accuracy(results, batch_labels.to(args.device))
                top1_val.update(acc1[0], BATCH_SIZE)
                top5_val.update(acc5[0], BATCH_SIZE)

                if (batch_idx+1) % 100 == 0 or (batch_idx + 1) == len(val_dataloader):
                    log_stats = {'epoch': epoch,
                                 'acc1_val': float(top1_val.avg.cpu().numpy()), 'acc5_val': float(top5_val.avg.cpu().numpy()),}
                    with open(os.path.join(args.output_root, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    print('epoch_val: {}, acc1_val: {}, acc5_val: {}' \
                          .format(epoch, float(top1_val.avg.cpu().numpy()), float(top5_val.avg.cpu().numpy())))

        eval_acc = float(top1_val.avg.cpu().numpy()) * 0.9 + float(top5_val.avg.cpu().numpy()) * 0.1
        best_acc = best_acc1 * 0.9 + best_acc5 * 0.1
        if eval_acc > best_acc:
            model_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(model_state, os.path.join(args.output_root, 'best_model.pth'.format(epoch + 1)))
            best_acc1 = float(top1_val.avg.cpu().numpy())
            best_acc5 = float(top5_val.avg.cpu().numpy())
    print('best_acc1: {}, best_acc5: {}'.format(best_acc1, best_acc5))



if __name__ == "__main__":
    os.environ['TORCH_LOGS'] = '+dynamo'
    os.environ['TORCHDYNAMO_VERBOSE'] = '1'

    main()
