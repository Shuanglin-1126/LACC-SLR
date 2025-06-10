import numpy as np
import os
from multiprocessing import cpu_count
import torch
import torchvision
from dataset.image_pose_dataset import AdhocImageDataset_che
from tqdm import tqdm
from config.test_config import get_args
from dataset.img_pose_transform import Stack, ToTorchFormatTensor, GroupNormalize, GroupScale, GroupCenterCrop
from models.build_model import build_model

torchvision.disable_beta_transforms_warning()


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
        input = os.path.join(input, 'train_val')
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

def accuracy(output, target, num_class=100, topk=(1, 5)):
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

def get_transform(is_train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    augments = []
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
    checkpoint = torch.load(args.resume_path)
    load_matched_state_dict(model, checkpoint)
    del checkpoint

    ## no precision conversion needed for torchscript. run at fp3
    dtype = torch.half if args.fp16 else torch.float32
    model.to(dtype).to(args.device)

    imgs_path_val, labels_val = make_img_path(args.file_test, args.input)
    val_transform = get_transform(is_train=False)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    val_dataset = AdhocImageDataset_che(
        imgs_path_val,
        labels_val,
        args.keypoints_body_path,
        shape=(input_shape[1], input_shape[2]),
        transform=val_transform,
        num_frames=args.num_frames,
        is_train=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()), 1),
    )

    cls_num = args.num_class
    model.eval()

    correct = correct_5 = num_samples = 0
    top1_t = np.zeros(cls_num, dtype=np.int32)
    top1_f = np.zeros(cls_num, dtype=np.int32)
    top5_t = np.zeros(cls_num, dtype=np.int32)
    top5_f = np.zeros(cls_num, dtype=np.int32)

    with torch.no_grad():
        for batch_idx, (batch_imgs, keypoints, keypoint_scores, batch_labels) in tqdm(
                enumerate(val_dataloader), total=len(val_dataloader)
            ):
            results=[]
            for i in range(batch_imgs.shape[1]):
                output = model(batch_imgs[:, i, ...].to(dtype).to(args.device), keypoints[:, i, ...],
                               keypoint_scores[:, i, ...], batch_labels.to(args.device))
                results.append(output['gloss_logits'])
                # results.append(output)
                del output
            results = torch.stack(results, dim=0)

            results = torch.sum(results, dim=0).squeeze()
            _, pred = results.topk(5, 1, True, True)

            for i in range(pred.shape[0]):
                if batch_labels[i] in pred[i, :, 0]:
                    correct += 1
                    top1_t[batch_labels[i]] += 1
                else:
                    top1_f[batch_labels[i]] += 1
                if batch_labels[i] in pred[i, :, :]:
                    correct_5 += 1
                    top5_t[batch_labels[i]] += 1
                else:
                    top5_f[batch_labels[i]] += 1
                num_samples += 1

            del results, pred


    print('per-instance: best_acc1: {}, best_acc5: {}\n'.format(correct/num_samples, correct_5/num_samples))
    print('per-class: best_acc1: {}, best_acc5: {}'.format(np.nanmean(top1_t/(top1_t+top1_f)), np.nanmean(top5_t/(top5_t+top5_f))))


if __name__ == "__main__":
    os.environ['TORCH_LOGS'] = '+dynamo'
    os.environ['TORCHDYNAMO_VERBOSE'] = '1'

    main()
