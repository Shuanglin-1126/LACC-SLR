import torch
from PIL import Image, ImageOps
import random
import numpy as np
import torchvision
from dataset.rand_augment import rand_augment_transform
from dataset.random_erasing import Random_Erasing


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BICUBIC


    def __call__(self, img_pose):
        img_group, keypoint, keypoint_scores = img_pose
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in
                         crop_img_group]

        offest = np.array([offset_h, offset_w])
        keypoint -= offest
        keypoint_scores[(keypoint[:, :, 0] >= crop_h) | (keypoint[:, :, 1] >= crop_w) | (keypoint[:, :, 0] <= 0.) | (
                    keypoint[:, :, 1] <= 0.)] = 0.
        keypoint[:, :, 0] = torch.clamp(keypoint[:, :, 0], min=0.0, max=crop_h) / crop_h * self.input_size[0]
        keypoint[:, :, 1] = torch.clamp(keypoint[:, :, 1], min=0.0, max=crop_w) / crop_w * self.input_size[1]
        keypoint_scores[keypoint_scores <= 0.3] = 0.

        return ret_img_group, keypoint, keypoint_scores

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        return ret


class Stack(object):

    def __init__(self, roll=True):
        self.roll = roll

    def __call__(self, img_pose):
        img_tuple, keypoint, keypoint_scores = img_pose
        img_group = img_tuple


        if self.roll:
            return np.concatenate([np.array(x).transpose(2, 0, 1) for x in img_group], axis=0), keypoint, keypoint_scores
        else:
            return np.concatenate(img_group, axis=0), keypoint, keypoint_scores


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, img_pose):
        pic, keypoint, keypoint_scores = img_pose

        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).contiguous()
        return img.float().div(255.) if self.div else img.float(), keypoint, keypoint_scores


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, shape, is_flow=False):
        self.is_flow = is_flow
        self.shape = shape

    def __call__(self, img_pose):
        img_group, keypoint, keypoint_scores = img_pose
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            keypoint[:, :, 1] = self.shape - keypoint[:, :, 1]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret, keypoint, keypoint_scores
        else:
            return img_group, keypoint, keypoint_scores


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img_pose):
        tensor, keypoint, keypoint_scores = img_pose
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor, keypoint, keypoint_scores


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_pose):
        img_group, keypoint, keypoint_scores = img_pose

        return [self.worker(img) for img in img_group], keypoint, keypoint_scores


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_pose):
        img_group, keypoint, keypoint_scores = img_pose
        w, h = img_group[0].size
        keypoint[..., 0] = keypoint[..., 0] / h * 224
        keypoint[..., 1] = keypoint[..., 1] / w * 224
        keypoint_scores[(keypoint[:, :, 0] >= 224) | (keypoint[:, :, 1] >= 224) | (keypoint[:, :, 0] <= 0.) | (
                    keypoint[:, :, 1] <= 0.)] = 0.
        keypoint = torch.clamp(keypoint, min=0.0, max=224)
        return [self.worker(img) for img in img_group], keypoint, keypoint_scores


class ColorJitter(object):
    def __init__(self):
        self.color = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

    def __call__(self, img_group):
        if random.random() < 0.5:
            ret = [self.color(img) for img in img_group]
            return ret
        else:
            return img_group

class RandomErasing(object):
    def __init__(self, prob=0.25):
        # self.randera = torchvision.transforms.RandomErasing(p=prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
        self.randera = RandomErasing(prob)
    def __call__(self, img_group):
        N, H, W = img_group.shape
        img_group = self.randera(img_group.view(-1, 3, H, W)).flatten(0, 1)
        return img_group

class RandAugment(object):
    def __init__(self, input_size, auto_augment=None, interpolation="bilinear"):
        self.randomaug = self.create_random_augment(input_size, auto_augment, interpolation)

    def create_random_augment(
            self,
            input_size,
            auto_augment=None,
            interpolation="bilinear",
    ):
        """
        Get video randaug transform.

        Args:
            input_size: The size of the input video in tuple.
            auto_augment: Parameters for randaug. An example:
                "rand-m7-n4-mstd0.5-inc1" (m is the magnitude and n is the number
                of operations to apply).
            interpolation: Interpolation method.
        """
        if isinstance(input_size, tuple):
            img_size = input_size[-2:]
        else:
            img_size = input_size

        if auto_augment:
            assert isinstance(auto_augment, str)
            if isinstance(img_size, tuple):
                img_size_min = min(img_size)
            else:
                img_size_min = img_size
            aa_params = {"translate_const": int(img_size_min * 0.45)}
            if interpolation and interpolation != "random":
                aa_params["interpolation"] = Image.BICUBIC
            if auto_augment.startswith("rand"):
                return rand_augment_transform(auto_augment, aa_params)
        raise NotImplementedError

    def __call__(self, img_group):
        img_group = [self.randomaug(img) for img in img_group]
        return img_group

class ResizeWithPadding:
    def __init__(self, target_size, fill=0):
        self.target_size = target_size  # (width, height)
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # 填充
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img = ImageOps.expand(img, padding, fill=self.fill)
        return img
