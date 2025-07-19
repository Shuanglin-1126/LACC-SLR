#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""

import math

from fvcore.common.config import CfgNode

from slr.models.MViT import custom_config

_C = CfgNode()

_C.device = "cuda:0"

_C.optim = 'adamw'
_C.momentum = 0.9
_C.eps = 1e-8
_C.lr = 1e-4
_C.minlr = 1e-6
_C.beta1 = 0.9
_C.beta2 = 0.999
_C.weight_decay = 5e-2
_C.epoch = 60
_C.accumulation_steps = 1
_C.warm_epoch = 0
_C.warm_lr = 1e-3
_C.save_model_fre = 20
_C.batch_size = 32

_C.emd_dim = 768
_C.drop_out = 0.1
_C.backbone_net = 'mvit_s'
_C.fused_posi = 1
_C.fp16 = False

_C.shape = [224, 224]
_C.output_root = r"E:\chexiao\projects\output\output_mobilenet"

_C.file_train = r'D:\SLR_dataset\WLASL\origin_frames\wlasl100train_all_frame.txt'
_C.file_val = r'D:\SLR_dataset\WLASL\origin_frames\wlasl100val_all_frame.txt'
_C.file_test = r'D:\SLR_dataset\WLASL\origin_frames\wlasl100test_all_frame.txt'
_C.input = r"D:\SLR_dataset\WLASL\origin_frames"
_C.keypoints_body_path = r"E:\SLR_dataset\wlasl\body_keypoint"
_C.dataset = 'wlasl100'
_C.datadir = r'D:\SLR_dataset\WLASL\origin_frames'
_C.num_frames = 16
_C.frames_per_group = 1
_C.num_clips = 1
_C.num_class = 100
_C.modality = 'rgb'
_C.dense_sampling = False
_C.resume = False
_C.resume_path = r'E:\chexiao\projects\output1\output_mvits_pose_100\best_model.pth'


# Add custom config with default values.
custom_config.add_custom_config(_C)



def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
