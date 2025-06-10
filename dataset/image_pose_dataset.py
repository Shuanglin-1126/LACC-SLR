# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
from PIL import Image
import numpy as np

class AdhocImageDataset_che(torch.utils.data.Dataset):
    def __init__(self, image_list, labels, keypoints_body_path, shape=None, transform=None, dataset='wlasl', num_frames=16, crops=3, is_train=True):
        self.image_list = image_list
        self.labels = labels
        if shape:
            assert len(shape) == 2
        self.shape = shape
        self.keypoints_body_path = keypoints_body_path
        self.transform = transform
        self.num_frames = num_frames
        self.num_groups = num_frames
        self.frames_per_group = 1
        self.temporal_jitter = None
        self.num_consecutive_frames = 1
        self.crops = crops
        self.is_train = is_train
        self.dataset = dataset

    def __len__(self):
        return len(self.image_list)


    def _sample_train_indices(self, total_frames):
        skip = total_frames // self.num_frames
        if skip == 0:
            frame_id_list_1 = np.arange(total_frames)
            res_id_list = np.ones(self.num_frames-total_frames) * (total_frames - 1)
            frame_id_list_1 = np.concatenate([frame_id_list_1, res_id_list], axis=0)

        else:
            start_idx = int(np.random.randint(total_frames - (self.num_frames - 1) * skip, size=1))
            frame_id_list_1 = np.arange(start_idx, start_idx + skip * self.num_frames, skip)

        num_frames_pose = int(self.num_frames * 2)
        skip = total_frames // (num_frames_pose)
        if skip == 0:
            frame_id_list_2 = np.arange(total_frames)
            res_id_list = np.ones(num_frames_pose - total_frames) * (total_frames - 1)
            frame_id_list_2 = np.concatenate([frame_id_list_2, res_id_list], axis=0)

        else:
            start_idx = int(np.random.randint(total_frames - (num_frames_pose - 1) * skip, size=1))
            frame_id_list_2 = np.arange(start_idx, start_idx + skip * num_frames_pose, skip)

        return frame_id_list_1.astype('int32') + 1, frame_id_list_2.astype('int32') + 1

    def _sample_train_indices_test(self, total_frames):
        skip = total_frames // self.num_frames
        frame_id_list_rgb = np.empty(self.crops * self.num_frames)
        frame_id_list_pose = np.empty(self.crops * self.num_frames * 2)
        if skip == 0:
            frame_id_list_1 = np.arange(total_frames)
            res_id_list = np.ones(self.num_frames-total_frames) * (total_frames - 1)
            frame_id_list_1 = np.concatenate([frame_id_list_1, res_id_list], axis=0)
            for i in range(self.crops):
                frame_id_list_rgb[i*self.num_frames:(i+1)*self.num_frames] = frame_id_list_1

        else:
            for i in range(self.crops):
                start_idx = int(np.random.randint(total_frames - (self.num_frames - 1) * skip, size=1))
                frame_id_list_1 = np.arange(start_idx, start_idx + skip * self.num_frames, skip)
                frame_id_list_rgb[i*self.num_frames:(i+1)*self.num_frames] = frame_id_list_1

        num_frames_pose = int(self.num_frames * 2)
        skip = total_frames // (num_frames_pose)
        if skip == 0:
            frame_id_list_2 = np.arange(total_frames)
            res_id_list = np.ones(num_frames_pose - total_frames) * (total_frames - 1)
            frame_id_list_2 = np.concatenate([frame_id_list_2, res_id_list], axis=0)
            for i in range(self.crops):
                frame_id_list_pose[i*num_frames_pose:(i+1)*num_frames_pose] = frame_id_list_2

        else:
            for i in range(self.crops):
                start_idx = int(np.random.randint(total_frames - (num_frames_pose - 1) * skip, size=1))
                frame_id_list_2 = np.arange(start_idx, start_idx + skip * num_frames_pose, skip)
                frame_id_list_pose[i*num_frames_pose:(i+1)*num_frames_pose] = frame_id_list_2

        return frame_id_list_rgb.astype('int32') + 1, frame_id_list_pose.astype('int32') + 1

    def _video_TSN_decord_batch_loader(self, directory, frame_id_list):
        sampled_list = []

        for _, path in enumerate(frame_id_list):
            if 'wlasl' in self.dataset:
                img_path = os.path.join(directory, '{:04d}.jpg'.format(path))
            else:
                img_path = os.path.join(directory, 'image_{:05d}.jpg'.format(path))
            img = Image.open(img_path).convert('RGB')
            sampled_list.append(img)

        return sampled_list

    def _pose_load(self, directory, frame_id_list):
        frame_id_list -= 1
        keypoint_data = np.load(os.path.join(self.keypoints_body_path, directory[-5:] + '.npz'))
        keypoints = keypoint_data['keypoint_vedio'][frame_id_list, :]
        keypoint_scores = keypoint_data['keypoint_score_vedio'][frame_id_list]

        return torch.tensor(keypoints), torch.tensor(keypoint_scores)

    def __getitem__(self, idx):
        target = self.labels[idx]
        directory = self.image_list[idx]

        # 获取目录下的所有文件和目录名
        entries = os.listdir(directory)
        # 计算文件数量
        file_count = sum(os.path.isfile(os.path.join(directory, entry)) for entry in entries)

        if self.is_train:
            frame_id_list_1, frame_id_list_2 = self._sample_train_indices(file_count)
            images_ori = self._video_TSN_decord_batch_loader(directory, frame_id_list_1)
            keypoints, keypoint_scores = self._pose_load(directory, frame_id_list_2)
            keypoints = torch.flip(keypoints, dims=[-1])
            images, keypoints, keypoint_scores = self.transform((images_ori, keypoints, keypoint_scores))
            images = images.view((self.num_frames, 3) + images.size()[-2:])
        else:
            frame_id_list_1, frame_id_list_2 = self._sample_train_indices_test(file_count)
            images_ori = self._video_TSN_decord_batch_loader(directory, frame_id_list_1)
            keypoints, keypoint_scores = self._pose_load(directory, frame_id_list_2)
            keypoints = torch.flip(keypoints, dims=[-1])
            images, keypoints, keypoint_scores = self.transform((images_ori, keypoints, keypoint_scores))
            images = images.view((-1, 3) + images.size()[-2:])
            images = images.view((self.crops, self.num_frames) + images.size()[-3:])
            keypoints = keypoints.view((self.crops, -1) + keypoints.size()[-2:])
            keypoint_scores = keypoint_scores.view((self.crops, -1) + keypoint_scores.size()[-1:])

        return images, keypoints, keypoint_scores, target

