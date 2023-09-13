import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import os
import os.path
import cv2
import torchvision

def make_dataset(image_list, labels):
    len_ = len(image_list)
    if labels:
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], int(val.split()[1]), int(val.split()[2])) for val in image_list]
        labels = [int(val.split()[1]) for val in image_list]
        sensitives = [int(val.split()[2]) for val in image_list]
    return images, labels, sensitives


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs, self.labels, self.sensitives = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs

        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target, sensitive = self.imgs[index]
        img = self.loader(os.path.join('./data/cardiomegaly/', path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, sensitive

    def __len__(self):
        return len(self.imgs)

    def group_counts(self, resample_which, flag=''):
        if resample_which == 'group' or resample_which == 'balanced' or resample_which == 'natural':
            group_array = self.sensitives
            if resample_which == 'balanced' or resample_which == 'natural':
                labels = self.labels
                num_labels = len(set(labels))
                num_groups = len(set(group_array))
                group_array = (np.asarray(group_array) * num_labels + np.asarray(labels)).tolist()
        elif resample_which == 'class':
            group_array = self.labels
            num_labels = len(set(group_array))

        self._group_array = torch.LongTensor(group_array)
        if resample_which == 'group':
            self._group_counts = (torch.arange(len(set(group_array))).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'balanced':
            self._group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'class':
            self._group_counts = (torch.arange(num_labels).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'natural':
            self._group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1)==self._group_array).sum(1).float()
            self._group_counts = torch.ones_like(self._group_counts)

        if 'Younger' in flag and resample_which in ['group', 'balanced', 'natural']:
            self._group_counts[-1] *= 10
            self._group_counts[-2] *= 10
            if resample_which == 'balanced' or resample_which == 'natural':
                self._group_counts[-3] *= 10
                self._group_counts[-4] *= 10
        if 'Older' in flag and resample_which in ['group', 'balanced', 'natural']:
            self._group_counts[-1] /= 10
            self._group_counts[-2] /= 10
            if resample_which == 'balanced' or resample_which == 'natural':
                self._group_counts[-3] /= 10
                self._group_counts[-4] /= 10

        return group_array, self._group_counts

    def get_weights(self, resample_which, flag=''):
        sens_attr, group_num = self.group_counts(resample_which, flag)
        group_weights = [1/max(1, x.item()) for x in group_num]
        sample_weights = [group_weights[int(i)] for i in sens_attr]
        return sample_weights


class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs, self.labels, self.sensitives = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target, sensitive = self.imgs[index]
        img = self.loader(os.path.join('./data/cardiomegaly/', path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, sensitive

    def __len__(self):
        return len(self.imgs)

    def group_counts(self, resample_which, flag=''):
        if resample_which == 'group' or resample_which == 'balanced' or resample_which == 'natural':
            group_array = self.sensitives
            if resample_which == 'balanced' or resample_which == 'natural':
                labels = self.labels
                num_labels = len(set(labels))
                num_groups = len(set(group_array))
                group_array = (np.asarray(group_array) * num_labels + np.asarray(labels)).tolist()
        elif resample_which == 'class':
            group_array = self.labels
            num_labels = len(set(group_array))

        self._group_array = torch.LongTensor(group_array)
        if resample_which == 'group':
            self._group_counts = (torch.arange(len(set(group_array))).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'balanced':
            self._group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'class':
            self._group_counts = (torch.arange(num_labels).unsqueeze(1)==self._group_array).sum(1).float()
        elif resample_which == 'natural':
            self._group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1)==self._group_array).sum(1).float()
            self._group_counts = torch.ones_like(self._group_counts)

        if 'Younger' in flag and resample_which in ['group', 'balanced', 'natural']:
            self._group_counts[-1] *= 10
            self._group_counts[-2] *= 10
            if resample_which == 'balanced' or resample_which == 'natural':
                self._group_counts[-3] *= 10
                self._group_counts[-4] *= 10
        if 'Older' in flag and resample_which in ['group', 'balanced', 'natural']:
            self._group_counts[-1] /= 10
            self._group_counts[-2] /= 10
            if resample_which == 'balanced' or resample_which == 'natural':
                self._group_counts[-3] /= 10
                self._group_counts[-4] /= 10

        return group_array, self._group_counts

    def get_weights(self, resample_which, flag=''):
        sens_attr, group_num = self.group_counts(resample_which, flag)
        group_weights = [1/max(1, x.item()) for x in group_num]
        sample_weights = [group_weights[int(i)] for i in sens_attr]
        return sample_weights
