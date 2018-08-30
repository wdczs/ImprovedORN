import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import random
import cv2



class CLSDataPrepare(Dataset):
    def __init__(self, txt_path, img_transform = None):

        self.img_list = []
        self.label_list = []

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.img_list.append(line.split()[0])
                self.label_list.append(line.split()[1])

        self.img_transform = img_transform

    def __getitem__(self, index):
        im, gt = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = cv2.imread(img_path)

        #################### RandomRotation #######################
        # N_45 = int(random.random()/0.125)
        # if N_45 != 0:
        #     img = rotate(img, N_45 * 45)
        #
        # img = img.astype(np.float32)
        ###########################################################

        if self.img_transform is not None:
            img = self.img_transform(img)
        else:
            img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        # to rgb
        img = img[(2, 1, 0), :, :]
        label = int(label)

        return img, label


    def __len__(self):
        return len(self.label_list)

def classifier_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    labels = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])

    return torch.stack(imgs, 0), torch.LongTensor(labels)

def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated