from __future__ import division
import torchvision
import torchvision.transforms as T
import os
import glob
import scipy.misc
import scipy.ndimage
import numpy as np
import h5py
import torch

import config
from torch.utils.data import Dataset

class DatasetFactory(object):

    def create_dataset(self, name, root, scale=3):
        if name == 'VDCNN':
            return VDCNN_dataset(root)
        elif name == 'VSRCNN':
            return VSRCNN_dataset(root)
        elif name == 'VRES':
            return VRES_dataset(root)
        elif name == 'VDSR':
            return VDSR_dataset(root)
        elif name == 'VRNET':
            return VRNET_dataset(root)
        elif name == 'MFCNN':
            return MFCNN_dataset(root)
        else:
            raise Exception('Unknown dataset {}'.format(name))

class VRES_dataset(Dataset):
    def __init__(self, root):
        
        root = os.path.join(root, 'dataset.h5')
        f = h5py.File(root)
        self.low_res_imgs = f.get('data')
        self.high_res_imgs = f.get('label')

        self.low_res_imgs = np.array(self.low_res_imgs)
        self.high_res_imgs = np.array(self.high_res_imgs)

    def __len__(self):
        return self.high_res_imgs.shape[0]

    def __getitem__(self, idx):
        center = 2

        low_res_imgs = self.low_res_imgs[idx]
        high_res_imgs = self.high_res_imgs[idx]
        
        # h5 in matlab is (H, W, C)
        # h5 in python is (C, W, H)
        # we need to transpose to (C, H, W)
        low_res_imgs = low_res_imgs.transpose(0, 2, 1)
        high_res_imgs = high_res_imgs.transpose(0, 2, 1)

        high_res_img = high_res_imgs[center]
        high_res_img = high_res_img[np.newaxis, :, :]

        low_res_imgs -= 0.5
        high_res_img -= 0.5

        # transform np image to torch tensor
        low_res_imgs = torch.Tensor(low_res_imgs)
        high_res_img = torch.Tensor(high_res_img)

        return low_res_imgs, high_res_img

class VSRCNN_dataset(VRES_dataset):
    def __getitem__(self, idx):
        center = 2
        low_res_imgs = self.low_res_imgs[idx]
        high_res_imgs = self.high_res_imgs[idx]
        
        # h5 in matlab is (H, W, C)
        # h5 in python is (C, W, H)
        # we need to transpose to (C, H, W)
        low_res_imgs = low_res_imgs.transpose(0, 2, 1)
        high_res_imgs = high_res_imgs.transpose(0, 2, 1)

        low_res_img = low_res_imgs[center]
        high_res_img = high_res_imgs[center]

        low_res_img = low_res_img[np.newaxis, :, :]
        high_res_img = high_res_img[np.newaxis, :, :]

        low_res_img -= 0.5
        high_res_img -= 0.5

        # transform np image to torch tensor
        low_res_img = torch.Tensor(low_res_img)
        high_res_img = torch.Tensor(high_res_img)
        

        return low_res_img, high_res_img

class VDSR_dataset(VSRCNN_dataset):
    pass

class VDCNN_dataset(VSRCNN_dataset):
    pass

class VRNET_dataset(VRES_dataset):
    pass

class MFCNN_dataset(VRES_dataset):
    pass


