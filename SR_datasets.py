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
#DATASETS = 'SR_dataset, SRCNN_dataset, ESPCN_dataset'


def rgb2ycbcr(rgb):
    return np.dot(rgb[...,:3], [65.738/256, 129.057/256, 25.064/256]) + 16

# default loader
def _gray_loader(path):
    #image = scipy.misc.imread(path, flatten=False, mode='YCbCr')
    image = scipy.misc.imread(path)
    if len(image.shape) == 2:
        return image
    return rgb2ycbcr(image)

# util func
def _get_img_paths(root):
    paths = glob.glob(os.path.join(root, '*.bmp'))
    paths.sort()
    return paths

def mod_crop(image, scale):
    h, w = image.shape[0], image.shape[1]
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    return image[:h, :w]

class DatasetFactory(object):

    def create_dataset(self, name, root, scale=3):
        if name == 'VDCNN':
            return VDCNN_dataset(root)
        elif name == 'VSRCNN':
            return VSRCNN_dataset(root)
        elif name == 'ESPCN':
            return ESPCN_dataset(root)
        elif name == 'VRES':
            return VRES_dataset(root)
        elif name == 'VDSR':
            return VDSR_dataset(root)
        elif name == 'VRNET':
            return VRNET_dataset(root)
        elif name == 'MFCNN':
            return MFCNN_dataset(root)

class SRCNN_dataset(Dataset):
    def __init__(self, root, scale=3, loader=_gray_loader):
        self.loader = loader
        high_res_root = os.path.join(root, 'high_res')
        low_res_root = os.path.join(root, 'low_res')
        self.hs_paths = _get_img_paths(high_res_root)
        self.ls_paths = _get_img_paths(low_res_root)
        self.scale = scale

    def __len__(self):
        return len(self.hs_paths)

    def __getitem__(self, idx):
        high_res = self.loader(self.hs_paths[idx])
        low_res = self.loader(self.ls_paths[idx])
        
        low_res = low_res[:, :, np.newaxis]
        high_res = high_res[:, :, np.newaxis]

        high_res = mod_crop(high_res, 3)
        low_res = mod_crop(low_res, 3)

        # transform np image to torch tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)

        # normalize
        low_res = low_res - 0.5
        high_res = high_res - 0.5
        
        return low_res, high_res

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

class ESPCN_dataset(SRCNN_dataset):
    
    def subpixel_deshuffle(self, img):
        # convert img of shape (S*H, S*W, C) to (H, W, C*S**2)
        SH, SW, C = img.shape
        S = self.scale
        W = SW//S
        H = SH//S

        out = np.zeros((H, W, C*S**2))
        for h in range(H):
            for w in range(W):
                for c in range(C*S**2):
                    out[h, w, c] = img[h*S + c//S%S, w*S + c%S, 0]
        
        return out
    
    def __getitem__(self, idx):
        high_res = self.loader(self.hs_paths[idx])
        low_res = self.loader(self.ls_paths[idx])
        
        low_res = low_res[:, :, np.newaxis]
        high_res = high_res[:, :, np.newaxis]

        high_res = self.subpixel_deshuffle(high_res)

        # transform np image to torch tensor
        transform = T.ToTensor()
        low_res = transform(low_res)
        high_res = transform(high_res)


        low_res = low_res - 0.5
        high_res = high_res - 0.5
        
        return low_res, high_res










        
        
        



        
        
    
        
