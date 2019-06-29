import glob
import os
import os.path as osp
import argparse

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--org-img-path', default='./data/test/IndMya/')
parser.add_argument('--out-img-path', default='./results/VRES/3x')
args = parser.parse_args()

save_img_path = osp.join(args.out_img_path, 'color')
if not osp.exists(save_img_path):
    os.makedirs(save_img_path)
org_imgs = glob.glob(osp.join(args.org_img_path, '*'))
out_imgs = glob.glob(osp.join(args.out_img_path, '*.png'))
org_imgs.sort()
out_imgs.sort()
assert len(org_imgs) == len(out_imgs)

for i in tqdm(range(len(org_imgs))):
    # Get center frame (frame3) of org img
    org_img = glob.glob(osp.join(org_imgs[i], '*f3*'))
    org_img = cv2.imread(org_img[0])
    org_img_ycbcr = cv2.cvtColor(org_img, cv2.COLOR_BGR2YCR_CB)

    # get out img in gray scale
    out_img = cv2.imread(out_imgs[i], 0)

    # merge images
    h, w = out_img.shape
    save_img = org_img_ycbcr[:h, :w, :]
    save_img[:, :, 0] = out_img
    save_img = cv2.cvtColor(save_img, cv2.COLOR_YCrCb2BGR)
    save_img_name = osp.join(save_img_path, osp.basename(out_imgs[i]))
    cv2.imwrite(save_img_name, save_img)
