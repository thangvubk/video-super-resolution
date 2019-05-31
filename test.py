import argparse
import os

import cv2
import numpy as np
from SR_datasets import DatasetFactory
from model import ModelFactory
from solver import Solver


description = 'Video Super Resolution pytorch implementation'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-m', '--model', metavar='M', type=str, default='VRES',
                    help='network architecture. Default False')
parser.add_argument('--model_path',
                    default='./check_point/VRES/3x/best_model.pt')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=3,
                    help='interpolation scale. Default 3')
parser.add_argument('--test-set', metavar='NAME', type=str, default='IndMya',
                    help='dataset for testing. Default IndMya')
args = parser.parse_args()


def get_full_path(scale, test_set):
    """
    Get full path of data based on configs and target path
    example: preprocessed_data/test/set5/3x
    """
    scale_path = str(scale) + 'x'
    return os.path.join('preprocessed_data/test', test_set, scale_path)


def display_config():
    print('############################################################')
    print('# Video Super Resolution - Pytorch implementation          #')
    print('# by Thang Vu (thangvubk@gmail.com                         #')
    print('############################################################')
    print('')
    print('-------YOUR SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')


def export(scale, model_name, stats, outputs):
    path = os.path.join('results', model_name, str(scale) + 'x')

    if not os.path.exists(path):
        os.makedirs(path)

    for i, img in enumerate(outputs):
        img_name = os.path.join(path, model_name + '_output%03d.png' % i)
        cv2.imwrite(img_name, img)

    with open(os.path.join(path, model_name + '.txt'), 'w') as f:
        psnrs, ssims, proc_time = stats
        f.write('\t\tPSNR\tSSIM\tTime\n')
        for i in range(len(psnrs)):
            print('Img%d: PSNR: %.3f SSIM: %.3f Time: %.4f'
                  % (i, psnrs[i], ssims[i], proc_time[i]))
            f.write('Img%d:\t%.3f\t%.3f\t%.4f\n'
                    % (i, psnrs[i], ssims[i], proc_time[i]))
    print('Average test psnr: %.3fdB' % np.mean(psnrs))
    print('Average test ssim: %.3f' % np.mean(ssims))
    print('Finish!!!')


def main():
    display_config()

    dataset_root = get_full_path(args.scale, args.test_set)

    print('Contructing dataset...')
    dataset_factory = DatasetFactory()
    train_dataset = dataset_factory.create_dataset(args.model,
                                                   dataset_root)

    model_factory = ModelFactory()
    model = model_factory.create_model(args.model)

    check_point = os.path.join(
        'check_point', model.name, str(args.scale) + 'x')
    solver = Solver(model, check_point)

    print('Testing...')
    stats, outputs = solver.test(train_dataset, args.model_path)
    export(args.scale, model.name, stats, outputs)


if __name__ == '__main__':
    main()
