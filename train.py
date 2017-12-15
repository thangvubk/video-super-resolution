import argparse
import sys
import os
from SR_datasets import DatasetFactory
from model import ModelFactory
from solver import Solver
from exporter import Exporter
description='Video Super Resolution pytorch implementation'

parser = argparse.ArgumentParser(description=description)

parser.add_argument('-m', '--model', metavar='M', type=str, default='VRES',
                    help='network architecture')
parser.add_argument('-c', '--scale', metavar='S', type=int, default=3, 
                    help='interpolation scale')
parser.add_argument('--train-set', metavar='T', type=str, default='train',
                    help='data set for training')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=32,
                    help='batch size used for training')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float, default=1e-4,
                    help='learning rate used for training')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=50,
                    help='number of training epochs')
parser.add_argument('-f', '--fine-tune', dest='fine_tune', action='store_true',
                    help='fine tune the model under check_point dir,\
                    instead of training from scratch')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help='print training information')

args = parser.parse_args()

def get_full_path(scale, train_set):
    """
    Get full path of data based on configs and target path
    example: data/interpolation/test/set5/3x
    """
    scale_path = str(scale) + 'x'
    return os.path.join('preprocessed_data', train_set, scale_path)
    
def display_config():
    print('############################################################')
    print('# Video Super Resolution - Pytorch implementation          #')
    print('# by Thang Vu (thangvubk@gmail.com                         #')
    print('############################################################')
    print('')
    print('-------YOUR SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" %(str(arg), str(getattr(args, arg))))
    print('')


def main():
    display_config()

    if args.model == 'VSRCNN' or 'VDCNN' or 'VRES':
        is_using_interp = True
    else:
        is_using_interp = False

    dataset_root = get_full_path(args.scale, args.train_set)

    print('Contructing dataset...')
    dataset_factory = DatasetFactory()
    train_dataset  = dataset_factory.create_dataset(args.model,
                                                    dataset_root)

    model_factory = ModelFactory()
    model = model_factory.create_model(args.model)
    
    check_point = os.path.join('check_point', model.name, str(args.scale) + 'x')
    solver = Solver(model, check_point, batch_size=args.batch_size,
                    num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                    fine_tune=args.fine_tune, verbose=args.verbose)

    print('Training...')
    solver.train(train_dataset)

if __name__ == '__main__':
    main()

