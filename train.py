import argparse
import os

from SR_datasets import DatasetFactory
from model import ModelFactory
from solver import Solver
from loss import get_loss_fn


description = 'Video Super Resolution pytorch implementation'

parser = argparse.ArgumentParser(description=description)

parser.add_argument('-m', '--model', metavar='M', type=str, default='VRES',
                    help='network architecture. Default VRES')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=3,
                    help='interpolation scale. Default 3')
parser.add_argument('--train-set', metavar='T', type=str, default='train',
                    help='data set for training. Default train')
parser.add_argument('--val-set', metavar='V', type=str, default='test/IndMya',
                    help='data set for validation. Default IndMya')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=100,
                    help='batch size used for training. Default 100')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float,
                    default=1e-3, help='learning rate. Default 1e-3')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=50,
                    help='number of training epochs. Default 100')
parser.add_argument('-f', '--fine-tune', dest='fine_tune', action='store_true',
                    help='fine tune the model under check_point dir,\
                    instead of training from scratch. Default False')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help='print training information. Default False')

args = parser.parse_args()


def get_full_path(scale, train_set):
    """
    Get full path of data based on configs and target path
    example: preprocessed_data/test/set5/3x
    """
    scale_path = str(scale) + 'x'
    return os.path.join('preprocessed_data', train_set, scale_path)


def display_config():
    print('############################################################')
    print('# Video Super Resolution - Pytorch implementation          #')
    print('# by Thang Vu (thangvubk@gmail.com)                        #')
    print('############################################################')
    print('')
    print('-------YOUR SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')


def main():
    display_config()

    train_root = get_full_path(args.scale, args.train_set)
    val_root = get_full_path(args.scale, args.val_set)

    print('Contructing dataset...')
    dataset_factory = DatasetFactory()
    train_dataset = dataset_factory.create_dataset(args.model,
                                                   train_root)
    val_dataset = dataset_factory.create_dataset(args.model,
                                                 val_root)

    model_factory = ModelFactory()
    model = model_factory.create_model(args.model)
    loss_fn = get_loss_fn(model.name)

    check_point = os.path.join(
        'check_point', model.name, str(args.scale) + 'x')

    solver = Solver(
        model, check_point, loss_fn=loss_fn, batch_size=args.batch_size,
        num_epochs=args.num_epochs, learning_rate=args.learning_rate,
        fine_tune=args.fine_tune, verbose=args.verbose)

    print('Training...')
    solver.train(train_dataset, val_dataset)


if __name__ == '__main__':
    main()
