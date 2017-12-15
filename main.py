import argparse
import sys
import os
from SR_datasets import DatasetFactory
from model import ModelFactory
from solver import Solver
from exporter import Exporter
description='SRCNN-pytorch implementation'

parser = argparse.ArgumentParser(description=description)


parser.add_argument('phase', metavar='PHASE', type=str,
                    help='train or test or both')
parser.add_argument('-m', '--model', metavar='M', type=str, default='SRCNN_proposed',
                    help='network architecture')
parser.add_argument('-c', '--scale', metavar='S', type=int, default=3, 
                    help='interpolation scale')
parser.add_argument('--train-path', metavar='PATH', type=str, default='train',
                    help='accompanied with other config (scale, interp)\
                    to create full path of train data')
parser.add_argument('--test-path', metavar='PATH', type=str, default='test',
                    help='accompanied with other configs (scale, interp)\
                    to create full path of test data')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=32,
                    help='batch size used for training')
parser.add_argument('-l', '--learning-rate', metavar='L', type=float, default=1e-4,
                    help='learning rate used for training')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=50,
                    help='number of training epochs')
parser.add_argument('-f', '--fine-tune', dest='fine_tune', action='store_true',
                    help='fine tune the model under TrainedModel dir,\
                    instead of training from scratch')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help='print training information')

args = parser.parse_args()

if args.phase not in ['train', 'test']:
    print('ERROR!!!')
    print('"Phase" must be "train" or "test"')
    print('')
    parser.print_help()
    sys.exit(1)

def get_full_path(scale, is_using_interp, target_path):
    """
    Get full path of data based on configs and target path
    example: data/interpolation/test/set5/3x
    """

    scale_path = str(scale) + 'x'
    if is_using_interp:
        interp_path = 'interpolation'
    else:
        interp_path = 'noninterpolation'
    return os.path.join('preprocessed_data_video', target_path, interp_path, scale_path)
    
def display_config():
    print('############################################################')
    print('# Image/Video Super Resolution - Pytorch implementation    #')
    print('# by Thang Vu                                              #')
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

    # get full path bases on config and target path
    root_train = get_full_path(args.scale, is_using_interp, args.train_path)
    root_test = get_full_path(args.scale, is_using_interp, args.test_path)

    print('Contructing dataset...')
    dataset_roots = root_train, root_test
    dataset_factory = DatasetFactory()
    train_dataset, test_dataset = dataset_factory.create_dataset(args.model,
                                                                 dataset_roots)

    model_factory = ModelFactory()
    model = model_factory.create_model(args.model)
    
    solver = Solver(model, batch_size=args.batch_size,
                    num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                    fine_tune=args.fine_tune, verbose=args.verbose)

    if args.phase == 'train':
        print('Training...')
        solver.train(train_dataset)
    elif args.phase == 'test':
        print('Testing...')
        psnrs, outputs = solver.test(test_dataset)
        exporter = Exporter(model.name, args.scale)
        exporter.export(psnrs, outputs)

if __name__ == '__main__':
    main()

