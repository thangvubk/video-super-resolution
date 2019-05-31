from __future__ import division
import os
import time
from shutil import copyfile

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import math
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_ssim


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training super resolution
    The Solver accepts both training and validation data label so it can
    periodically check the PSNR on training

    To train a model, you will first construct a Solver instance,
    pass the model, datasets, and various option (optimizer, loss_fn,
    batch_size, etc) to the constructor.

    After train() method is called. The best model is saved into
    'check_point' dir, which is used for the testing time.

    """
    def __init__(self, model, check_point, **kwargs):
        """
        Construct a new Solver instance

        Required arguments
        - model: a torch nn module describe the neural network architecture
        - check_point: save trained model for testing for finetuning

        Optional arguments:
        - num_epochs: number of epochs to run during training
        - batch_size: batch size for train phase
        - optimizer: update rule for model parameters
        - loss_fn: loss function for the model
        - fine_tune: fine tune the model in check_point dir instead of training
                     from scratch
        - verbose: print training information
        - print_every: period of statistics printing
        """
        self.model = model
        self.check_point = check_point
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 128)
        self.learning_rate = kwargs.pop('learning_rate', 1e-4)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate, weight_decay=1e-6)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5)
        self.loss_fn = kwargs.pop('loss_fn', nn.MSELoss())
        self.fine_tune = kwargs.pop('fine_tune', False)
        self.verbose = kwargs.pop('verbose', False)
        self.print_every = kwargs.pop('print_every', 10)

        self._reset()

    def _reset(self):
        """ Initialize some book-keeping variable, dont call it manually"""
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model = self.model.cuda()

    def _epoch_step(self, dataset, epoch):
        """ Perform 1 training 'epoch' on the 'dataset'"""
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=4)

        num_batchs = len(dataset)//self.batch_size

        running_loss = 0
        for i, (input_batch, label_batch) in enumerate(tqdm(dataloader)):

            # Wrap with torch Variable
            input_batch, label_batch = self._wrap_variable(input_batch,
                                                           label_batch,
                                                           self.use_gpu)

            # zero the grad
            self.optimizer.zero_grad()

            # Forward
            output_batch = self.model(input_batch)
            loss = self.loss_fn(output_batch, label_batch)

            running_loss += loss.item()

            # Backward + update
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)
            self.optimizer.step()

        average_loss = running_loss/num_batchs
        if self.verbose:
            print('Epoch  %5d, loss %.5f' % (epoch, average_loss))

    def _wrap_variable(self, input_batch, label_batch, use_gpu):
        if use_gpu:
            input_batch, label_batch = (Variable(input_batch.cuda()),
                                        Variable(label_batch.cuda()))
        else:
            input_batch, label_batch = (Variable(input_batch),
                                        Variable(label_batch))
        return input_batch, label_batch

    def _comput_PSNR(self, imgs1, imgs2):
        """Compute PSNR between two image array and return the psnr sum"""
        N = imgs1.size()[0]
        imdiff = imgs1 - imgs2
        imdiff = imdiff.view(N, -1)
        rmse = torch.sqrt(torch.mean(imdiff**2, dim=1))
        psnr = 20*torch.log(255/rmse)/math.log(10)  # psnr = 20*log10(255/rmse)
        psnr = torch.sum(psnr)
        return psnr

    def _check_PSNR(self, dataset, is_test=False):
        """
        Get the output of model with the input being 'dataset' then
        compute the PSNR between output and label.

        if 'is_test' is True, psnr and output of each image is also
        return for statistics and generate output image at test phase
        """

        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=4)

        avr_psnr = 0
        avr_ssim = 0

        # book keeping variables for test phase
        psnrs = []  # psnr for each image
        ssims = []  # ssim for each image
        proc_time = []   # processing time
        outputs = []     # output for each image

        for batch, (input_batch, label_batch) in enumerate(dataloader):
            input_batch, label_batch = self._wrap_variable(input_batch,
                                                           label_batch,
                                                           self.use_gpu)
            if is_test:
                start = time.time()
                output_batch = self.model(input_batch)
                elapsed_time = time.time() - start
            else:
                output_batch = self.model(input_batch)

            # ssim is calculated with the normalize (range [0, 1]) image
            ssim = pytorch_ssim.ssim(
                output_batch + 0.5, label_batch + 0.5, size_average=False)
            ssim = torch.sum(ssim).item()
            avr_ssim += ssim

            # calculate PSRN
            output = output_batch.data
            label = label_batch.data

            output = (output + 0.5)*255
            label = (label + 0.5)*255

            output = output.squeeze(dim=1)
            label = label.squeeze(dim=1)

            psnr = self._comput_PSNR(output, label)
            psnr = psnr.item()
            avr_psnr += psnr

            # save psnrs and outputs for stats and generate image at test time
            if is_test:
                psnrs.append(psnr)
                ssims.append(ssim)
                proc_time.append(elapsed_time)
                np_output = output.cpu().numpy()
                outputs.append(np_output[0])

        epoch_size = len(dataset)
        avr_psnr /= epoch_size
        avr_ssim /= epoch_size
        stats = (psnrs, ssims, proc_time)

        return avr_psnr, avr_ssim, stats, outputs

    def train(self, train_dataset, val_dataset):
        """
        Train the 'train_dataset',
        if 'fine_tune' is True, we finetune the model under 'check_point' dir
        instead of training from scratch.

        The best model is save under checkpoint which is used
        for test phase or finetuning
        """

        # check fine_tuning option
        model_path = os.path.join(self.check_point, 'model.pt')
        if self.fine_tune and not os.path.exists(model_path):
            raise Exception('Cannot find %s.' % model_path)
        elif self.fine_tune and os.path.exists(model_path):
            if self.verbose:
                print('Loading %s for finetuning.' % model_path)
            self.model = torch.load(model_path)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.learning_rate)

        # capture best model
        best_val_psnr = -1

        # Train the model
        for epoch in range(self.num_epochs):
            self._epoch_step(train_dataset, epoch)
            self.scheduler.step()

            if self.verbose:
                print('Validate PSNR...')

            # compuate validate PSNR and SSIM on val dataset
            val_psnr, val_ssim, _, _ = self._check_PSNR(val_dataset)

            if self.verbose:
                print('Val PSNR: %.3fdB. Val ssim: %.3f'
                      % (val_psnr, val_ssim))

            # write the model to hard-disk for testing
            print('Saving model')
            if not os.path.exists(self.check_point):
                os.makedirs(self.check_point)
            model_path = os.path.join(self.check_point, 'epoch{}.pt'.format(epoch))
            torch.save(self.model, model_path)
            if best_val_psnr < val_psnr:
                print('Copy best model')
                target_path = os.path.join(self.check_point, 'best_model.pt')
                copyfile(model_path, target_path)
                best_val_psnr = val_psnr
            print('')

    def test(self, dataset, model_path):
        """
        Load the model stored in train_model.pt from training phase,
        then return the average PNSR on test samples.
        """
        if not os.path.exists(model_path):
            raise Exception('Cannot find %s.' % model_path)

        self.model = torch.load(model_path)
        _, _, stats, outputs = self._check_PSNR(dataset, is_test=True)
        return stats, outputs
