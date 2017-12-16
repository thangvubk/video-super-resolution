from __future__ import division
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import math
import scipy.misc
import progressbar
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os



class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training super resolution
    The Solver accepts both training and validation data label so it can 
    periodically check the PSNR on training and validation.
    
    To train a model, you will first construct a Solver instance, pass the model,
    datasets, and various option (optimizer, loss_fn, batch_size, etc) to the
    constructor.

    After train() method is called, the 'self.model' will be the best model on 
    validation set. The best model is saved into 'check_point' dir, which is used
    for the testing time. 

    For statistics, 'loss' history, 'avr_train_psnr' history, and 'avr_val_psnr' history
    are also saved. 
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
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
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
        self.hist_train_psnr = []
        self.hist_val_psnr = []
        self.hist_loss = []
    
    def _epoch_step(self, dataset, epoch):
        """ Perform 1 training 'epoch' on the 'dataset'"""
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=8)

        num_batchs = len(dataset)//self.batch_size

        # observe the training progress
        if self.verbose:
            bar = progressbar.ProgressBar(max_value=num_batchs)

        running_loss = 0
        for i, (input_batch, label_batch) in enumerate(dataloader):

            #Wrap with torch Variable
            input_batch, label_batch = self._wrap_variable(input_batch,
                                                           label_batch,
                                                           self.use_gpu)

            #zero the grad
            self.optimizer.zero_grad()

            # Forward
            output_batch = self.model(input_batch)
            loss = self.loss_fn(output_batch, label_batch)

            # save statistic
            self.hist_loss.append(loss.data[0])
            running_loss += loss.data[0]
            
            # Backward + update
            loss.backward()
            self.optimizer.step()

            if self.verbose:
                bar.update(i, force=True)

        if self.verbose:
            print('Epoch  %5d, loss %.5f' \
                        %(epoch, running_loss/num_batchs))

    def _wrap_variable(self, input_batch, label_batch, use_gpu):
        if use_gpu:
            input_batch, label_batch = (Variable(input_batch.cuda()),
                                        Variable(label_batch.cuda()))
        else:
            input_batch, label_batch = (Variable(input_batch),
                                        Variable(label_batch))
        return input_batch, label_batch
    
    def _comput_PSNR(self, imgs1, imgs2):
        """Compute PSNR between two image array and return the psnr summation"""
        N = imgs1.size()[0]
        imdiff = imgs1 - imgs2
        imdiff = imdiff.view(N, -1)
        rmse = torch.sqrt(torch.mean(imdiff**2, dim=1))
        psnr = 20*torch.log(255/rmse)/math.log(10) # psnr = 20*log10(255/rmse)
        psnr =  torch.sum(psnr)
        return psnr

    def _check_PSNR(self, dataset, is_test=False):
        """
        Get the output of model with the input being 'dataset' then 
        compute the PSNR between output and label.
        
        if 'is_test' is True, psnr and output of each image is also 
        return for statistics and generate output image at test phase
        """

        # process one image per iter for test phase
        if is_test:
            batch_size = 1
        else:
            batch_size = self.batch_size

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8)
        
        avr_psnr = 0
        
        # book keeping variables for test phase
        psnrs = [] # psnr for each image
        outputs = [] # output for each image

        for batch, (input_batch, label_batch) in enumerate(dataloader):
            input_batch, label_batch = self._wrap_variable(input_batch,
                                                           label_batch,
                                                           self.use_gpu)
            
            output_batch = self.model(input_batch)

            output = output_batch.data
            label = label_batch.data

            output = (output + 0.5)*255
            label = (label + 0.5)*255
            
            output = output.squeeze(dim=1)
            label = label.squeeze(dim=1)
            
            psnr = self._comput_PSNR(output, label)
            avr_psnr += psnr
            
            # save psnrs and outputs for statistics and generate image at test time
            if is_test:
                psnrs.append(psnr)
                np_output = output.cpu().numpy()
                outputs.append(np_output[0])

            
        epoch_size = len(dataset)
        avr_psnr /= epoch_size

        return avr_psnr, psnrs, outputs
     
    def train(self, train_dataset):
        """
        Train the 'train_dataset',
        if 'fine_tune' is True, we finetune the model under 'check_point' dir
        instead of training from scratch.

        The best model is save under checkpoint which is used
        for test phase or finetuning
        """

        # check fine_tuning option
        model_path = os.path.join(self.check_point + 'model.pt')
        if self.fine_tune and not os.path.exists(model_path):
            raise Exception('Cannot find %s.' %model_path)
        elif self.fine_tune and os.path.exists(model_path):
            if self.verbose:
                print('Loading %s for finetuning.' %model_path)
            self.model = torch.load(model_path)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # capture best model
        best_val_psnr = -1
        best_model_state = self.model.state_dict()

        # Train the model
        for epoch in range(self.num_epochs):
            self._epoch_step(train_dataset, epoch)
            self.scheduler.step()

            if self.verbose:
                print('Computing PSNR...')

            # capture running PSNR on train and val dataset
            train_psnr, _, _ = self._check_PSNR(train_dataset)
            self.hist_train_psnr.append(train_psnr)
            
            if self.verbose:
                print('%s Average train PSNR %.3fdB' %(self.model.name, train_psnr))
                print('')
            
        # write the model to hard-disk for testing
        if not os.path.exists(self.check_point):
            os.makedirs(self.check_point)
        model_path = os.path.join(self.check_point, 'model.pt')
        torch.save(self.model, model_path)

    def test(self, dataset):
        """
        Load the model stored in train_model.pt from training phase,
        then return the average PNSR on test samples. 
        """
        model_path = os.path.join(self.check_point, 'model.pt')
        if not os.path.exists(model_path):
            raise Exception('Cannot find %s.' %model_path)
        
        self.model = torch.load(model_path)
        _, psnrs, outputs = self._check_PSNR(dataset, is_test=True)
        return psnrs, outputs
            
