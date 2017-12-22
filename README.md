# Video Super Resolution, SRCNN, MFCNN, VDCN (ours) benchmark comparison  
This is a pytorch implementation of video super resolution algorithms [SRCNN](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf), [MFCNN](http://cs231n.stanford.edu/reports/2016/pdfs/212_Report.pdf), and [VDCN](https://drive.google.com/open?id=1A6mHsTWZZhWai8evuEjS-HEGmB2q49fh) (ours). This project is used for one of my course, which aims to improve the performance of the baseline (SRCNN, VDCN). 

To run this project you need to setup the environment, download the dataset, run script to process data, and then you can train and test the network models. I will show you step by step to run this project and i hope it is clear enough :D. 
## Prerequisite 
I tested my project in Corei7, 64G RAM, GPU Titan X. Because it use big dataset so you should have CPU/GPU strong enough and about 16 or 24G RAM. 
## Dataset
First, download dataset from this [link](https://drive.google.com/open?id=1-5eKvxDnIqrXE3ABSk6RcPwMrgsKeCsw) and put it in this project. FYI, the training set (IndMya trainset) is taken the India and Myanmar video from [Hamonics](https://www.harmonicinc.com/free-4k-demo-footage/) website. The test sets include IndMya and vid4 (city, walk, foliage, and calendar). After the download completes, unzip it. Your should see the path of data is ``video-super-resolution/data/train/``. 
## Process data
The data is processed by MATLAB scripts, the reason for that is interpolation implementation of MATLAB is different from Python. To do that, open your MATLAB then
```
$ cd matlab_scripts/
$ generate_train_video
```
When the script is running, you should see the output as follow

![create_train](https://github.com/thangvubk/video-super-resolution/blob/master/install-instructions/create_train.PNG)

After the scipt finishes, you should see something like

![creat_train_result](https://github.com/thangvubk/video-super-resolution/blob/master/install-instructions/create_train_result.PNG)

As you can see, we have a dataset of ``data`` and ``label``. The train dataset will be stored in the path ``video-super-resolution/preprocessed_data/train/3x/dataset.h5``

Do the similar thing with test set:
```
$ generate_test_video
```
> NOTE: If you want to run train and test the network with different dataset and frame up-scale factor, you should modify the dataset, and scale variable in the ``generate_test_video`` and ``generate_train_video`` scripts (see the scripts for instructions).
## Setup
Install depenencies: ``pip install -r requirement.txt``

Install Pytorch: follow instruction in Pytorch official [website](http://pytorch.org/). Based on your hardware and python version, install the appopriate Pytorch version.
For example, my machine run Python2.7 and Cuda8, so i can install Pytorch with 
```
$ pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl 
$ pip install torchvision 
```
## Execute the code
To train the network:
```python train.py --verbose```

you should see something like

![train](https://github.com/thangvubk/video-super-resolution/blob/master/install-instructions/training.PNG)
To test the network:
```python test.py```

you should see something like

![test](https://github.com/thangvubk/video-super-resolution/blob/master/install-instructions/testing.PNG)

The experiment results will be saved in results/
>NOTE: That is the simplest way to train and test the model, all the settings will take default values. You can add options for training and testing. For example if i want to train model ``MFCNN``, initial learning-rate 1e-2, num of epoch 100, batch_size 64, scale factor 3, verbose true: ``python train.py -m MFCNN -l 1e-2 -n 100 -b 64 -s 3 --verbose``. See ``python main.py --help`` and ``python test.py --help`` for detail information. 

## Benchmark comparisions
our network architecture is similar to figure below. Which use 5 consecutive low-resolution frames as the input and produce the high resolution center frame.

![network_architecture](https://github.com/thangvubk/video-super-resolution/blob/master/install-instructions/network_architecture.PNG)

Benchmark comparsions on vid4 dataset

Quantity:
![quantity](https://github.com/thangvubk/video-super-resolution/blob/master/install-instructions/quantitative.PNG)

Quality:
![quality](https://github.com/thangvubk/video-super-resolution/blob/master/install-instructions/qualitative.PNG)

see our report [VDCN](https://drive.google.com/open?id=1A6mHsTWZZhWai8evuEjS-HEGmB2q49fh) for more comparison. 

## Project explaination
- ``train.py``: where you can start to train the network
- ``test.py``: where you can start to test the network
- ``model.py``: declare SRCNN, MFCNN, and our model with different network depth (default 20 layers). Note that our network in the code have name VRES. 
- ``SR_dataset.py``: declare dataset for each model
- ``solver.py``: encapsulate all the logics to train the network
- ``pytorch_ssim.py``: pytorch implementation for SSIM loss (with autograd), clone from this [repo](https://github.com/Po-Hsun-Su/pytorch-ssim)
- ``loss.py``: loss function for models

## Building your own model
To create your new model you need to define a new network architecture and new dataset class. See ``model.py`` and ``SR_datset.py`` for the idea :D. 

I hope my instructions are clear enough for you. If you have any problem, you can contact me through thangvubk@gmail.com or use the issue tab. If you are insterested in this project, you are very welcome. Many Thanks. 
