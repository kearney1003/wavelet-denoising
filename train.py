import logging
from torch import optim
from tqdm import tqdm
from models.networks import *
from utils.dataset import *
from utils.plot import plot_curve
from utils.losses import CharbonnierLoss
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import torch
from torch import nn
import math
import argparse

'''
The script provides a training function.

PREREQUISITE

You are required to provide the train_dir/val_dir, and the training files needed to be renamed as 
wavelet channel / bval + (data/clean) + subset_id + .nii.gz;
e.g.
--train_dir
    |--subject id
        |--aaa
            |--b1data0.nii.gz
            |--b1clean0.nii.gz
            |--b1data1.nii.gz
            |--b1clean1.nii.gz
            |--mask.nii.gz (optional)
        |--aad
            ...
            
Or you can revise the code in utils.dataset.py by your accustomed format.

PATH
train_dir: the directory for train samples;
val_dir: the directory for validate samples;

TRAINING CONFIGURATION
device: the device for calculation, cuda or cpu;
epochs: the number of training iterations;
batch: the number of batch size;
lr: the learning rate;
wd: the weight decay in adam optimizer;
loss: the loss function for back propagation, l1 or l2;
no_cp: do not save the checkpoint;
no_curve: do not save the loss curve;
no_init: do not initialize the net;
no_norm: do not apply normalization method;

DATASET CONFIGURATION
wt: the decomposed wavelet channel;
bval: the modality (depends on b-value) of DWI;
num_subset: you can generate multiple subsets from one subject (6 out of 90) in HCP dataset;

NET CONFIGURATION
net: the network;
denoise_in: the input channels (volumes) for DnCNN denoising net;
denoise_out: the output channels (volumes) for DnCNN denoising net;
spade_channels: the convolutional filters in each SPADE;
dropout: the dropout rate;

Use terminal to change the parameters, see detailed usage by the command:
python train.py -h

'''


def train_net(net, args):  # whether normalize the data
    if args.wt in ['aaa', 'aad', 'ada', 'daa']:
        fre_key = 'low'
    elif args.wt in ['add', 'dad', 'dda', 'ddd']:
        fre_key = 'high'
    else:
        logging.info('Please input the right wavelet channel. The code only supports for one level of 3D wavelet '
                     'decomposition in total of eight channels')
        sys.exit()

    train_dataset = BasicDataset(args.train_dir, args, fre_key)
    val_dataset = BasicDataset(args.val_dir, args, fre_key)

    logging.info(f'Creating training dataset with {len(train_dataset)} examples')
    logging.info(f'and validation dataset with {len(val_dataset)} examples')

    if args.loss == 'l2':
        criterion = nn.MSELoss(reduction='mean')
    elif args.loss == 'l1':
        criterion = CharbonnierLoss()
    else:
        logging.info('Wrong type of the loss function, please use l1 or l2.')
        sys.exit()

    # net = config_net(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch}
        Learning rate:   {args.lr}
        Loss function:   {str(criterion)}
        Checkpoints:     {not args.no_cp}
        Loss Curve:      {not args.no_curve}      
        Initialization:  {not args.no_init}
        Calculate Norms: {not args.no_norm}
        Training dir:    {args.train_dir}
        Validation dir:  {args.val_dir}
        device:          {str(args.device)}

    ''')

    if not args.no_init:
        net.apply(weight_init)
        logging.info('Initializing network...')

    # load data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                               pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                                             pin_memory=False)

    # train epoch loss, validation epoch loss
    train_loss = []
    validate_loss = []
    min_loss = 1

    # iteration loop
    for epoch in range(args.epochs):
        epoch_trainloss, epoch_valloss = 0, 0

        # set progress bar, start iteration
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='dwi') as pbar:

            # training
            net.train()
            for batch, (img, gt) in enumerate(train_loader):
                img = img.to(device=device, dtype=torch.float32)  # to device

                gt = gt.to(device=device, dtype=torch.float32)

                img = net(img)
                loss = criterion(img, gt)
                pbar.set_postfix(**{'loss (dwi)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update progress bar
                pbar.update(img.shape[0])
                logging.info('training batch loss:{:.3e}'.format(loss.item()))
                epoch_trainloss += loss.item()

        epoch_trainloss = epoch_trainloss / (batch + 1)
        logging.info('traning epoch loss:{:.3e}'.format(epoch_trainloss))

        train_loss.append(epoch_trainloss)
        logging.info('Validating...')

        net.eval()
        with torch.no_grad():
            for val_batch, (val_img, val_gt) in enumerate(val_loader):
                val_img = val_img.to(device=device, dtype=torch.float32)
                val_gt = val_gt.to(device=device, dtype=torch.float32)
                val_img = net(val_img)

                val_loss = criterion(val_img, val_gt)

                epoch_valloss += val_loss.item()
        epoch_valloss = epoch_valloss / (val_batch + 1)

        validate_loss.append(epoch_valloss)
        logging.info('validation epoch loss:{:.3e}'.format(epoch_valloss))

        if not args.no_cp:
            torch.save(net.state_dict(), dir_checkpoint + f'/CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        if epoch_valloss < min_loss:
            min_loss = val_loss
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'best_val_cp.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

        np.savetxt(os.path.join(dir_results, 'train_loss.txt'), train_loss)
        np.savetxt(os.path.join(dir_results, 'val_loss.txt'), validate_loss)

        if not args.no_curve:
            plot_curve(train_loss, validate_loss, args.wt)
            logging.info('Loss curve saved!')


def weight_init(layer):  # Initialization
    if isinstance(layer, nn.Conv3d):
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2.0 / n))
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm3d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()


def config_argparser(p):
    p.add_argument('--train_dir', help='Name the training directory', type=str, default='train/')
    p.add_argument('--val_dir', help='Name the validation directory', type=str, default='val/')

    p.add_argument('-b', '--batch', help='Number the batch size, the default is 1', type=int, default=1)
    p.add_argument('-e', '--epochs', help='Number the iterations, the default is 100', type=int, default=2)
    p.add_argument('-l', '--lr', help='Change the learning rate, the default is 1e-4', type=float, default=1e-4)
    p.add_argument('--wd', help='weight decay rate of Adam optimizer, the default is 1e-8', type=float, default=1e-8)
    p.add_argument('--loss', help='Name the loss function', type=str, default='l2')

    p.add_argument('--device', help='Name the device for calculation, the default is CUDA', default='cuda', type=str)
    p.add_argument('--no_cp', help='Do not save the checkpoint', action='store_true', default=False)
    p.add_argument('--no_curve', help='Do not save the loss curve', action='store_true', default=False)
    p.add_argument('--no_init', help='Do not initial the model', action='store_true', default=False)
    p.add_argument('--no_norm', help='Do not normalize the data with the z-score method', action='store_true',
                   default=False)

    p.add_argument('--wt', help='Name the decomposed wavelet channel', type=str, default='aaa')

    p.add_argument('--bval', help='Name the modality (depends on b-value) of DWI', type=str, default='b1')
    p.add_argument('--num_subset', help='Number of the subset of one subject', type=int, default=5)

    p.add_argument('--net',
                   help='Choose the denoising / channel-translation network (dncnn, unet or unet+spade), the default is '
                        'DnCNN denoising net',
                   type=str, default='dncnn')

    p.add_argument('--denoise_in', help='Number the input channels in denoising DnCNN net, the default is 6', type=int,
                   default=6)
    p.add_argument('--denoise_channels', help='Number the mid conv channels in denoising DnCNN net, the default is 128',
                   type=int, default=128)
    p.add_argument('--denoise_out', help='Number the output channels in denoising DnCNN net, the default is 6',
                   type=int, default=6)

    p.add_argument('--spade_channels', help='Number the initial Conv channels of the SPADE layer, the default is 32',
                   type=int, default=32)
    p.add_argument('--dropout', help='Number the dropout rate', type=float, default=0.1)

    return p


def config_net(arg):
    if arg.net == 'dncnn':
        return Dncnn(arg).to(arg.device)
    elif arg.net == 'unet':
        return UNet(arg).to(arg.device)
    elif arg.net == 'unet+spade':
        return UNet_with_SPADE(arg).to(arg.device)
    else:
        logging.info('ERROR! Wrong type of the translation net.')
        sys.exit()


def exception_check(arg):
    train_dir = arg.train_dir
    if not os.path.exists(train_dir):
        logging.info('ERROR! The training directory does not exist, please input the right directory.')
        sys.exit()

    val_dir = arg.val_dir
    if not os.path.exists(val_dir):
        logging.info('ERROR! The validation directory does not exist, please input the right directory.')
        sys.exit()

    dev = arg.device
    if dev == 'cuda' and (not torch.cuda.is_available()):
        logging.info('ERROR! The CUDA is not available in current environment, '
                     'please check your cuda version and related configuration')
        sys.exit()

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)


def save_args(arg):
    dict = arg.__dict__
    with open(os.path.join(dir_results, 'settings.txt'), 'w') as f:
        f.writelines('--------------- start ---------------\n')
        for k, v in dict.items():
            f.writelines(k + ' : ' + str(v) + '\n')
        f.writelines('--------------- end ---------------\n')
    print('Saved settings.')


if __name__ == '__main__':
    # Config parser
    parser = argparse.ArgumentParser()
    parser = config_argparser(parser)
    args = parser.parse_args()

    # Config basic logs
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Define the path
    dir_checkpoint = os.path.join('checkpoint', args.wt)
    dir_results = os.path.join('results', args.wt)

    # Report import exceptions raised by parser error
    exception_check(args)

    # save settings
    save_args(args)

    # Logging device information
    device = args.device
    logging.info(f'Using device {device}')

    model = config_net(args)

    train_net(net=model, args=args)
