import os
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from skimage.metrics import structural_similarity
from models.networks import *
import pywt
import argparse
import logging
import sys
import time


'''
The script provides the denoising pipeline for whole DWIs.

PREREQUISITE
You are required to prepare followings for inference:
data: format as .nii.gz, which should contains 6 volumes (6*b3) or 7 volumes (6*b1+b0);
models: should contain well-trained models for 8 channels, and organized as:
--model_dir
    |--bval
        |--aaa
            |--cnn128.pth
        ...
        |--add
            |--unet_with_SPADE.pth
        ...
normalization dictionary (optional): the normalization parameters are organized in 'key-value' format for eight channels.
--norm_dir
    |--bval
        |--aaa.npy
        |--aad.npy
        ...
        |--ddd.npy
NOTE that our models are trained with normalization, so you are required to use the same normalization method.
mask (optional): the wavelet coefficient mask for brain. 

You can download above files from the drive, see link in readme.

PATH
test_dir: the directory for the test example;
model_dir: the directory for the well-trained models, a folder contained eight sub-channel models;
mask_dir: the directory for the coefficients' mask;
norm_dir: the directory for the normalization directory, key word "mean_train", "std_train" for normalization,
"mean_gt", "std_gt" for denormalization. Note that the directory should be a folder that contained eight norm dictionary 
for eight sub-channels;
output_dir: the directory for the inference (denosied) output;
gt_dir: the directory for the clean DWI, which is used for evaluation metrix (optional);

NET CONFIGURATION
denoise_in: the input channels (volumes) for DnCNN denoising net;
denoise_out: the output channels (volumes) for DnCNN denoising net;
spade_channels: the convolutional filters in each SPADE;
dropout: the dropout rate;

OTHERS
bval: we prepared models for denoising two modality of DWIs (b=1000 and b=3000 ms/mm^2), note that the input of b1
well-trained models comprised of 7 volumes (6*b1+b0), and the input of b3 models simply contains 6 volumes (6*b3);
device: the device for calculation, cuda or cpu;

Use terminal to change the parameters, see detailed usage by the command:
python test.py -h
'''


def config_argparser(p):
    # path
    p.add_argument('--test_dir', help='Name the test directory', type=str)
    p.add_argument('--model_dir', help='Name the well-trained model directory, folder dir for whole dwi inference'
                                       'and specific model directory for channel inference', type=str,
                   default='model_zoo')
    p.add_argument('--mask_dir', help='Name the brain mask directory (wavelet)', type=str)


    p.add_argument('--norm_dir', help='Name the normalization dictionary directory', type=str, default='norm-dict')

    p.add_argument('--output_dir', help='Name the output directory', type=str)
    p.add_argument('--gt_dir',
                   help='Name the clean DWI or wavelet coefficient directory if evaluation metrix were used.',
                   type=str)

    p.add_argument('--bval', help='Name the modality (depends on b-value) of DWI', type=str, default='b1')
    p.add_argument('--device', help='Name the device for calculation, the default is CUDA', default='cuda', type=str)
    p.add_argument('--denoise_in', help='Number the input channels in denoising DnCNN net, the default is 6', type=int,
                   default=6)
    p.add_argument('--denoise_channels', help='Number the mid conv channels in denoising DnCNN net, the default is 128',
                   type=int, default=128)
    p.add_argument('--denoise_out', help='Number the output channels in denoising DnCNN net, the default is 6',
                   type=int, default=6)
    p.add_argument('--spade_channels', help='Number the initial Conv channels of the SPADE layer, the default is 32',
                   type=int, default=32)
    p.add_argument('--dropout', help='Number the dropout rate', type=int, default=0)

    p.add_argument('--metrix', help='Output the quantitative metrix (MSE, PSNR, SSIM) or not, the default is false',
                   action='store_true', default=False)

    return p


def test(arg):
    # Denoising + Translation + IDWT

    data, affine = load_nifti(arg.test_dir)
    res = dwi_infer(data, arg)

    return res, affine


def dwi_infer(dwi, arg):
    denoise_wt = ['aaa', 'aad', 'ada', 'daa']
    trans_wt = ['add', 'dad', 'dda', 'ddd']
    coeffs_pred = dict()
    coeffs = pywt.dwtn(dwi, 'db1', axes=(0, 1, 2))

    for wt in denoise_wt:
        logging.info(f'Denoising on the {wt} Channel...')
        model_dir = os.path.join(arg.model_dir, arg.bval, wt, 'cnn128.pth')
        coeffs_pred[wt] = denoise_infer(coeffs[wt], wt, model_dir, arg)[:,:,:,:6]

    lower_coeffs = np.concatenate((coeffs_pred['aaa'], coeffs_pred['aad'], coeffs_pred['ada'],
                                   coeffs_pred['daa']), axis=3)
    for wt in trans_wt:
        logging.info(f'Translation on the {wt} Channel...')
        model_dir = os.path.join(arg.model_dir, arg.bval, wt, 'unet_with_SPADE.pth')
        coeffs_pred[wt] = np.zeros(coeffs_pred['aaa'].shape)
        for i in range(6):
            idx = (i, i + 6, i + 12, i + 18)
            single_lower_coeffs = np.array([lower_coeffs[:, :, :, j] for j in idx]).transpose((1, 2, 3, 0))
            coeffs_pred[wt][:, :, :, i:i + 1] = trans_infer(single_lower_coeffs, wt, model_dir, arg)

    return pywt.idwtn(coeffs_pred, 'db1', axes=(0, 1, 2))



def read_norm(norm_dir):
    norm = dict()
    aaa_norm = np.load(os.path.join(norm_dir, 'aaa.npy'), allow_pickle=True).item()
    aad_norm = np.load(os.path.join(norm_dir, 'aad.npy'), allow_pickle=True).item()
    ada_norm = np.load(os.path.join(norm_dir, 'ada.npy'), allow_pickle=True).item()
    daa_norm = np.load(os.path.join(norm_dir, 'daa.npy'), allow_pickle=True).item()

    norm['mean'] = np.array([aaa_norm['mean_gt'], aad_norm['mean_gt'], ada_norm['mean_gt'], daa_norm['mean_gt']])
    norm['std'] = np.array([aaa_norm['std_gt'], aad_norm['std_gt'], ada_norm['std_gt'], daa_norm['std_gt']])

    return norm


def trans_infer(coeffs, wt, model_dir, arg):
    model = UNet_with_SPADE(arg).to(arg.device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    mask = load_nifti(arg.mask_dir)[0]
    norm_dir = os.path.join(arg.norm_dir, arg.bval)
    norm_dict = read_norm(norm_dir)

    coeffs = trans_norm(coeffs, mask, norm_dict)
    coeffs = torch.tensor(coeffs.transpose((3, 0, 1, 2)), dtype=torch.float32, device=device).unsqueeze(0)
    pred = model(coeffs).cpu().detach().numpy().squeeze(0).transpose((1, 2, 3, 0))

    norm_dict = np.load(os.path.join(norm_dir, wt + '.npy'), allow_pickle=True).item()
    a = trans_denorm(pred, mask, norm_dict)
    return a

def trans_norm(data, mask, norm):
    for x in range(data.shape[3]):
        data[:, :, :, x] = (data[:, :, :, x] - norm['mean'][x][0]) / norm['std'][x][0] * mask
    return data


def trans_denorm(data, mask, norm):
    return (data * norm['std_gt'][0] + norm['mean_gt'][0]) * np.expand_dims(mask, axis=3)


def denoise_infer(coeff, wt, model_dir, arg):
    model = Dncnn(arg).to(arg.device)
    model.load_state_dict(torch.load(model_dir))
    norm_dir = os.path.join(arg.norm_dir, arg.bval)
    norm_dict = np.load(os.path.join(norm_dir, wt + '.npy'), allow_pickle=True).item()
    model.eval()

    mask = load_nifti(os.path.join(arg.mask_dir))[0]

    coeff = denoise_norm(coeff, mask, norm_dict)
    coeff = torch.tensor(coeff.transpose((3, 0, 1, 2)), dtype=torch.float32, device=device).unsqueeze(0)
    pred = model(coeff).cpu().detach().numpy().squeeze(0).transpose((1, 2, 3, 0))[:,:,:,:6]
    return denoise_denorm(pred, mask, norm_dict)


def denoise_norm(data, mask, norm):
    mean = norm['mean_train']
    std = norm['std_train']
    for x in range(data.shape[3]):
        data[:, :, :, x] = (data[:, :, :, x] - mean[x]) / std[x] * mask
    return data


def denoise_denorm(data, mask, norm):
    mean = norm['mean_gt']
    std = norm['std_gt']
    for x in range(data.shape[3]):
        data[:, :, :, x] = (data[:, :, :, x] * std[x] + mean[x]) * mask
    return data


def measure(data, gt):
    gt = keep_dims(data, gt)
    mse = np.mean((gt - data) ** 2)
    psnr = 20 * np.log10(np.max(gt) / np.sqrt(mse))
    ssim = structural_similarity(data, gt, data_range=1.0, multichannel=True)
    return mse, psnr, ssim


def keep_dims(data, gt):
    diffX = data.shape[0] - gt.shape[0]
    diffY = data.shape[1] - gt.shape[1]
    diffZ = data.shape[2] - gt.shape[2]
    gt = np.pad(gt, ((diffX // 2, diffX - diffX // 2),
                     (diffY // 2, diffY - diffY // 2),
                     (diffZ // 2, diffZ - diffZ // 2),
                     (0, 0)))
    return gt


if __name__ == '__main__':
    # Config parser
    parser = argparse.ArgumentParser()
    parser = config_argparser(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Logging device information
    device = args.device
    logging.info(f'Using device {device}')

    start = time.time()
    res, affine = test(args)
    res = np.clip(res, a_min=0, a_max=None)
    end = time.time()

    save_nifti(args.output_dir, res, affine)
    logging.info(f'{args.output_dir} has been successfully saved, consuming {end - start}s')

    if args.metrix:
        data = load_nifti(args.test_dir)[0][:,:,:,:6]
        gt = load_nifti(args.gt_dir)[0][:,:,:,:6]
        mse, psnr, ssim = measure(data, gt)
        print('Original data:\n'
              'MSE:', mse, '\n',
              'PSNR:', psnr, '\n',
              'SSIM:', ssim, '\n')
        mse, psnr, ssim = measure(res, gt)
        print('denoised data:\n'
              'MSE:', mse, '\n',
              'PSNR:', psnr, '\n',
              'SSIM:', ssim, '\n')
