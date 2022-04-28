import os
from dipy.io.image import load_nifti
from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):

    def __init__(self, data_dir, args, fre_key):
        # id
        self.ids = [splitext(file)[0] for file in listdir(data_dir)]
        self.dir = data_dir
        self.num_subset = args.num_subset
        self.pred_coef = args.wt
        self.num_volumes = 6  # total volumes in one subject that used for synthesis
        self.fre_key = fre_key
        self.coeffs = ['aaa', 'aad', 'ada', 'daa']
        self.bval = args.bval
        self.cal_norm = not args.no_norm
        self.device = args.device

    def get_path_high(self, x, y):
        idx = self.ids[x]
        aaa_dir = os.path.join(self.dir, idx, 'wavelet', 'aaa', str(self.bval) + 'data' + str(y) + '.nii.gz')
        aad_dir = os.path.join(self.dir, idx, 'wavelet', 'aad', str(self.bval) + 'data' + str(y) + '.nii.gz')
        ada_dir = os.path.join(self.dir, idx, 'wavelet', 'ada', str(self.bval) + 'data' + str(y) + '.nii.gz')
        daa_dir = os.path.join(self.dir, idx, 'wavelet', 'daa', str(self.bval) + 'data' + str(y) + '.nii.gz')

        mask_dir = os.path.join(self.dir, idx, 'wavelet_mask.nii.gz')

        gt_dir = os.path.join(self.dir, idx, 'wavelet', str(self.pred_coef),
                              str(self.bval) + 'clean' + str(y) + '.nii.gz')
        return [aaa_dir, aad_dir, ada_dir, daa_dir, gt_dir, mask_dir]

    def get_path_low(self, x, y):
        idx = self.ids[x]
        dwi_dir = os.path.join(self.dir, idx, 'wavelet', str(self.pred_coef),
                               str(self.bval) + 'data' + str(y) + '.nii.gz')
        gt_dir = os.path.join(self.dir, idx, 'wavelet', str(self.pred_coef),
                              str(self.bval) + 'clean' + str(y) + '.nii.gz')

        mask_dir = os.path.join(self.dir, idx, 'wavelet_mask.nii.gz')
        return dwi_dir, gt_dir, mask_dir

    def read_data_high(self, path_):
        aaa_dir, aad_dir, ada_dir, daa_dir, gt_dir, mask_dir = path_

        dwi_aaa = np.expand_dims(load_nifti(aaa_dir)[0][:, :, :, self.z], axis=3)
        dwi_aad = np.expand_dims(load_nifti(aad_dir)[0][:, :, :, self.z], axis=3)
        dwi_ada = np.expand_dims(load_nifti(ada_dir)[0][:, :, :, self.z], axis=3)
        dwi_daa = np.expand_dims(load_nifti(daa_dir)[0][:, :, :, self.z], axis=3)

        mask = load_nifti(mask_dir)[0]
        gt = np.expand_dims(load_nifti(gt_dir)[0][:, :, :, self.z], axis=3)

        dwi = np.concatenate((dwi_aaa, dwi_aad, dwi_ada, dwi_daa), axis=3)
        if self.cal_norm:
            return self.trans_norm(dwi, gt, mask)
        else:
            return mask_dwi.transpose((3, 0, 1, 2)), gt.transpose((3, 0, 1, 2))

    def read_data_low(self, path_):
        dwi_dir, gt_dir, mask_dir = path_
        dwi = load_nifti(dwi_dir, return_img=False)[0][:, :, :, :6]
        gt = load_nifti(gt_dir, return_img=False)[0][:, :, :, :6]
        mask = load_nifti(mask_dir, return_img=False)[0]
        if self.cal_norm:
            return self.denoise_norm(dwi, gt, mask)
        else:
            return dwi.transpose((3, 0, 1, 2)), gt.transpose((3, 0, 1, 2))

    def __len__(self):
        if self.fre_key == 'low':
            return len(self.ids) * self.num_subset
        else:
            return len(self.ids) * self.num_subset * self.num_volumes

    def trans_norm(self, data, gt, mask):
        for wt in self.coeffs:
            x = self.coeffs.index(wt)
            data[:, :, :, x] = (data[:, :, :, x] - np.mean(data[:, :, :, x])) / np.std(data[:, :, :, x]) * mask

        gt = ((gt - np.mean(gt)) / np.std(gt)) * np.expand_dims(mask, axis=3)
        return data.transpose((3, 0, 1, 2)), gt.transpose((3, 0, 1, 2))

    @staticmethod
    def denoise_norm(data, gt, mask):
        data = (data - np.mean(data)) / np.std(data)
        gt = (gt - np.mean(gt)) / np.std(gt)
        for x in range(data.shape[3]):
            data[:, :, :, x] *= mask
            gt[:, :, :, x] *= mask
        return data.transpose((3, 0, 1, 2)), gt.transpose((3, 0, 1, 2))

    def __getitem__(self, i):
        self.x = int(i / (self.num_subset * self.num_volumes))
        self.y = int(i % (self.num_subset * self.num_volumes) / self.num_volumes)
        self.z = int(i % (self.num_volumes * self.num_subset) % self.num_volumes)

        if self.fre_key == 'high':
            dwi, gt = self.read_data_high(self.get_path_high(self.x, self.y))
        else:
            dwi, gt = self.read_data_low(self.get_path_low(self.x, self.y))

        return torch.tensor(dwi, dtype=torch.float32, device=torch.device(self.device)), \
               torch.tensor(gt, dtype=torch.float32, device=torch.device(self.device))
