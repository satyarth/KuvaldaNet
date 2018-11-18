import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize

import pydicom
from pydicom.data import get_testdata_files

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Brains(Dataset):
    
    def __init__(self, folder='../scanmasks-train/', crop=True, crop_size=(64,64), proper_crop_proba=0.7, 
                 histeq=False):
        self.folder = folder
        self.patients = os.listdir(folder)
        self.max_dim = 24
        self.histeq = histeq
        
        self.crop = crop
        self.crop_size = crop_size
        self.proper_crop_proba = proper_crop_proba
        self.i_inds = {}
        self.j_inds = {}
        
        if self.crop:
            self.i, self.j = 0, 0
        
    def __len__(self):
        return len(self.patients)
    
    def pad(self, x): #never used
        dim_differs = self.max_dim - x.shape[0]
        zeros = torch.zeros_like(x)[:dim_differs,...]
        x = torch.cat([x, zeros])
        return x
    
    def hist_equalize(self, x):
        for i in range(x.shape[0]):
            x[i] = equalize_hist(x[i], nbins=64)
        return x
    
    def get_random_crop_index(self, x):
        size=self.crop_size
        x_shape = x.shape[1:]
        i = np.random.randint(low=0, high=x_shape[0]-size[0])
        j = np.random.randint(low=0, high=x_shape[1]-size[1])
        return i, j
    
    def get_proper_crop(self, scan, mask, random_i, random_j):
        max_i_size, max_j_size = scan.shape[1:]

        max_i = random_i + self.crop_size[0]//2
        min_i = random_i - self.crop_size[0]//2

        if max_i > max_i_size:
            min_i -= max_i-max_i_size
            max_i = max_size
        elif min_i < 0:
            max_i -= min_i
            min_i = 0

        max_j = random_j + self.crop_size[1]//2
        min_j = random_j - self.crop_size[1]//2

        if max_j > max_j_size:
            min_j -= max_j-max_j_size
            max_j = max_j_size
        elif min_j < 0:
            max_j -= min_j
            min_j = 0

        return scan[:,min_i:max_i, min_j:max_j], mask[:,min_i:max_i, min_j:max_j]
    
    def do_crop(self, x, index=(0,0)):
        size=self.crop_size
        return x[:,index[0]:index[0]+size[0], index[1]:index[1]+size[1]]
    
    def transform_scan(self, x):
        #x = torch.FloatTensor(np.array(x, dtype=np.float64))
        x = np.array(x, dtype=np.float64)
        #x = self.pad(x)
        return x
    
    def transform_mask(self, x):
        x = torch.LongTensor(np.array(x, dtype=np.float64))
        x = self.pad(x)
        return x
    
    def hist_equalize(self, x):
        for i in range(x.shape[0]):
            x[i] = equalize_hist(x[i], nbins=64)
        return x
    
    def __getitem__(self, i):
        print(self.patients[i])
        scan = np.load(self.folder+self.patients[i]+'/' + self.patients[i] + '_scan.npy')
        mask = np.load(self.folder+self.patients[i]+'/' + self.patients[i] + '_mask.npy')
        
        scan = self.transform_scan(scan)
        mask = self.transform_mask(mask)

        if self.crop:
            if np.random.rand() < self.proper_crop_proba:
                if i not in self.i_inds:
                    k_arr, i_arr, j_arr = np.where(mask==1)
                    self.i_inds[i] = i_arr
                    self.j_inds[i] = j_arr
                else:
                    i_arr = self.i_inds[i]
                    j_arr = self.j_inds[i]   

                random_ind = np.random.randint(i_arr.shape[0])
                random_i, random_j = i_arr[random_ind], j_arr[random_ind]

                scan, mask = self.get_proper_crop(scan, mask, random_i, random_j)
            else:
                scan = self.do_crop(scan, index=self.get_random_crop_index(scan))
                mask = self.do_crop(mask, index=self.get_random_crop_index(mask))
#                 scan = self.do_crop(scan, index=(self.i,self.j))
#                 mask = self.do_crop(mask, index=(self.i,self.j))
        
        if self.histeq:
            scan_equalized = self.hist_equalize(scan)
            scan_equalized = self.pad(torch.FloatTensor(scan_equalized))
            
        scan = self.pad(torch.FloatTensor(scan))
        
        if self.histeq:
            scan = torch.cat([scan, scan_equalized],0)
        
        return scan[None,...], mask[None,...]