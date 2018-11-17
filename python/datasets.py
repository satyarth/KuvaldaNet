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
    
    def __init__(self, folder='../scansmasks/', crop=True, crop_size=(256,256)):
        self.folder = folder
        self.patients = os.listdir(folder)
        self.max_dim = 24
        
        self.crop = crop
        self.crop_size = crop_size
        
        if self.crop:
            self.i, self.j = 0, 0
        
    def __len__(self):
        return len(self.patients)
    
    def pad(self, x): #never used
        dim_differs = self.max_dim - x.shape[0]
        zeros = torch.zeros_like(x)[:dim_differs,...]
        x = torch.cat([x, zeros])
        return x
    
    def get_random_crop_index(self, x):
        size=self.crop_size
        x_shape = x.shape[1:]
        i = np.random.randint(low=0, high=x_shape[0]-size[0])
        j = np.random.randint(low=0, high=x_shape[1]-size[1])
        return i, j
    
    def do_crop(self, x, index=(0,0)):
        size=self.crop_size
        return x[:,index[0]:index[0]+size[0], index[1]:index[1]+size[1]]
    
    def transform_scan(self, x):
        x = torch.FloatTensor(np.array(x, dtype=np.float64))
        x = self.pad(x)
        return x
    
    def transform_mask(self, x):
        x = torch.LongTensor(np.array(x, dtype=np.float64))
        x = self.pad(x)
        return x
    
    def __getitem__(self, i):
        
        scan = np.load(self.folder+self.patients[i]+'/' + self.patients[i] + '_scan.npy')
        mask = np.load(self.folder+self.patients[i]+'/' + self.patients[i] + '_mask.npy')
        
        scan = self.transform_scan(scan)
        mask = self.transform_mask(mask)
        
        if self.crop:
            scan = self.do_crop(scan, index=(self.i,self.j))
            mask = self.do_crop(mask, index=(self.i,self.j))
        
        return scan[None,...], mask[None,...]