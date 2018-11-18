import numpy as np
from datasets import Brains
import matplotlib.pyplot as plt
from IPython import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms
from torch.utils.data import Dataset

def dice_score(input, target):
    smooth = 1e-3
    iflat = input.reshape((input.size(0),-1))
    tflat = target.reshape((target.size(0),-1))
    intersection = (iflat * tflat).sum(1)
    
    return ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth))

def validate(model, postprocess=None):
    data = Brains(crop=False, folder='../scanmasks-val/')
    batch_size = 1
    valit = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    dices = np.array([])
    for X_batch, masks_batch in valit:
        preds = test_sequential(X_batch, model)
        if postprocess:
            preds = postprocess(preds)
        d = dice_score(preds, masks_batch.type(torch.FloatTensor))
        dices = np.hstack([dices, d.numpy()])
        
    return np.mean(dices)

def test_sequential(image, model, size=64, stride=64, verbose=False):
    #image = float tensor from dataloader
    model.eval()
    
    out = torch.zeros_like(image)
    
    for i in range(image.shape[3]//size):
        for j in range(image.shape[4]//size):
            crop = image[:,:,:,i*size:(i + 1)*size,j*size:(j+1)*size]
            crop_predicted = model(Variable(crop).cuda())
            crop_predicted = crop_predicted.max(1, keepdim=True)[1]
            out[:,:,:,i*size:(i+1)*size,j*size: (j+1)*size] = crop_predicted.data
    
    return out

def postprocess(network_output, mask_path='../convexmask.npy'):
    
    mask = np.load(mask_path)
    
    postproc = torch.zeros_like(network_output)
    postproc[0] = torch.FloatTensor(mask)[None,...]
#     postproc[1] = torch.FloatTensor(mask)[None,...]
    
    return network_output*postproc

def pad_bedik(x):
    zeros = torch.zeros(1,1,24,512,24)
    x=torch.cat([x,zeros], dim=4)
    return x

def unpad_bedik(x):
    return x[:,:,:,:,:488]