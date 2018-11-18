import argparse
import os
from tensorboardX import SummaryWriter

from datasets import Brains
from models import UNet3D

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np

class dice_loss(nn.Module):
    def forward(self, input, target):
        smooth = 1e-3
        iflat = input.reshape((input.size(0),-1))
        tflat = target.reshape((target.size(0),-1))
        intersection = (iflat * tflat).sum(1)
    
        return 1-(2*(intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth))

def dice_score(input, target):
    smooth = 1e-3
    iflat = input.reshape((input.size(0),-1))
    tflat = target.reshape((target.size(0),-1))
    intersection = (iflat * tflat).sum(1)
    
    return (2*(intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth))


def test_sequential(image, model, size=64, stride=64, verbose=False):
    #image = float tensor from dataloader
    model.eval()
    
    out = torch.zeros_like(image)
    
    for i in range(image.shape[3]//size):
        for j in range(image.shape[4]//size):
            crop = image[:,:,:,i*size:(i + 1)*size,j*size:(j+1)*size]
            crop_predicted = model(crop.cuda())
            crop_predicted = crop_predicted.max(1, keepdim=True)[1]
            out[:,:,:,i*size:(i+1)*size,j*size: (j+1)*size] = crop_predicted.data
    
    return out


class ExpRunner():
    def __init__(self):       
        self.model_is_initialized = False
        
        
    def init_model(self, device_id, last_ckpt=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id #string
        self.model = UNet3D(input_nc=1, output_nc=2)
        if last_ckpt is not None:
            state_dict = torch.load(last_ckpt)
            self.model.load_state_dict(state_dict)
        self.model = self.model.cuda()
        self.criterion = dice_loss().cuda()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        self.model_is_initialized = True

    def run_experiments(self, exp_name, n_epochs, batch_size=2):
        assert self.model_is_initialized, "Model had not been initialized! Use init_model()!"
            
        log_folder_path = "../logs/" + exp_name
        if not os.path.exists(log_folder_path):
            os.makedirs(log_folder_path)
        self.writer = SummaryWriter(log_folder_path)            
            
        ckpt_folder_path = "../ckpt/" + exp_name + "/"
        if not os.path.exists(ckpt_folder_path):
            os.makedirs(ckpt_folder_path)
        
        train_data = Brains(folder='../scanmasks-train/', crop=True, crop_size=(64,64), proper_crop_proba=0.5)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=20)
        
        val_data = Brains(folder='../scanmasks-val/', crop=False, proper_crop_proba=-1)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, drop_last=False, num_workers=0)
        
        global_train_i = 0
        for i in range(n_epochs):
            print("Training: epoch %d" % i)
            train_loss, global_train_i = self.train_epoch(i, train_dataloader, global_train_i)
            val_loss = self.validate(i, val_dataloader)
            
            if i%10==0:
                torch.save(self.model.state_dict(), ckpt_folder_path + "model_%d_epochs" % i)
            
            
    def train_epoch(self, n_epoch, dataloader, i):
        self.model.train(True)
    
        epoch_ce_loss = []
        epoch_dice = []
        
        for X_batch, masks_batch in dataloader:
        #         X_batch = X_batch[0]
        #         X_batch = X_batch.float()
            probs = self.model(X_batch.cuda())
            _, preds = probs.max(dim=1, keepdim=True)

#             print(preds.shape, masks_batch.shape)
            loss = self.criterion(probs[:,0,:,:,:], masks_batch.squeeze(1).cuda().float()).mean()
            self.writer.add_scalar('Dice loss per iter', loss.item(), i)
            dice = dice_score(preds.float(), masks_batch.cuda().float()).mean()
            self.writer.add_scalar('Dice score', dice.item(), i)
        #         print(X_batch.shape)
        #         print(torchvision.utils.make_grid(X_batch).shape)
            bool_mask, idx = masks_batch.reshape((*masks_batch.shape[:3],-1)).max(dim=-1)
            if torch.any(bool_mask.reshape((1,-1)).byte()==1):
                self.writer.add_image('Input image vs mask vs pred', torch.cat((X_batch[bool_mask.byte()][0], masks_batch[bool_mask.byte()][0].float(), preds[bool_mask.byte()][0].cpu().float()), dim=1), i)

        #         score = dice_score(F.threshold(preds, 0.5, 1), masks_batch)

            # train on batch
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            i += 1
            
            epoch_ce_loss.append(loss.cpu().data.numpy())
            epoch_dice.append(dice.cpu().data.numpy())
            
        total_ce_loss = np.mean(epoch_ce_loss)
        self.writer.add_scalar('Train cross entropy loss per epoch', total_ce_loss, n_epoch)
        self.writer.add_scalar('Train dice per epoch', np.mean(epoch_dice), n_epoch)
        return total_ce_loss, i
        
    def validate(self, n_epoch, dataloader):
        self.model.eval()
        
        epoch_dice = [] 
        for X_batch, masks_batch in dataloader:
            preds = test_sequential(X_batch.cuda(), self.model)
            dice = dice_score(preds.float(), masks_batch.cuda().float()).mean()
            epoch_dice.append(dice.cpu().data.numpy())
            
        total_dice = np.mean(epoch_dice)
        self.writer.add_scalar('Val dice per epoch', total_dice, n_epoch)
        return total_dice

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='name of the experiment')
    parser.add_argument('-d', '--device', type=str, required=True,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('-b', '--batch_size', type=int, default=2,
                        help='batch_size (default: 2)')
    args = parser.parse_args()
    
    exp = ExpRunner()
    exp.init_model(args.device, last_ckpt=args.resume)
    exp.run_experiments(args.name, args.epochs, batch_size=args.batch_size)