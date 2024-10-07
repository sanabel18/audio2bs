import sys
from audio2bs import Audio2BS
from data_capitulator import get_dataloaders
import torch
import argparse
import os
import shutil
from tqdm import tqdm
import numpy as np
import logging
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data_capitulator import Dataset
import pickle


DEVICE = 'cuda'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, targets, weights):
        return ((inputs - targets)**2 ) * weights



def train():    
    """training audio2blendshape from prepared datasets
       if model_path was assigned, the corresponding config will be loaded 
       the trainning will continue through the epoches.

    """
    args = get_args()
    save_path = args.save_path
    logdir = args.logdir
    ppe = args.ppe
    bs_number = args.bs_number
    dataset_path = args.dataset_path
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    
    writer = SummaryWriter(logdir)
    
    if args.model_path:
        state_dict = torch.load(args.model_path)
        model_params = state_dict['model']
        optimizer_params = state_dict['optimizer']
        epoch_start = state_dict['epoch']
        iteration_start = state_dict['iteration']
        ppe = state_dict['ppe']
        bs_number = state_dict['bs_number']
        model = Audio2BS(ppe, bs_number)
        model.load_state_dict(model_params)
    else:
        epoch_start = 0
        iteration_start = 0
        model = Audio2BS(ppe, bs_number)
        
        


    model.cuda()
    model.train()
 

    f = open(dataset_path,'rb')
    dataset = pickle.load(f)
    data_loader = get_dataloaders(dataset)
     
    train_loader = data_loader["train"]
    valid_loader = data_loader["valid"]

    if args.wmse: 
        mse_loss_weight = dataset['mse_loss_weight']
        logger.info("mse_loss_weight {}".format(mse_loss_weight))
        mse_loss_weight = torch.FloatTensor(mse_loss_weight).to(DEVICE)
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    if args.model_path:
        optimizer.load_state_dict(optimizer_params)
    if args.wmse:
        criterion = weighted_MSELoss()
    else:
        criterion = torch.nn.MSELoss()


    iteration = iteration_start
    epoch = epoch_start
    mean_loss_list = []
    mean_valid_loss_list = []
    for e in range(args.epoch):
        logger.info("e, epoch_start  {} {}".format(e, epoch_start))
        epoch = epoch_start + (e+1)
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, bswts, bswts_frm_num, key) in pbar:
            iteration +=1
            predicted_bswts = model(audio, bswts_frm_num)
            if args.wmse:
                loss = criterion(predicted_bswts, bswts, mse_loss_weight)
            else:
                loss = criterion(predicted_bswts, bswts)
            loss = torch.mean(loss)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((epoch), iteration ,np.mean(loss_log)))

        current_loss = np.mean(loss_log)
        writer.add_scalar("Loss/train", current_loss, epoch)
        #++++++++++validate++++++
        mean_loss_list.append(np.mean(loss_log))
        valid_loss_log = []
        model.eval()

        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader))

        for i, (audio, bswts, bswts_frm_num, key) in pbar:
            predicted_bswts = model(audio, bswts_frm_num)
            if args.wmse:
                loss = criterion(predicted_bswts, bswts, mse_loss_weight)
            else:
                loss = criterion(predicted_bswts, bswts)
            loss = torch.mean(loss)
            valid_loss_log.append(loss.item())
    
        current_loss = np.mean(valid_loss_log)
        logger.info("epcoh: {}, current loss:{:.7f}".format(epoch, current_loss))
        writer.add_scalar("Loss/validate", current_loss, epoch)
        mean_valid_loss_list.append(np.mean(valid_loss_log))
 
        if (epoch > 0 and epoch % 5 == 0) or epoch == args.max_epoch:
            state_dict = {}
            state_dict['model'] = model.state_dict()
            state_dict['optimizer'] = optimizer.state_dict()
            state_dict['epoch'] = epoch
            state_dict['iteration'] = iteration 
            state_dict['ppe'] = ppe
            state_dict['bs_number'] = bs_number

            torch.save(state_dict, os.path.join(save_path,'{}_model.pth'.format(epoch)))

       
    np.save(os.path.join(save_path, 'mean_loss_log_list.npy'), np.array(mean_loss_list) )
    np.save(os.path.join(save_path, 'mean_valid_log_list.npy'), np.array(mean_valid_loss_list) )
    return model

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_args():
    parser = argparse.ArgumentParser(description='Audio2BS: Speech-Driven 3D Facial Blendshaps')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--save_path", type=str, help='model save path')
    parser.add_argument("--logdir", type=str, help='logdir')
    parser.add_argument("--model_path", type=str, default=None, help='model checkpoint pth file')
    parser.add_argument("--dataset_path", type=str, help='data set path')
    parser.add_argument("--fps", type=int, default=30, help='fps')
    parser.add_argument("--epoch", type=int, default=10, help='number of epoches for this run')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation step')
    parser.add_argument("--max_epoch", type=int, default=500, help='number of max epochs')
    parser.add_argument("--wmse", action='store_true', help='use weighted MSE loss')
    parser.add_argument("--ppe", action='store_true', help='use PPE layer')
    parser.add_argument("--bs_number", type=int, default=46, help='blend shape number')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    train()
