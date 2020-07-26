# -*- coding: utf-8 -*-

import argparse
import torch
import json
import numpy as np
from torch import optim, nn
from shakespeare_dataset import RomeoDataset, RandomCrop, OneHotEncoder, ToTensor
from network import Network, train_batch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pathlib import Path

    
##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Train the sonnet generator network.')

# Dataset
parser.add_argument('--datasetpath',    type=str,   default='Shakespeare_RomeoAndJuliet.txt', help='Path of the train txt file')
parser.add_argument('--crop_len',       type=int,   default=100,               help='Number of input letters')
parser.add_argument('--alphabet_len',   type=int,   default=34,                help='Number of letters in the alphabet')

# Network
parser.add_argument('--hidden_units',   type=int,   default=128,    help='Number of RNN hidden units')
parser.add_argument('--layers_num',     type=int,   default=2,      help='Number of RNN stacked layers')
parser.add_argument('--dropout_prob',   type=float, default=0.3,    help='Dropout probability')

# Training
parser.add_argument('--batchsize',      type=int,   default=154,   help='Training batch size')
parser.add_argument('--num_epochs',     type=int,   default=5000,    help='Number of training epochs')

# Save
parser.add_argument('--out_dir',     type=str,   default='model',    help='Where to save models and params')

##############################
##############################
##############################


if __name__ == '__main__':
    
    ##############################
    ##############################
    ## CROSS VALIDATION WITH K-FOLD
    ##############################
    
    # Parse input arguments
    args = parser.parse_args()
    
    #%% Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)
    
    # Parameters grid
    params = {
        'lr':[1e-3, 1e-4],
        'crop_len': [200, 400],
        'batchsize': [500],
        'hidden_units': [256, 512]
    }
    
    loss_opt = 1000000000
    lr_opt = 0
    crop_len_opt = 0
    batchsize_opt = 0
    hidden_units_opt = 0
    
    # Create a dataset for each different batch size and crop length
    for batchsize in params['batchsize']:
        for crop_len in params['crop_len']:
            # Create dataset
            trans = transforms.Compose([RandomCrop(crop_len),
                                            OneHotEncoder(args.alphabet_len),
                                            ToTensor()
                                            ])
                
            dataset = RomeoDataset(filepath=args.datasetpath, 
                                             crop_len=crop_len,
                                             transform=trans)       
            
            # Fold division
            kfold_ind = list(range(0, len(dataset)))
            train_set_1 = Subset(dataset, kfold_ind[0:int(len(kfold_ind)/3)])
            train_set_2 = Subset(dataset, kfold_ind[int(len(kfold_ind)/3):2*int(len(kfold_ind)/3)])
            train_set_3 = Subset(dataset, kfold_ind[2*int(len(kfold_ind)/3):])
            
            set_list = [train_set_1, train_set_2, train_set_3]
            
            for lr in params['lr']:
                for hidden_units in params['hidden_units']:
                    loss_temp = [] # to save the MSE for each fold
                    flag=True
                    for folds in range(3):
                        if (flag==True):
                            # Initialize network
                            net = Network(input_size=args.alphabet_len, 
                                          hidden_units=hidden_units, 
                                          layers_num=args.layers_num, 
                                          dropout_prob=args.dropout_prob)
                            net.to(device)
                            
                            # Train network
                            # Define Dataloader
                            dataloader = DataLoader(set_list[folds], batch_size=batchsize, shuffle=True)
                            # Define optimizer
                            #optimizer = optim.Adam(net.parameters(), weight_decay=5e-4)
                            optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
                            # Define loss function
                            loss_fn = nn.CrossEntropyLoss()
                            
                            # Start training
                            for epoch in range(args.num_epochs):
                                if ((epoch + 1)%100 == 0):
                                    print('##################################')
                                    print('## EPOCH %d' % (epoch + 1))
                                    print('##################################')
                                # Iterate batches
                                for batch_sample in dataloader:
                                    # Extract batch
                                    batch_onehot = batch_sample['encoded_onehot'].to(device)
                                    # Update network
                                    batch_loss = train_batch(net, batch_onehot, loss_fn, optimizer)
                                    if ((epoch + 1)%100 == 0):
                                        print('\t Training loss (single batch):', batch_loss)
                            
                            loss_temp.append(batch_loss)
                            if (np.mean(loss_temp)>4):
                                flag=False
                                
                    # Compare with the previous results 
                    loss_avg = np.mean(np.array(loss_temp))
                    if (loss_avg<=loss_opt):
                        loss_opt = loss_avg
                        lr_opt = lr
                        crop_len_opt = crop_len
                        batchsize_opt = batchsize
                        hidden_units_opt = hidden_units
                    
                    params_set = {
                        'lr':[lr],
                        'crop_len': [crop_len],
                        'batchsize': [batchsize],
                        'hidden_units': [hidden_units]
                                    }
                    print('SET OF PARAMETERS: ' + str(params_set))
                    print('AVG LOSS FOR THIS SET: ' + str(loss_avg))
                    
    params_set_opt = {
        'lr':[lr_opt],
        'crop_len': [crop_len_opt],
        'batchsize': [batchsize_opt],
        'hidden_units': [hidden_units_opt]
                        }                
    print('OPTIMAL SET OF PARAMETERS: ' + str(params_set_opt))
    print('AVG LOSS FOR OPTIMAL SET: ' + str(loss_opt))
    
    
    
    

        
