# -*- coding: utf-8 -*-

import argparse
import torch
import json
from torch import nn
from shakespeare_dataset import RomeoDataset, RandomCrop, OneHotEncoder, ToTensor
from network import Network, train_batch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path


##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Train the sonnet generator network.')

# Dataset
parser.add_argument('--datasetpath',    type=str,   default='Shakespeare_RomeoAndJuliet.txt', help='Path of the train txt file')
parser.add_argument('--crop_len',       type=int,   default=200,               help='Number of input letters')
parser.add_argument('--alphabet_len',   type=int,   default=34,                help='Number of letters in the alphabet')

# Network
parser.add_argument('--hidden_units',   type=int,   default=256,    help='Number of RNN hidden units')
parser.add_argument('--layers_num',     type=int,   default=2,      help='Number of RNN stacked layers')
parser.add_argument('--dropout_prob',   type=float, default=0.3,    help='Dropout probability')

# Training
parser.add_argument('--batchsize',      type=int,   default=500,   help='Training batch size')
parser.add_argument('--num_epochs',     type=int,   default=100000,    help='Number of training epochs')

# Save
parser.add_argument('--out_dir',     type=str,   default='model',    help='Where to save models and params')

##############################
##############################
##############################

if __name__ == '__main__':
    
    # Parse input arguments
    args = parser.parse_args()
    
    #%% Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)

    #%% Create dataset
    trans = transforms.Compose([RandomCrop(args.crop_len),
                                OneHotEncoder(args.alphabet_len),
                                ToTensor()
                                ])
    
    dataset = RomeoDataset(filepath=args.datasetpath, 
                                 crop_len=args.crop_len,
                                 transform=trans)
    
    #%% Initialize network
    net = Network(input_size=args.alphabet_len, 
                  hidden_units=args.hidden_units, 
                  layers_num=args.layers_num, 
                  dropout_prob=args.dropout_prob)
    net.to(device)
    
    #%% Train network
    
    # Define Dataloader
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    # Define optimizer
    #optimizer = optim.Adam(net.parameters(), weight_decay=5e-4)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    epochs_count = 0
    loss_temp = 1000000
    # Start training
    print('## TRAINING STARTED ##')
    for epoch in range(args.num_epochs):
        if (epochs_count < 300):
            if ((epoch + 1) % 200 == 0):
                print('##################################')
                print('## EPOCH %d' % (epoch + 1))
                print('##################################')
            # Iterate batches
            for batch_sample in dataloader:
                # Extract batch
                batch_onehot = batch_sample['encoded_onehot'].to(device)
                # Update network
                batch_loss = train_batch(net, batch_onehot, loss_fn, optimizer)
                if ((epoch + 1) % 200 == 0):
                    print('\t Training loss (single batch):', batch_loss)
                if (batch_loss < 1):
                    if (batch_loss<loss_temp):
                        print('## New optimal model found! ##')
                        print('\t Training loss (optimal model):', batch_loss)
                        print('\t At epoch:', epoch + 1)
                        loss_temp = batch_loss
                        epochs_count = 0
                        ### Save all needed parameters (OPTIMAL)
                        # Create output dir
                        out_dir = Path(args.out_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        # Save network parameters
                        torch.save(net.state_dict(), out_dir / 'net_params_OPT.pth')
                        # Save training parameters
                        with open(out_dir / 'training_args_OPT.json', 'w') as f:
                            json.dump(vars(args), f, indent=4)
                        # Save encoder dictionary
                        with open(out_dir / 'char_to_number_OPT.json', 'w') as f:
                            json.dump(dataset.char_to_number, f, indent=4)
                        # Save decoder dictionary
                        with open(out_dir / 'number_to_char_OPT.json', 'w') as f:
                            json.dump(dataset.number_to_char, f, indent=4)
                    else:
                        epochs_count = epochs_count + 1            
        
            if ((epoch + 1) % 1000 == 0):
                print('## Intermediate model saved! ##')
                ### Save all needed parameters
                # Create output dir
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                # Save network parameters
                torch.save(net.state_dict(), out_dir / 'net_params.pth')
                # Save training parameters
                with open(out_dir / 'training_args.json', 'w') as f:
                    json.dump(vars(args), f, indent=4)
                # Save encoder dictionary
                with open(out_dir / 'char_to_number.json', 'w') as f:
                    json.dump(dataset.char_to_number, f, indent=4)
                # Save decoder dictionary
                with open(out_dir / 'number_to_char.json', 'w') as f:
                    json.dump(dataset.number_to_char, f, indent=4)
            
        else:
            print('## TRAINING STOPPED ##')
            print('\t At epoch:', epoch + 1)
            break
    
    

        
