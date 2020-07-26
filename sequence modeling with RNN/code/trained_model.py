# -*- coding: utf-8 -*-

import json
import torch
import argparse
from network import Network
from shakespeare_dataset import encode_text, decode_text, create_one_hot_matrix
from pathlib import Path
from torch import nn


##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Generate sonnet starting from a given text')

parser.add_argument('--seed', type=str, default='o, romeo, romeo!', help='Initial context')
parser.add_argument('--length', type=str, default='500', help='Number of produced characters')


##############################
##############################
##############################

##############################
##############################
## SAMPLING FROM SOFTMAX
##############################

torch.manual_seed(3)

def sample(net_out, temperature):
    EPSILON = 10e-16 # to avoid taking the log of zero
    out_temp = (net_out+EPSILON)/temperature
    preds = nn.functional.softmax(out_temp, dim=1)
    preds_tens = torch.as_tensor(preds).float()
    probas = torch.multinomial(preds_tens, 1)
    return probas.item()

##############################
##############################
##############################

if __name__ == '__main__':
    
    ### Parse input arguments
    args = parser.parse_args()
    
    #%% Load training parameters
    model_dir = Path('model')
    print ('Loading model from: %s' % model_dir)
    training_args = json.load(open(model_dir / 'training_args_OPT.json'))
      
    #%% Load encoder and decoder dictionaries
    number_to_char = json.load(open(model_dir / 'number_to_char_OPT.json'))
    char_to_number = json.load(open(model_dir / 'char_to_number_OPT.json'))
        
    #%% Initialize network
    net = Network(input_size=training_args['alphabet_len'], 
                  hidden_units=training_args['hidden_units'], 
                  layers_num=training_args['layers_num'])
        
    #%% Load network trained parameters
    net.load_state_dict(torch.load(model_dir / 'net_params_OPT.pth', map_location='cpu'))
    net.eval() # Evaluation mode (e.g. disable dropout)
    
    #%% Define temperature
    temperature = 0.1
    
    #%% Find initial state of the RNN
    with torch.no_grad():
        # Encode seed
        seed_encoded = encode_text(char_to_number, args.seed)
        # One hot matrix
        seed_onehot = create_one_hot_matrix(seed_encoded, training_args['alphabet_len'])
        # To tensor
        seed_onehot = torch.tensor(seed_onehot).float()
        # Add batch axis
        seed_onehot = seed_onehot.unsqueeze(0)
        # Forward pass
        net_out, net_state = net(seed_onehot)
        # Sample from softmax last output index
        next_char_encoded = sample(net_out[:, -1, :], temperature)
        # Print the seed letters
        print(args.seed, end='', flush=True)
        print(number_to_char[str(next_char_encoded)])
        
    #%% Generate sonnet
    new_line_count = 0
    tot_char_count = 0
    while True:
        with torch.no_grad(): # No need to track the gradients
            # The new network input is the one hot encoding of the last chosen letter
            net_input = create_one_hot_matrix([next_char_encoded], training_args['alphabet_len'])
            net_input = torch.tensor(net_input).float()
            net_input = net_input.unsqueeze(0)
            # Forward pass
            net_out, net_state = net(net_input, net_state)
            # Sample from softmax last output index
            next_char_encoded = sample(net_out[:, -1, :], temperature)
            # Decode the letter
            next_char = number_to_char[str(next_char_encoded)]
            print(next_char, end='', flush=True)
            # Count total letters
            tot_char_count += 1
            # Break if n letters
            if tot_char_count > int(args.length):
                break
        
        
        
        
        
        
        
        
        
        
        
        
