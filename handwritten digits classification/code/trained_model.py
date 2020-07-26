# -*- coding: utf-8 -*-
"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering

HOMEWORK 2
Bressan Giulia
"""
import numpy as np

import scipy.io as io

import torch
import torch.nn as nn


### Define the model
class Net(nn.Module):
    
    def __init__(self, Nh1, Nh2):
        
        """
        Arguments:
        Nh1 -- scalar, neurons of the first hidden layer
        Nh2 -- scalar, neurons of the second hidden layer
        
        """
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=784, out_features=Nh1)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(Nh2, 10)
        
        self.act1 = nn.ReLU()
        
    def forward(self, x, additional_out=False):
        
        x = self.act1(self.fc1(x))
        x = self.drop1(x)
        x = self.act1(self.fc2(x))
        x = self.drop2(x)
        out = self.fc3(x)
        
        if additional_out:
            softmax = nn.functional.softmax(out.cpu(), dim=1)
            val, out_labels = torch.max(softmax.cpu(), dim=1)
            return out, val.detach().numpy(), out_labels.detach().numpy()
        
        return out

Nh1 = 64
Nh2 = 256
net = Net(Nh1, Nh2)


### Load the trained model
net = torch.load('model_ff_150ep_')


### Load Test data samples
fname ='./MNIST.mat'
data = io.loadmat(fname)

test_images = data['input_images']
test_labels = data['output_labels'].astype(int)
input_test_tensor = torch.tensor(test_images)
label_test_tensor = torch.tensor(test_labels)


### Evaluate test 

# If CUDA available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Forward pass
conc_out = torch.Tensor().float().to(device)
conc_label = torch.Tensor().long().to(device)
print('Evaluating the test set...')
for i in range(0, test_images.shape[0]):
    # Get input and output arrays
    input_test = input_test_tensor[i].float().view(-1, test_images.shape[1])
    label_test = label_test_tensor[i].long().view(-1, test_labels.shape[1]).squeeze(1)
    # Forward pass
    out, prob, label = net(input_test.to(device), additional_out=True)
    #to_print = 'Predicted ' + str(label) + ' with probability ' + str(prob)
    #print(to_print)
    # Concatenate with previous outputs
    conc_out = torch.cat([conc_out, out])
    conc_label = torch.cat([conc_label, label_test.to(device)])
    
# Predicted classes
softmax = nn.functional.softmax(conc_out.cpu(), dim=1).squeeze().detach().numpy()
val, out_labels = torch.max(conc_out.cpu(),1)
errors = conc_label.cpu()-out_labels
errors_ind = torch.nonzero(errors)

print('Accuracy: ', str((len(test_images)-len(errors_ind))/len(test_images)))