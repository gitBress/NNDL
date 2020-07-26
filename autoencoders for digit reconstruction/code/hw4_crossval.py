# -*- coding: utf-8 -*-

"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering

A.A. 2019/20 

Giulia Bressan

Homework 4
 
 
"""

import torch
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np

random.seed(3)
#%% Define paths

data_root_dir = '../datasets'


#%% Create dataset

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = MNIST(data_root_dir, train=True,  download=True, transform=train_transform)
test_dataset  = MNIST(data_root_dir, train=False, download=True, transform=test_transform)

### Plot some sample
plt.close('all')
fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    img, label = random.choice(train_dataset)
    ax.imshow(img.squeeze().numpy(), cmap='gist_gray')
    ax.set_title('Label: %d' % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()


#%% Subset division

kfold_ind = list(range(0, len(train_dataset)))
train_set_1 = Subset(train_dataset, kfold_ind[int(len(kfold_ind)/3):])
train_set_2 = Subset(train_dataset, kfold_ind[:int(len(kfold_ind)/3)]+kfold_ind[2*int(len(kfold_ind)/3):])
train_set_3 = Subset(train_dataset, kfold_ind[:2*int(len(kfold_ind)/3)])
val_set_1 = Subset(train_dataset, kfold_ind[:int(len(kfold_ind)/3)])
val_set_2 = Subset(train_dataset, kfold_ind[int(len(kfold_ind)/3):2*int(len(kfold_ind)/3)])
val_set_3 = Subset(train_dataset, kfold_ind[2*int(len(kfold_ind)/3):])

train_set_list = [train_set_1, train_set_2, train_set_3]
val_set_list = [val_set_1, val_set_2, val_set_3]

#%% Define the network architecture
    
class Autoencoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        
        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x
    
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

#%% Network training
    
### Training function
def train_epoch(net, dataloader, loss_fn, optimizer):
    # Training
    net.train()
    for sample_batch in dataloader:
        # Extract data and move tensors to the selected device
        image_batch = sample_batch[0].to(device)
        # Forward pass
        output = net(image_batch)
        loss = loss_fn(output, image_batch)
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Print loss
        #print('\t partial train loss: %f' % (loss.data))


### Testing function
def test_epoch(net, dataloader, loss_fn, optimizer):
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Forward pass
            out = net(image_batch)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()]) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data
    
#%% K-fold cross-validation

# Parameters grid
params = {
    'lr':[1e-3, 1e-4, 1e-5],
    'encoded_space_dim': [2, 4, 6, 8],
    'batch_size': [256, 512, 1024],
}

loss_opt = 1000000000
lr_opt = 0
encoded_space_dim_opt = 0
batch_size_opt = 0
        
# Cross-validation
for encoded_space_dim in params['encoded_space_dim']:
    for batch_size in params['batch_size']:
        for lr in params['lr']:
            loss_temp = [] # to save the MSE for each fold
            for folds in range(3):
                print('FOLD %d/%d' % (folds + 1, 3))
                
                ### Initialize the network
                net = Autoencoder(encoded_space_dim=encoded_space_dim)
                
                ### Define dataloader
                train_dataloader = DataLoader(train_set_list[folds], batch_size=batch_size, shuffle=True)
                val_dataloader = DataLoader(val_set_list[folds], batch_size=batch_size, shuffle=False)
                
                ### Define a loss function
                loss_fn = torch.nn.MSELoss()
    
                ### Define an optimizer
                optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
                
                ### If cuda is available set the device to GPU
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
                # Move all the network parameters to the selected device (if they are already on that device nothing happens)
                net.to(device)
                
                ### Training cycle
                training = True
                num_epochs = 10
                if training:
                    for epoch in range(num_epochs):
                        print('\t ## EPOCH %d' % (epoch + 1))
                        ### Training
                        train_epoch(net, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim) 
                        ### Validation
                        val_loss = test_epoch(net, dataloader=val_dataloader, loss_fn=loss_fn, optimizer=optim) 
                        # Print Validationloss
                        print('\t\t VALIDATION - loss: %f' % (val_loss))
                
                ### Save current loss (last loss obtained)
                loss_temp.append(val_loss)
                
            ### Compare with the previous results 
            loss_avg = np.mean(np.array(loss_temp))
            if (loss_avg<=loss_opt):
                loss_opt = loss_avg
                lr_opt = lr
                encoded_space_dim_opt = encoded_space_dim
                batch_size_opt = batch_size
            
            ### Print results
            params_set = {
                'lr':[lr],
                'encoded_space_dim': [encoded_space_dim],
                'batch_size': [batch_size],
                                }
            print('SET OF PARAMETERS: ' + str(params_set))
            print('AVG LOSS FOR THIS SET: ' + str(loss_avg))
        
params_set_opt = {
        'lr':[lr_opt],
        'encoded_space_dim': [encoded_space_dim_opt],
        'batch_size': [batch_size_opt],
                        }                
print('OPTIMAL SET OF PARAMETERS: ' + str(params_set_opt))
print('AVG LOSS FOR OPTIMAL SET: ' + str(loss_opt))
