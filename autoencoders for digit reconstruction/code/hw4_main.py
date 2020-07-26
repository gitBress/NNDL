# -*- coding: utf-8 -*-

"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering

A.A. 2019/20 

Giulia Bressan

Homework 4
 
"""

import os
import torch
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np


#%% Define paths

data_root_dir = '../datasets'


#%% Create dataset

def noisy(image, std): 
    row,col= image.shape
    mean = 0
    std = std
    gauss = np.random.normal(mean,std,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy


train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.33, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.RandomApply([
        transforms.Lambda(lambda x: noisy(x.squeeze().numpy(), 0.1)), 
        transforms.Lambda(lambda x: transforms.functional.to_tensor(x).float())
        ], 
        p=0.11),
    transforms.RandomApply([
        transforms.Lambda(lambda x: noisy(x.squeeze().numpy(), 0.05)), 
        transforms.Lambda(lambda x: transforms.functional.to_tensor(x).float())
        ], 
        p=0.11),
    transforms.RandomApply([
        transforms.Lambda(lambda x: noisy(x.squeeze().numpy(), 0.03)), 
        transforms.Lambda(lambda x: transforms.functional.to_tensor(x).float())
        ], 
        p=0.11)
])

train_dataset = MNIST(data_root_dir, train=True,  download=True, transform=train_transform)
test_dataset  = MNIST(data_root_dir, train=False, download=True, transform=test_transform)
print(train_dataset)
print(test_dataset)
### Plot some sample
plt.close('all')
fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    img, label = random.choice(test_dataset)
    ax.imshow(img.squeeze().numpy(), cmap='gist_gray')
    ax.set_title('Label: %d' % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.pause(0.1)

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

### Initialize the network
encoded_space_dim = 8
net = Autoencoder(encoded_space_dim=encoded_space_dim)


### Some examples
# Take an input image (remember to add the batch dimension)
img = test_dataset[0][0].unsqueeze(0)
print('Original image shape:', img.shape)
# Encode the image
img_enc = net.encode(img)
print('Encoded image shape:', img_enc.shape)
# Decode the image
dec_img = net.decode(img_enc)
print('Decoded image shape:', dec_img.shape)


#%% Prepare training

### Define dataloader
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

### Define a loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer
lr = 1e-3 # Learning rate
optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

### If cuda is available set the device to GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# Move all the network parameters to the selected device (if they are already on that device nothing happens)
net.to(device)


#%% Network training

### Training function
def train_epoch(net, dataloader, loss_fn, optimizer):
    # Training
    net.train()
    loss_epoch = []
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
        print('\t partial train loss: %f' % (loss.data))
        loss_epoch.append(loss.data)
    return torch.mean(torch.stack(loss_epoch))


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
            out = net(image_batch.float())
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()]) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


### Training cycle
train_loss_list = []
val_loss_list = []
training = True
num_epochs = 1000

epochs_count = 0
loss_temp = 1000000
if training:
    for epoch in range(num_epochs):
        if (epochs_count<20):
            print('EPOCH %d/%d' % (epoch + 1, num_epochs))
            ### Training
            train_loss = train_epoch(net, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim) 
            ### Validation
            val_loss = test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim) 
            # Print Validationloss
            print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, num_epochs, val_loss))
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            if (val_loss<loss_temp):
                loss_temp = val_loss
                epochs_count = 0
            else:
                epochs_count = epochs_count + 1
    
            ### Plot progress
            img = test_dataset[0][0].unsqueeze(0).to(device)
            net.eval()
            with torch.no_grad():
                rec_img  = net(img)
            fig, axs = plt.subplots(1, 2, figsize=(12,6))
            axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[0].set_title('Original image')
            axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[1].set_title('Reconstructed image (EPOCH %d)' % (epoch + 1))
            plt.tight_layout()
            plt.pause(0.1)
            # Save figures
            os.makedirs('autoencoder_progress_%d_features' % encoded_space_dim, exist_ok=True)
            plt.savefig('autoencoder_progress_%d_features/epoch_%d.png' % (encoded_space_dim, epoch + 1))
            plt.show()
            plt.close()
    
            # Save network parameters
            torch.save(net.state_dict(), 'net_params.pth')
        
        else:
            print('## TRAINING STOPPED ##')
            print('\t At epoch:', epoch + 1)
            break
        


#%% Testing

### Plot tests
num_test = 0
img = test_dataset[num_test][0].unsqueeze(0).to(device)
label = test_dataset[num_test][1]
net.eval()
with torch.no_grad():
    rec_img  = net(img)
fig, axs = plt.subplots(1, 2, figsize=(12,6))
axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
axs[0].set_title('Original image, Label: %d' % label)
axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
axs[1].set_title('Reconstructed image (final)')
plt.tight_layout()
plt.pause(0.1)

#%% Final loss

### Evaluate test loss
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
test_loss_final = test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim) 
print('\n\n\t TEST - loss: %f\n\n' % (test_loss_final))

#%% Losses

### Plot losses
plt.close('all')
plt.figure(figsize=(12,8))
plt.semilogy(train_loss_list, label='Train loss')
plt.semilogy(val_loss_list, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
#%% Network analysis


# Load network parameters    
net.load_state_dict(torch.load('net_params.pth', map_location='cpu'))

### Get the encoded representation of the test samples
encoded_samples = []
for sample in tqdm(test_dataset):
    img = sample[0].unsqueeze(0).to(device)
    label = sample[1]
    # Encode image
    net.eval()
    with torch.no_grad():
        encoded_img  = net.encode(img)
    # Append to list
    encoded_samples.append((encoded_img.flatten().cpu().numpy(), label))
    

### Visualize encoded space
color_map = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
        5: '#8c564b',
        6: '#e377c2',
        7: '#7f7f7f',
        8: '#bcbd22',
        9: '#17becf'
        }
    
# Plot just 1k points
encoded_samples_reduced = random.sample(encoded_samples, 1000)
plt.figure(figsize=(12,10))
for enc_sample, label in tqdm(encoded_samples_reduced):
    plt.plot(enc_sample[0], enc_sample[1], marker='.', color=color_map[label])
plt.grid(True)
plt.legend([plt.Line2D([0], [0], ls='', marker='.', color=c, label=l) for l, c in color_map.items()], color_map.keys())
plt.tight_layout()
plt.show()
        
if encoded_space_dim == 2:
    # Generate samples

    encoded_value = torch.tensor([8.0, -12.0]).float().unsqueeze(0)

    net.eval()
    with torch.no_grad():
        new_img  = net.decode(encoded_value)

    plt.figure(figsize=(12,10))
    plt.imshow(new_img.squeeze().numpy(), cmap='gist_gray')
    plt.show()
