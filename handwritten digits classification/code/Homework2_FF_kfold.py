# -*- coding: utf-8 -*-
"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering

HOMEWORK 2
Bressan Giulia
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import random
from sklearn.model_selection import KFold

import scipy.io as io
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#%% Training and test data

# Set random seed
np.random.seed(3)

# Load data from .mat file
fname='./MNIST.mat'
data= io.loadmat(fname)

input_images_orig = data['input_images']
output_labels_orig = data['output_labels'].astype(int)

len_dataset = len(input_images_orig)

input_images = input_images_orig.astype(np.float32)
output_labels = output_labels_orig.astype(np.int64)

# Example of pictures and their label
def plot_example(X, y):
    for i, (img, y) in enumerate(zip(input_images[:5].reshape(5, 28, 28), output_labels[:5])):
        img = img.transpose()
        img = np.flip(img)
        plt.subplot(151 + i)
        plt.imshow(img)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.xticks([])
        plt.yticks([])
        plt.title(y)
        
plot_example(input_images, output_labels)

# Train and Test split
xTrain, xTest, yTrain, yTest = train_test_split(input_images, output_labels, test_size = 0.2, random_state = 0)
num_train_points = xTrain.shape[0]
num_test_points = xTest.shape[0]

print ("number of training examples = " + str(xTrain.shape[0]))
print ("number of test examples = " + str(xTest.shape[0]))
print ("xTrain shape: " + str(xTrain.shape))
print ("yTrain shape: " + str(yTrain.shape))
print ("xTest shape: " + str(xTest.shape))
print ("yTest shape: " + str(yTest.shape))


#%% Neural Network

### Define the network class
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
            return out, val.numpy(), out_labels.numpy()
        
        return out

#%%
### Cross Validation with grid search

# Define the split - into 3 folds
xTrain_small = xTrain[:10000].copy()
yTrain_small = yTrain[:10000].copy()
result = zip(xTrain_small, yTrain_small)
resultList = list(result)
random.seed(13)
random.shuffle(resultList)
xTrain_s, yTrain_s =  zip(*resultList)

kf = KFold(n_splits=3, shuffle=False)
print(kf)

# Returns the number of splitting iterations in the cross-validator
X = np.asarray(xTrain_s)
y = np.asarray(yTrain_s)
kf.get_n_splits(X)

# Print the result
for train_index, test_index in kf.split(X):
    print('TRAIN:', train_index, 'TEST:', test_index)
    xTrain_kf, xTest_kf = X[train_index], X[test_index]
    yTrain_kf, yTest_kf = y[train_index], y[test_index] 

# Parameters grid
params = {
    'lr':[1e-2, 2e-2],
    'max_epochs': [10],
    'Nh1': [32, 64, 128],
    'Nh2': [64, 128, 256]
}
loss_opt = 1000000000
Nh1_opt = 0
Nh2_opt = 0
lr_opt = 0
num_epochs_opt = 0

### Define the loss function (the most used are already implemented in pytorch, see the doc!)
loss_fn = nn.CrossEntropyLoss()

### If CUDA available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for lr in params['lr']:
    for max_epochs in params['max_epochs']:
        for Nh1 in params['Nh1']:
            for Nh2 in params['Nh2']:
                loss_temp = [] # to save the MSE for each fold
                fold = 0 # to count the folds
                for train_index, test_index in kf.split(X):
                    fold = fold + 1
                    print('Fold', fold)
                    ### Initialize the network
                    net = Net(Nh1, Nh2)
                    net.to(device)
                    ### Define an optimizer
                    lr_init = lr
                    optimizer = optim.Adam(net.parameters(), lr=lr_init, weight_decay=5e-4)
                    #optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.5)
    
                    # Obtain the folds
                    X_train_kf, X_test_kf = X[train_index], X[test_index]
                    y_train_kf, y_test_kf = y[train_index], y[test_index]    
                    
                    for num_epoch in range(max_epochs):
    
                        print('\t Epoch', num_epoch + 1)
                        # Training
                        net.train() # Training mode (e.g. enable dropout)
                        # Eventually clear previous recorded gradients
                        optimizer.zero_grad()
                        conc_out = torch.Tensor().float().to(device)
                        conc_label = torch.Tensor().long().to(device)
                        for i in range(0, X_train_kf.shape[0]):
                            input_train = torch.tensor(X_train_kf[i]).float().view(-1, X_train_kf.shape[1])
                            label_train = torch.tensor(y_train_kf[i]).long().view(-1, y_train_kf.shape[1]).squeeze(1)
                            # Forward pass
                            out = net(input_train.to(device))
                            conc_out = torch.cat([conc_out, out])
                            conc_label = torch.cat([conc_label, label_train.to(device)])
                        # Evaluate loss
                        loss = loss_fn(conc_out, conc_label)
                        # Backward pass
                        loss.backward()
                        # Update
                        optimizer.step()
                        # Print loss
                        print('\t\t Training loss ():', float(loss.data))
                            
                        # Validation
                        net.eval() # Evaluation mode (e.g. disable dropout)
                        with torch.no_grad(): # No need to track the gradients
                            conc_out = torch.Tensor().float().to(device)
                            conc_label = torch.Tensor().long().to(device)
                            for i in range(0, X_test_kf.shape[0]):
                                # Get input and output arrays
                                input_test = torch.tensor(X_test_kf[i]).float().view(-1, X_train_kf.shape[1])
                                label_test = torch.tensor(y_test_kf[i]).long().view(-1, y_train_kf.shape[1]).squeeze(1)
                                # Forward pass
                                out = net(input_test.to(device))
                                # Concatenate with previous outputs
                                conc_out = torch.cat([conc_out, out])
                                conc_label = torch.cat([conc_label, label_test.to(device)])
                            # Evaluate global loss
                            test_loss = loss_fn(conc_out, conc_label)
                            # Print loss
                            print('\t\t Validation loss:', float(test_loss.data))
                            
                        if num_epoch == max_epochs-1:
                            loss_temp.append(float(test_loss.data))
                    del net
                
                # Compare with the previous results 
                loss_avg = np.mean(np.array(loss_temp))
                if (loss_avg<=loss_opt):
                    loss_opt = loss_avg
                    Nh1_opt = Nh1
                    Nh2_opt = Nh2
                    lr_opt = lr_init
                    num_epochs_opt = max_epochs
                
                params_set = {
                    'lr':[lr_init],
                    'max_epochs': [max_epochs],
                    'Nh1': [Nh1],
                    'Nh2': [Nh2]
                }
                print('SET OF PARAMETERS: ' + str(params_set))
                print('AVG LOSS FOR THIS SET: ' + str(loss_avg))

params_set_opt = {
    'lr':[lr_opt],
    'max_epochs': [num_epochs_opt],
    'Nh1': [Nh1_opt],
    'Nh2': [Nh2_opt]
}                
print('OPTIMAL SET OF PARAMETERS: Nh1: '+ str(params_set_opt))
print('AVG LOSS FOR OPTIMAL SET: ' + str(loss_opt))
#%%

### If CUDA available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Transform arrays in tensors
input_train_tensor = torch.tensor(xTrain)
label_train_tensor = torch.tensor(yTrain)
input_test_tensor = torch.tensor(xTest)
label_test_tensor = torch.tensor(yTest)

### Initialize the network with the optimal parameters
Nh1 = params_set_opt['Nh1'][0]
Nh2 = params_set_opt['Nh2'][0]
net_opt = Net(Nh1, Nh2)
net_opt.to(device)

### Define the loss function (the most used are already implemented in pytorch, see the doc!)
loss_fn = nn.CrossEntropyLoss()

### Define an optimizer
lr = params_set_opt['lr'][0]
optimizer = torch.optim.Adam(net_opt.parameters(), lr=lr, weight_decay=5e-4)

### Training
train_loss_log = []
test_loss_log = []
max_epochs = 150

for num_epoch in range(max_epochs):
    
    print('Epoch', num_epoch + 1)
    # Training
    net_opt.train() # Training mode (e.g. enable dropout)
    # Eventually clear previous recorded gradients
    optimizer.zero_grad()
    conc_out = torch.Tensor().float().to(device)
    conc_label = torch.Tensor().long().to(device)
    for i in range(0, xTrain.shape[0]):
        input_train = input_train_tensor[i].float().view(-1, xTrain.shape[1])
        label_train = label_train_tensor[i].long().view(-1, yTrain.shape[1]).squeeze(1)
        # Forward pass
        out = net_opt(input_train.to(device))
        conc_out = torch.cat([conc_out, out])
        conc_label = torch.cat([conc_label, label_train.to(device)])
    # Evaluate loss
    loss = loss_fn(conc_out, conc_label)
    # Backward pass
    loss.backward()
    # Update
    optimizer.step()
    # Print loss
    print('\t Training loss ():', float(loss.data))
        
    # Test
    net_opt.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float().to(device)
        conc_label = torch.Tensor().long().to(device)
        for i in range(0,  xTest.shape[0]):
            # Get input and output arrays
            input_test = input_test_tensor[i].float().view(-1, xTest.shape[1])
            label_test = label_test_tensor[i].long().view(-1, yTest.shape[1]).squeeze(1)
            # Forward pass
            out = net_opt(input_test.to(device))
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out])
            conc_label = torch.cat([conc_label, label_test.to(device)])
        # Evaluate global loss
        test_loss = loss_fn(conc_out, conc_label)
        # Print loss
        print('\t Validation loss:', float(test_loss.data))
        
    # Log
    train_loss_log.append(float(loss.data))
    test_loss_log.append(float(test_loss.data))
    
    
#%%

# Predicted classes
softmax = nn.functional.softmax(conc_out.cpu(), dim=1).squeeze()
val, out_labels = torch.max(softmax.cpu(),1)
errors = conc_label.cpu()-out_labels
errors_ind = torch.nonzero(errors)

print('Real classes: ', conc_label.cpu().numpy())
print('Predicted classes: ', out_labels.cpu().numpy())
print('Number of errors (out of 12000 test samples): ', len(errors_ind))

# Example of pictures and their label
rand_ind = np.random.choice(len(errors_ind), 5)
wrong_img = xTest[errors_ind[rand_ind],:].reshape(5,784)
wrong_label =  yTest[errors_ind[rand_ind]].reshape(5,1)
out_labels_set = out_labels[errors_ind[rand_ind]].numpy()
def plot_example_wrong(X, y_t, y_o):
    for i, (img, y_true, y_out) in enumerate(zip(wrong_img.reshape(5, 28, 28), wrong_label, out_labels_set)):
        img = img.transpose()
        img = np.flip(img)
        plt.subplot(151 + i)
        plt.imshow(img)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.xticks([])
        plt.yticks([])
        title = str(y_true) + ' as ' + str(y_out)
        plt.title(title)

plot_example_wrong(wrong_img, wrong_label, out_labels_set)

#%%
        
# Plot losses
plt.close('all')
plt.figure(figsize=(12,8))
plt.semilogy(train_loss_log, label='Train loss')
plt.semilogy(test_loss_log, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%%

# Confusion matrix, classification report, accuracy score
#class_report = classification_report(conc_label.cpu(), out_labels)
#print(class_report)
accuracy = accuracy_score(conc_label.cpu(), out_labels.cpu())
print('Accuracy: ', accuracy)
conf_matrix = confusion_matrix(conc_label.cpu(), out_labels.cpu())
print('Confusion matrix: \n', conf_matrix)

#%%

# Save the network
#torch.save(net_opt, 'model_ff_150ep_')

#%%

# Visualize the receptive field of hidden neurons

trained_weights_h1 = net_opt.fc1.weight
weights1 = trained_weights_h1.cpu().detach().numpy()
trained_weights_h2 = net_opt.fc2.weight
weights2 = trained_weights_h2.cpu().detach().numpy()
trained_weights_h3 = net_opt.fc3.weight
weights3 = trained_weights_h3.cpu().detach().numpy()

# Hidden Layer 1
rf1 = weights1

# Show random set of receptive fields
rand_ind = np.random.choice(64, 10)
rf1_subset = rf1[rand_ind,:]

fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
counter = 0;
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(rf1_subset[counter].reshape(28,28))
        counter = counter + 1

# Hidden Layer 2
rf2 = np.matmul(weights2, rf1)

# Show random set of receptive fields
rand_ind = np.random.choice(256, 10)
rf2_subset = rf2[rand_ind,:]

fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
counter = 0;
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(rf2_subset[counter].reshape(28,28))
        counter = counter + 1

# Output layer
rf3 = np.matmul(weights3, rf2)

# Show random set of receptive fields
fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
counter = 0;
for i in range(2):
    for j in range(5):
        ax[i, j].imshow(rf3[counter].reshape(28,28))
        counter = counter + 1

