#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 20:41:21 2020

@author: hkyeremateng-boateng

Dermatologist AI - Mini Project
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, utils,datasets

class DermNet(nn.Module):
    
    def __init__(self):
        super(DermNet,self).__init__()
        
        self.conv1 = nn.Conv2d(3,32,5)
        self.conv2 = nn.Conv2d(32,74,3)
        self.conv3 = nn.Conv2d(74,24,3)
        self.pool = nn.AvgPool2d(3,3)
        self.fc = nn.Linear(1176, 1176)
        self.fc1 = nn.Linear(1176,136)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = nn.Dropout2d(p=0.25)(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = nn.Dropout2d(p=0.25)(x)
        x = self.pool(F.relu(self.conv3(x)))    
        x = nn.Dropout2d(p=0.5)(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        x = self.fc1(x)
        
        return x
DATASET_TRAIN_PATH = 'data/train'
DATASET_TEST_PATH = 'data/test'
DATASET_VALID_PATH = 'data/valid'

BATCH_SIZE = 4
# Transforms Iamges
train_transform = transforms.Compose([
        transforms.Resize(225),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
valid_transform = transforms.Compose([
        transforms.Resize(225),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_transform = transforms.Compose([
        transforms.Resize(225),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


#Loads images from directory and uses data_transform to transform the images
#ds = datasets.ImageFolder(root=DATASET_TRAIN_PATH,transform = data_transform )

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

train_datasets = {x: datasets.ImageFolder(root=DATASET_TRAIN_PATH,transform = train_transform ) for x in ['train']}

train_dataloaders = {x: torch.utils.data.DataLoader(train_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)
              for x in ['train']}


test_datasets = {x: datasets.ImageFolder(root=DATASET_TEST_PATH,transform = test_transform ) for x in ['test']}
test_loader = {x: torch.utils.data.DataLoader(test_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['test']}
class_labels = train_datasets['train'].classes
dataset_size = len(train_datasets['train'])
test =test_loader['test']
#dataiter = iter(test_loader)
#images, labels = dataiter.next()
device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu")

def test_out(model , criterion,batch_size=4):
    test_loss = torch.zeros(1)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))    
    model.cuda()
    model.eval()
    # iterate through the test dataset
    for batch_i,data in enumerate(test_loader['test']):
        
        # get sample data: images and ground truth keypoints
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass to get net output
        outputs = model(inputs)
    
        # calculate the loss
        loss = criterion(outputs, labels)
                
        # update average test loss 
        test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
        
        # get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)
        
        # compare predictions to true label
        # this creates a `correct` Tensor that holds the number of correctly classified images in a batch
        correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
        
        # calculate test accuracy for *each* object class
        # we get the scalar value of correct items for a class, by calling `correct[i].item()`
        for i in range(batch_size):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    
    print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0])) 
    return class_correct,class_total
def train_model(model, optimizer, criterion, phase='train', batch_size=20, epochs=1):
        # prepare the net for training
    model.cuda()
    if phase == 'train':
        model.train() # Set mode of Model to training
    else:
        model.eval() # Set mode of Model to evaludate
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1,epochs))
        print('--'*10)
        running_loss = 0.0
        running_corrects =0
        for batch_i, data in enumerate(train_dataloaders[phase]):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            outputs = net(inputs)
            _,preds = torch.max(outputs,1)
            # calculate the loss
            loss = criterion(outputs, labels)

            # backward pass to calculate the parameter gradients
            loss.backward()

            # update the parameters
            optimizer.step()   
            # print loss statistics
            # to convert loss into a scalar and add it to running_loss, we use .item()
            running_loss += loss.item()   
            running_corrects += torch.sum(preds == labels.data)
            if batch_i % 100 == 9: 
                print('{} Batch: {} Loss: {:.4f} '.format(phase,batch_i, running_loss/dataset_size))
                running_loss = 0.0
                
    print('Finished Training')
    return running_loss

criterion = nn.CrossEntropyLoss().to(device)
net = DermNet()
optm = optim.Adam(net.parameters(), lr=0.001)
train_model(net, optm, criterion)
test_out(net, criterion)