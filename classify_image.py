# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:35:33 2021

@author: bmcma
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image


# define network
class Net(nn.Module):
    '''A class to represent a convolutional neural network.

    ...

    Attributes
    ---------

    Methods
    -------
    forward(torch tensor):
        forward propogates x
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 250, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(250, 180, 3)
        self.conv3 = nn.Conv2d(180, 80, 3)
        self.fc1 = nn.Linear(80 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.25)  # probability of dropping unit

    def forward(self, x):
        '''
        defines forward propogation

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        x : TYPE
            DESCRIPTION.

        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 80 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
device = torch.device("cuda:0")
#net.to(device)   # moves network onto GPU

# load state dict
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
## create GUI
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename() # gets file_path
print("file path selected:", file_path)

user_im = Image.open(file_path)
resize_transform = transforms.Resize((32,32))
user_im = resize_transform.forward(user_im)
tensor_transform = transforms.ToTensor()
user_im = tensor_transform(user_im)
user_im = torch.unsqueeze(user_im[:3], 0)   # only use first 3 channels in case of .png file
user_out = net(user_im)
user_out = torch.argmax(user_out).item()
print("This image is predicted to be a:", classes[user_out])