# -*- coding: utf-8 -*-
"""
This project is based on the tutorial found at tutorial:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

This project implements an image classifier using a convolutional neural network
in PyTorch. Below are a list of items I wish to implement beyond that is found
in the tutorial along with a list of things I have succesfully added.

@author: bmcma

Additions I have made to the code provided in the tutorial:
-custom loss allows for easy plotting of loss history after training
-implemented a learning rate schedule
-network moved to GPU
-training for 100 epochs
-made network deeper
-implemented dropout

Future work for project:
TODO: get above 75% accuracy accross all classes
TODO: for a given number of units compare effect of depth on performance
TODO: try a different activation function
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

##############################
# STEP 1: LOAD DATA
##############################
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image


def imshow(img):
    '''
    plots image data

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

##############################
# STEP 2: DEFINE NETWORK
##############################
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
net.to(device)   # moves network onto GPU
##############################
# STEP 3: DEFINE LOSS
##############################
class MyLoss(nn.CrossEntropyLoss):
    '''Class built on top of nn.CrossEntropyLoss

    Attributes
    _loss_history: list of training loss history

    Methods
    -------
    record() -- adds current loss to this history
    plot()-- plots the loss history
    '''
    def __init__(self):
        super().__init__()
        self._loss_history = []
        
    def forward(self, inpt, target):
        '''computes loss and appends it to history.'''
        loss_value = super().forward(inpt, target)
        self._loss_history.append(loss_value.item())
        return loss_value

    def plot(self):
        '''plots training loss history.'''
        plt.figure()    # open a new figure to plot loss
        plt.plot(np.array(self._loss_history)[::50])
        plt.xlabel("Training Iteration")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Training Loss")


criterion = MyLoss()   # attempt to use my custom loss class

#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=True)
##############################
# STEP 4: TRAIN NETWORK
##############################
training_start_time = time.time()
NUM_EPOCHS = 64
for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs = inputs.to(device)  # moves data onto GPU
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    scheduler.step()

training_end_time = time.time()
training_time = training_end_time - training_start_time
print('Finished Training')
print("Training took", training_time, "seconds")
criterion.plot()

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
##############################
# STEP 5: TEST NETWORK
##############################
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
plt.figure()   # opens a new figure
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# how network performs on entire dataset
correct = 0
total = 0
with torch.no_grad():
    for (images, labels) in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# get accuracies of each class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for (images, labels) in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    