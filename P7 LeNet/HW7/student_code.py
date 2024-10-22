# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1) #Convulution Layer 1
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2) #pooling layer 1
        
        self.conv2=nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1) #Conv layer 2
        
        self.flatten= nn.Flatten() #Flatten Layer
        
        self.fc1 = nn.Linear(400, 256) #Linear Layer w/ output dimension 256
        
        self.fc2 = nn.Linear (256, 128) #Linear Layer w/ output dimension 128
        
        self.fc3 = nn.Linear(128, num_classes)
  

    def forward(self, x):
        shape_dict = {}
        
        #conv. layer 1, relu activation, pooling 
        x=self.pool(nn.functional.relu(self.conv1(x)))
        shape_dict[1]=x.size()
        
        #conv. layer 2, relu activation, pooling 
        x=self.pool(nn.functional.relu(self.conv2(x)))
        shape_dict[2]=x.size()
        
        
        #FLatten Layer
        x = x.view(-1, 400)
        shape_dict[3] = x.size()
        
        
        #Linear layer 1
        x = nn.functional.relu(self.fc1(x))
        shape_dict[4] = x.size()
        
        
        #Linear layer 2
        x = nn.functional.relu(self.fc2(x))
        shape_dict[5] = x.size()
        
        #Linear layer 3
        x =self.fc3(x)
        shape_dict[6] = x.size()
        
        return x, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            model_params += torch.prod(torch.tensor(param.size()))
            
    model_params = model_params/1e6

    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
