import torch
import torch.nn as nn
import torch.nn.functional as F

class TetrisNet(nn.Module):

    def __init__(self,in_channels=4,out_dim=512):
 
        super(TetrisNet, self).__init__()
        self.feature_dim = out_dim
    
        self.conv1 = nn.Conv2d(in_channels,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.fc = nn.Linear(1*7*64,self.feature_dim)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1,1*7*64)
        x = self.fc(x)
        return x

class CartpoleNet(nn.Module):

    def __init__(self,in_channels=4,out_dim=512):

        super(CartpoleNet,self).__init__()
        self.feature_dim = out_dim

        self.fc1 = nn.Linear(in_channels,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,out_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    