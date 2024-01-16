import torch 
from torch import nn
class CNNnet(nn.Module):
    def __init__(self, channel, num_classes):
        super(CNNnet, self).__init__()
        self.time_conv = nn.Conv1d(in_channels=channel, out_channels=64, kernel_size=3, padding=1)
        self.space_conv = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=channel, groups=64, padding=0)
        
        self.conv1 = nn.Conv1d(channel, 32, kernel_size=85, stride=1) #16
        self.conv2 = nn.Conv1d(32, 32, kernel_size=10, stride=1) #16*32
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)

        self.act = nn.LeakyReLU(0.01)        
        self.fc1 = nn.Linear(97344, num_classes) 

        self.drop = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):

        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        x = self.time_conv(x)
        x = self.space_conv(x)
        
        x = self.act(x)
        # x = self.pool(x) 
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.drop(x)
        x = torch.sigmoid(x)       
        
        return x