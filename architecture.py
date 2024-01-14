import torch 
from torch import nn
class CNNnet(nn.Module):
    def __init__(self, channel, num_classes):
        super(CNNnet, self).__init__()

        self.conv1 = nn.Conv1d(channel, 16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2, stride=2)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)

        self.act = nn.LeakyReLU(0.01)        
        self.fc = nn.Linear(6144, num_classes) 
        self.drop = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):

        x = self.conv1(x)
        # x = self.pool(x) 
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x) 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.drop(x)
        x = torch.sigmoid(x)       
        
        return x