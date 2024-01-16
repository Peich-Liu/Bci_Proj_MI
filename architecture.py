import torch 
from torch import nn
import torch.nn.functional as F
class DLSignal():
    def __init__(self, mu, beta, label):
        self.mu = mu
        self.beta = beta
        self.labels = label
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        muSignal = self.mu[idx]
        betaSignal = self.beta[idx]
        labelOutput = self.labels[idx]
        return muSignal, betaSignal, labelOutput

class NeuralNetworkStream(nn.Module):
    def __init__(self, n_channels, sample_length):
    # def __init__(self, n_channels):
        super(NeuralNetworkStream, self).__init__()
        self.conv0 = nn.Conv1d(n_channels, 32, kernel_size=2, stride=2 ,groups=n_channels)
        self.conv1 = nn.Conv1d(32, 32, kernel_size=4,groups=1)
        self.bn = nn.BatchNorm1d(32)
        self.act = nn.LeakyReLU(0.01)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        



    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.flatten(x)

        return x

    
class MultiStreamEEGNet(nn.Module):
    def __init__(self, num_bands, input_channels, sample_length, num_classes):
        super(MultiStreamEEGNet, self).__init__()

        self.streams = nn.ModuleList([
            NeuralNetworkStream(input_channels, sample_length) for _ in range(num_bands)
        ])
        self.classifier = nn.Linear(24448, num_classes)

        

    def forward(self, mu_input, beta_input):
        mu_output = self.streams[0](mu_input)
        beta_output = self.streams[1](beta_input)
        combined = torch.cat([mu_output, beta_output], dim=-1)
        # print(combined.shape)
        # logits = nn.functional.log_softmax(self.classifier(combined),dim=1)
        x = self.classifier(combined) 
        x = x = torch.sigmoid(x)  
        return x
