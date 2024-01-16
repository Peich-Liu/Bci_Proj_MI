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
        super(NeuralNetworkStream, self).__init__()
        self.pre_conv0 = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2 ,groups=n_channels)
        self.pre_conv1 = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2 ,groups=n_channels)
        self.pre_conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2 ,groups=n_channels)
        self.pre_conv3 = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2 ,groups=n_channels)
        
        self.sequence1 =  self.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01)
        )
        
        self.sequence2 =  self.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01)
        )
                
        self.sequence3 =  self.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01)
        )


    def forward(self, x):
        x0 = self.pre_conv0(x)
        pre_x1 = self.pre_conv1(x0)
        pre_x2 = self.pre_conv2(pre_x1)
        pre_x3 = self.pre_conv3(pre_x2)
        
        x1 = self.sequence0(pre_x1).mean(dim=(-1))
        x2 = self.sequence1(pre_x2).mean(dim=(-1))
        x3 = self.sequence1(pre_x3).mean(dim=(-1))
        
        final = torch.cat([x1,x2,x3],1)
        
        return final
    
class MultiStreamEEGNet(nn.Module):
    def __init__(self, num_bands, input_channels, sample_length, num_classes):
        super(MultiStreamEEGNet, self).__init__()

        self.streams = nn.ModuleList([
            NeuralNetworkStream(input_channels, sample_length) for _ in range(num_bands)
        ])

        self.classifier = nn.Linear(64 * sample_length // 2 * num_bands, num_classes)

    def forward(self, mu_input, beta_input):
        mu_output = self.streams[0](mu_input)
        beta_output = self.streams[1](beta_input)
        combined = torch.cat([mu_output, beta_output], dim=-1)
        logits = self.classifier(combined)
        return logits
    # def forward(self, *inputs):

    #     outputs = []
    #     for i, input in enumerate(inputs):
    #         outputs.append(self.streams[i](input))

    #     combined = torch.cat(outputs, dim=-1)
    #     logits = self.classifier(combined)
    #     return logits
