import scipy.io as scio
import os

class dataParameters:
    subject = []
    winLen = 1536 #it's a fix number: len(eegData[4][0][0])
    channelLen = 16 #it's a fix number: len(eegData[4][0][0][0])
    Fs = 256

class filterParameter:
    lowCut = 0.8
    highCut = 30.0
    order = 5
    Fs = 256

