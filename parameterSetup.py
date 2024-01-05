import scipy.io as scio
import os
class dataParameters:
    
    winLen = 1536 #it's a fix number: len(eegData[4][0][0])
    channelLen = 16 #it's a fix number: len(eegData[4][0][0][0])
    
def loadfile(file_path):
    ''' 
    it's a summary of 16 channels
    using trialsData[n] to get window, 0<=n<=winNum-1
    using trialsData[n][0][m] to get different channel in one sample point, 0<=m<=1535
    using trialsData[n][0][m][k] to get different point data in different channels, 0<=k<=15
    using trailsLabels[n][0] to get window's label, 0<=n<=winNum
    '''
    data = scio.loadmat(file_path)
    eegData = data['subjectData'][0][0]
    Fs = eegData[3]
    winNum = len(eegData[4])
    trialsData = eegData[4] 
    trialsLabels = eegData[5]
    return Fs, winNum, trialsData, trialsLabels