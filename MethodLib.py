import scipy.io as scio
import os
import numpy as np
from scipy.linalg import eigh
import csv
    
def loadFile(filePath):
    ''' 
    it's a summary of 16 channels
    using trialsData[n] (len=winNum) to get window 'n', 0<=n<=winNum-1
    using trailsData[n][0] (len=1) to get the window 'n' information, 0<=n<=winNum-1
    using trialsData[n][0][m] (len=16) to get every sample data in window 'n', sample point 'm' , 0<=m<=1535
    using trialsData[n][0][m][k] (specific number) to get sample data in 'k' channel in 'm' sample point, 0<=k<=15
    using trailsLabels[n][0] to get window's label, 0<=n<=winNum
    '''
    print("loading file", filePath)
    data = scio.loadmat(filePath)
    eegData = data['subjectData'][0][0]
    subjectId = eegData[2][0]
    Fs = eegData[3][0][0]
    winNum = len(eegData[4])
    trialsData = eegData[4]
    trialsLabels = eegData[5]
    
    windowInfo = []
    for i, (window, label) in enumerate(zip(trialsData, trialsLabels)):
        # print("winNum",winNum,"shape=",window[0].shape[0])
        start = i * window[0].shape[0] / Fs
        end = (i + 1) * window[0].shape[0] / Fs
        windowInfo.append({
            'subjectId': f'{subjectId}',
            'Fs': Fs,
            'start': start,
            'end': end,
            'label': label[0],
            'filePath': filePath
        })
    return windowInfo
    # return {'id': subjectId, 'file_path': filePath, 'Fs': Fs, 'winNum': winNum}

def readDataFromPath(filePath):
    data = scio.loadmat(filePath)
    eegData = data['subjectData'][0][0]
    trialsData = eegData[4] 
    trialsLabels = eegData[5]
    return trialsData, trialsLabels

def createAnnotationFile(folderPath, annotationPath):
    '''
    Create an annotation CSV file for all .mat files in the specified folder.
    '''
    headers = ['subjectId', 'Fs', 'start', 'end', 'label', 'filePath']

    with open(annotationPath, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        # Iterate over files in folder_path
        for file_name in os.listdir(folderPath):
            if file_name.endswith('.mat'):
                file_path = os.path.join(folderPath, file_name)
                window_info = loadFile(file_path)
                
                for info in window_info:
                    writer.writerow(info)
    # annotations = []

    # for fileId, fileName in enumerate(os.listdir(folderPath)):
    #     if fileName.endswith('.mat'):
    #         filePath = os.path.join(folderPath, fileName)
    #         fileData = loadFile(filePath)
    #         annotations.append(fileData)

    # with open(annotationPath, 'w', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=['id', 'Fs', 'winNum', 'file_path'])
    #     writer.writeheader()
    #     for data in annotations:
    #         writer.writerow(data)






windowInfo = loadFile('/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/MI_BCI_Data/PAT013.mat')
print(windowInfo)
def loadAllFile(floderPath):
    pass
def preprocessing():
    pass
def ML_method():
    pass
def DL_method():
    pass
def evaluate():
    pass