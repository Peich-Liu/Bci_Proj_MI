import os
import csv
import numpy as np

import scipy.io as scio
from scipy.linalg import eigh
import scipy.signal as signal
from mne.decoding import CSP


    
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
    return windowInfo, trialsData, trialsLabels
    # return {'id': subjectId, 'file_path': filePath, 'Fs': Fs, 'winNum': winNum}

def loadAllFile(fileFolder):
    allData = []
    allLabel = []
    for fileName in os.listdir(fileFolder):
        if fileName.endswith('.mat'):
            filePath = os.path.join(fileFolder, fileName)
            _, winData, winLabel = loadFile(filePath)
            allData.extend(winData)
            allLabel.extend(winLabel)
    # print("datalen",len(allData),"labellen",len(allLabel),"data",allData)
    return allData, allLabel
    
def readDataFromPath(filePath):
    data = scio.loadmat(filePath)
    eegData = data['subjectData'][0][0]
    trialsData = eegData[4] 
    trialsLabels = eegData[5]
    return trialsData, trialsLabels

def createOriAnnotationFile(folderPath, annotationPath):
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
                window_info, _, _ = loadFile(file_path)
                
                for info in window_info:
                    writer.writerow(info)

def createAnnotationFromInfo(winInfo):
    pass

def bandPass(data, lowCut, highCut, fs, order=5):
    nyq = 0.5 * fs
    low = lowCut / nyq
    high = highCut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def loadFilterFile(filePath):
    '''
    use filteredData[n] to get different window
    use filteredData[n][m] to get different sample point
    use filteredLabel[n][0] to get window's label
    '''
    data = scio.loadmat(filePath)
    filteredData = data['filteredSignal']
    filteredLabel = data['filteredLabel']
    subjectId = data['subjectId']
    Fs = data['Fs'][0][0]
    filePathInMat = data['filePath']
    
    windowInfo = []
    for i, (window, label) in enumerate(zip(filteredData, filteredLabel)):
        # print("winNum",winNum,"shape=",window[0].shape[0])
        start = i * window.shape[0] / Fs
        end = (i + 1) * window.shape[0] / Fs
        # print("shape",start)
        windowInfo.append({
            'subjectId': f'{subjectId}',
            'Fs': Fs,
            'start': start,
            'end': end,
            'label': label[0],
            'filePath': filePath
        })
    return filteredData, filteredLabel, windowInfo
# a,b = loadFilterFile('/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/resultStore/Standard/afterFilter/PAT013.mat')
# data1 = a[0]
# data2 = a[0][0]
# label1 = b[0]
# label2 = b[0][0]
# print(data2[0])
def loadAllFilterFile(fileFolder):
    allData = []
    for fileName in os.listdir(fileFolder):
        if fileName.endswith('.mat'):
            filePath = os.path.join(fileFolder, fileName)
            winData = loadFilterFile(filePath)
            allData.extend(winData)
    return allData

def filterSignal(filePath, lowCut, highCut, Fs, mat_filename):
    filteredSignal = []
    filteredLabel = []
    winInfo, winData, winLabel = loadFile(filePath)
    subjectIdStore = winInfo[0]['subjectId']
    FsStore = winInfo[0]['Fs']
    filePathStore = winInfo[0]['filePath']
    for win, label in zip(winData, winLabel):
        windowOri = win[0]
        if windowOri.shape[1] == 16:
            filteredWinow = bandPass(windowOri, lowCut, highCut,Fs)
            filteredSignal.append(filteredWinow)   
            filteredLabel.append(label)
    
    scio.savemat(mat_filename, {
                            'filteredSignal': filteredSignal,
                            'filteredLabel':filteredLabel,
                            'subjectId':subjectIdStore,
                            'Fs':FsStore,
                            'filePath':filePathStore
                            })
    # return filteredInfo, filteredSignal, filteredLabel

def TrainCsp(filePath):
    filteredData, filteredLabel, _ = loadFilterFile(filePath)
    reshaped_data = filteredData.reshape(filteredData.shape[0], -1)
    test = reshaped_data[0]
    transposed_data = np.transpose(filteredData, (0, 2, 1))
    
    class1_data = transposed_data[filteredLabel.flatten() == 0, :]
    class2_data = transposed_data[filteredLabel.flatten() == 1, :]
    min_length = min(len(class1_data), len(class2_data))
    class1_data = class1_data[:min_length]
    class2_data = class2_data[:min_length]
    csp = CSP(n_components=4)
    print(class1_data.shape)
    csp.fit(class1_data, class2_data)
    
    csp_features = csp.transform(reshaped_data)
    print(csp_features)
    print("123")
    
TrainCsp('/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/resultStore/Standard/afterFilter/PAT013.mat')
def ML_method():
    pass
def DL_method():
    pass
def evaluate():
    pass