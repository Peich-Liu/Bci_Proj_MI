import os
import csv
import mne
import numpy as np
import pandas as pd 

import scipy.io as scio
from scipy.linalg import eigh
import scipy.signal as signal
from mne.decoding import CSP
from parameterSetup import dataParameters

    
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
    convertTrailData = []
    for i, (window, label) in enumerate(zip(trialsData, trialsLabels)):
        # print("winNum",winNum,"shape=",window[0].shape[0])
        debug = window[0]
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
        transWindowData = np.array(window[0]).T
        convertTrailData.append(transWindowData)
    MneTrailData = np.stack(convertTrailData)
    return windowInfo, MneTrailData, trialsLabels
    # return {'id': subjectId, 'file_path': filePath, 'Fs': Fs, 'winNum': winNum}

def loadAllFile(fileFolder):
    #need change
    allData = []
    allLabel = []
    allInfo = []
    convertTrailData = []
    for fileName in os.listdir(fileFolder):
        if fileName.endswith('.mat'):
            filePath = os.path.join(fileFolder, fileName)
            winInfo, winData, winLabel = loadFile(filePath)
            allData.extend(winData)
            allLabel.extend(winLabel)
            allInfo.extend(winInfo)
    for window in allData:
        transWindow = np.array(window)
        convertTrailData.append(transWindow)
    MneTrailData = np.stack(convertTrailData)

    return allInfo, MneTrailData, allLabel
    
    
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

def generateMneData(filePath, Annotation, lowBand, highBand):
    '''
    Generate the data after band filter
    '''
    # filteredData, filteredLabel, filteredInfo = loadFilterFile(filePath)
    events = []
    filteredInfo, filteredData, filteredLabel = loadFile(filePath)
    # filteredLabel, filteredData, filteredInfo = loadFile(filePath)
    channelName = ['CH0','CH1','CH2','CH3','CH4','CH5','CH6',
                    'CH7','CH8','CH9','CH10','CH11','CH12','CH13','CH14','CH15']
    fsMne = dataParameters.Fs
    chType = 'eeg'
    mneInfo = mne.create_info(ch_names=channelName, sfreq=fsMne, ch_types=chType)
    # mneInfo.set_montage('biosemi64')
    transposed_data = np.transpose(filteredData[0], (1, 0))
    dataDf = pd.read_csv(Annotation)
    for _, row in dataDf.iterrows():
        if row['subjectId'] == filteredInfo[0]['subjectId']:
            start = int(row['start'] * row['Fs'])
            end = int(row['end'] * row['Fs'])
            label = row['label']
            events.append([start, 0, label])
    epochs = mne.EpochsArray(filteredData, info=mneInfo, events=events)
    epochs.filter(l_freq=lowBand, h_freq=highBand)
    
    # epochs[0].plot(scalings='auto')
    print("123")
    return epochs, filteredLabel[:,0]
# epochs = generateMneData('/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/MI_BCI_Data/PAT013.mat', '/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/resultStore/Standard/Annotation.csv')
def ML_method():
    pass
def DL_method():
    pass
def evaluate():
    pass




# def bandPass(data, lowCut, highCut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowCut / nyq
#     high = highCut / nyq
#     b, a = signal.butter(order, [low, high], btype='band')
#     return signal.filtfilt(b, a, data, axis=0)

# def filterSignal(filePath, lowCut, highCut, Fs, mat_filename):
#     filteredSignal = []
#     filteredLabel = []
#     winInfo, winData, winLabel = loadFile(filePath)
#     subjectIdStore = winInfo[0]['subjectId']
#     FsStore = winInfo[0]['Fs']
#     filePathStore = winInfo[0]['filePath']
#     for win, label in zip(winData, winLabel):
#         windowOri = win[0]
#         if windowOri.shape[1] == 16:
#             filteredWinow = bandPass(windowOri, lowCut, highCut,Fs)
#             filteredSignal.append(filteredWinow)   
#             filteredLabel.append(label)
    
#     scio.savemat(mat_filename, {
#                             'filteredSignal': filteredSignal,
#                             'filteredLabel':filteredLabel,
#                             'subjectId':subjectIdStore,
#                             'Fs':FsStore,
#                             'filePath':filePathStore
#                             })
    # return filteredInfo, filteredSignal, filteredLabel
# def loadFilterFile(filePath):
#     '''
#     use filteredData[n] to get different window
#     use filteredData[n][m] to get different sample point
#     use filteredLabel[n][0] to get window's label
#     '''
#     data = scio.loadmat(filePath)
#     filteredData = data['filteredSignal']
#     filteredLabel = data['filteredLabel']
#     subjectId = data['subjectId']
#     Fs = data['Fs'][0][0]
#     filePathInMat = data['filePath']
    
#     windowInfo = []
#     for i, (window, label) in enumerate(zip(filteredData, filteredLabel)):
#         # print("winNum",winNum,"shape=",window[0].shape[0])
#         start = i * window.shape[0] / Fs
#         end = (i + 1) * window.shape[0] / Fs
#         # print("shape",start)
#         windowInfo.append({
#             'subjectId': f'{subjectId}',
#             'Fs': Fs,
#             'start': start,
#             'end': end,
#             'label': label[0],
#             'filePath': filePath
#         })
#     return filteredData, filteredLabel, windowInfo

# def loadAllFilterFile(fileFolder):
#     allData = []
#     for fileName in os.listdir(fileFolder):
#         if fileName.endswith('.mat'):
#             filePath = os.path.join(fileFolder, fileName)
#             winData = loadFilterFile(filePath)
#             allData.extend(winData)
#     return allData