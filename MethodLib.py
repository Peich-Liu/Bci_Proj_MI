import os
import csv
import mne
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


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
            'winNum':winNum,
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

def createOriAnnotationFile(folderPath, annotationPath):
    '''
    Create an annotation CSV file for all .mat files in the specified folder.
    '''
    headers = ['subjectId', 'Fs', 'winNum', 'start', 'end', 'label', 'filePath']

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
    events = []
    filteredInfo, filteredData, filteredLabel = loadFile(filePath)
    channelName = ['CH0','CH1','CH2','CH3','CH4','CH5','CH6',
                    'CH7','CH8','CH9','CH10','CH11','CH12','CH13','CH14','CH15']
    fsMne = dataParameters.Fs
    chType = 'eeg'
    mneInfo = mne.create_info(ch_names=channelName, sfreq=fsMne, ch_types=chType)
    dataDf = pd.read_csv(Annotation)
    annotations = []
    onsets = []
    durations = []
    descriptions = []
    for _, row in dataDf.iterrows():
        if row['subjectId'] == filteredInfo[0]['subjectId']:
            subjectId = str(row['subjectId'])
            start = int(row['start'] * row['Fs'])
            end = int(row['end'] * row['Fs'])
            duration = row['end'] - row['start']
            onset = row['start']
            label = row['label']
            events.append([start, 0, label])
            onsets.append(onset)
            durations.append(duration)
            descriptions.append(subjectId)
    all_annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    epochs = mne.EpochsArray(filteredData, info=mneInfo, events=events)
    epochs.set_annotations(all_annotations)

    return epochs, filteredLabel[:,0]

def generateFilterMneData(filePath, Annotation, lowBand, highBand):
    '''
    Generate the data after band filter
    '''

    events = []
    filteredInfo, filteredData, filteredLabel = loadFile(filePath)
    channelName = ['CH0','CH1','CH2','CH3','CH4','CH5','CH6',
                    'CH7','CH8','CH9','CH10','CH11','CH12','CH13','CH14','CH15']
    fsMne = dataParameters.Fs
    chType = 'eeg'
    mneInfo = mne.create_info(ch_names=channelName, sfreq=fsMne, ch_types=chType)
    dataDf = pd.read_csv(Annotation)
    annotations = []
    onsets = []
    durations = []
    descriptions = []
    for _, row in dataDf.iterrows():
        if row['subjectId'] == filteredInfo[0]['subjectId']:
            subjectId = str(row['subjectId'])
            start = int(row['start'] * row['Fs'])
            end = int(row['end'] * row['Fs'])
            duration = row['end'] - row['start']
            onset = row['start']
            label = row['label']
            events.append([start, 0, label])
            onsets.append(onset)
            durations.append(duration)
            descriptions.append(subjectId)

    all_annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)

    epochs = mne.EpochsArray(filteredData, info=mneInfo, events=events)
    epochs.set_annotations(all_annotations)
    epochs.filter(l_freq=lowBand, h_freq=highBand)

    print("123")
    return epochs, filteredLabel[:,0]
# epochs = generateMneData('/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/MI_BCI_Data/PAT013.mat', '/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/resultStore/Standard/Annotation.csv')
def CspFilter(outDir, filePath, outPutFolder):
    # outPutPath = outPutFolder + ''
    epoch = mne.read_epochs(filePath, preload=True)
    if len(epoch.annotations) > 0:
        firstAnnotation = epoch.annotations[0]
        subjectId = firstAnnotation['description']
        print("subjectId:", subjectId)
    else:
        print("Annotation Error.")
    featureResult = outPutFolder + subjectId + 'cspFeatureData.csv'
    labelsInLoop = epoch.events[:, 2]
    csp = CSP(n_components=4, norm_trace=False)
    csp.fit(epoch.get_data(), labelsInLoop)
    cspFeatures = csp.transform(epoch.get_data())
    start = [data['onset'] for data in epoch.annotations]
    extra_info_df = pd.DataFrame({
    'subjectId':subjectId,
    'start_time': start,
    'label': labelsInLoop
    })  
    featureDf = pd.DataFrame(cspFeatures, columns=['Feature0', 'Feature1', 'Feature2', 'Feature3'])
    combinedDf = pd.concat([extra_info_df, featureDf], axis=1)
    combinedDf.to_csv(featureResult, index=False)
    
def Csp2DFeatureGenerate(outDir, filePath, outPutDir):

    df = pd.read_csv(filePath)
    featureResult = outPutDir + df['subjectId'][0] + 'cspFeatureOut.png'
    labelsInLoop = df['label']
    
    feature1 = df['Feature0']
    feature2 = df['Feature1']

    plt.figure(figsize=(10, 6))
    plt.scatter(feature1[labelsInLoop == 0], feature2[labelsInLoop == 0], c='blue', label='Class 0', alpha=0.5)
    plt.scatter(feature1[labelsInLoop == 1], feature2[labelsInLoop == 1], c='red', label='Class 1', alpha=0.5)

    plt.xlabel('CSP Feature 1')
    plt.ylabel('CSP Feature 2')
    plt.title('2D Plot of CSP Features')
    plt.legend()
    
    plt.savefig(featureResult)

    # plt.show()
    
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