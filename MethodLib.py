import os
import csv
import mne
import torch

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import scipy.io as scio
from scipy.linalg import eigh
import scipy.signal as signal
from mne.decoding import CSP
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

from architecture import *
from parameterSetup import dataParameters
from Others.architecture import *
########################################
####File Methods
def loadFile(filePath):
    ''' 
    it's a summary of 16 channels
    using trialsData[n] (len=winNum) to get window 'n', 0<=n<=winNum-1
    using trailsData[n][0] (len=1) to get the window 'n' information, 0<=n<=winNum-1
    using trialsData[n][0][m] (len=16) to get every sample data in window 'n', sample point 'm' , 0<=m<=1535
    using trialsData[n][0][m][k] (specific number) to get sample data in 'k' channel in 'm' sample point, 0<=k<=15
    using trailsLabels[n][0] to get window's label, 0<=n<=winNum
    maybe change the data load methods in the future
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
                    
def loadFeature(featureDir, sub):
    loadPath = featureDir + sub + 'cspFeatureData.csv'
    featureDf = pd.read_csv(os.path.abspath(loadPath))
    return featureDf

def loadAllFeature(featureFile, subjects):
    AllFeature = pd.DataFrame([])
    for sub in subjects:
        singleFeature = loadFeature(featureFile, sub)
        # AllFeature.append(singleFeature)
        AllFeature = pd.concat([AllFeature, singleFeature],axis=0)
    return AllFeature

def loadCspSignal(filePath):
    dataLoad = scio.loadmat(filePath)
    signal = dataLoad['cspSignal']
    subjectId = dataLoad['subjectId']
    startTime = dataLoad['startTime']
    signalLabel = dataLoad['label']
    return signal, subjectId, startTime, signalLabel

########################################
####Data Methods
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

########################################
####CSP Methods
def CspFilter(outDir, filePath, outPutFolder):
#generate original csp signal
    # outPutPath = outPutFolder + ''
    epoch = mne.read_epochs(filePath, preload=True)
    if len(epoch.annotations) > 0:
        firstAnnotation = epoch.annotations[0]
        subjectId = firstAnnotation['description']
        print("subjectId:", subjectId)
    else:
        print("Annotation Error.")
    # featureResult = outPutFolder + subjectId + 'cspFeatureData.csv'
    featureResult = outPutFolder + subjectId + 'cspFeatureData.mat'
    labelsInLoop = epoch.events[:, 2]
    csp = CSP(n_components=4, norm_trace=False, transform_into='csp_space')
    csp.fit(epoch.get_data(), labelsInLoop)
    cspSignal = csp.transform(epoch.get_data())
    start = [data['onset'] for data in epoch.annotations]
    scio.savemat(featureResult, {'cspSignal': cspSignal,
                                        'subjectId':subjectId,
                                        'startTime': start,
                                        'label': labelsInLoop
                                        })

def extractCspFeature(outDir, filePath, outPutFolder):
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

def oriCspAnalysis(outDir, filePath, outPutDir):
    cspSignal, subjectId, startTime, signalLabel = loadCspSignal(filePath)
    numWin = cspSignal.shape[0]
    numSample = cspSignal.shape[2]
    
    plt.figure()
    for i in range(2):
        full_signal = np.concatenate([cspSignal[j, i, :] for j in range(numWin)])

        plt.subplot(2, 1, i + 1)
        plt.plot(range(len(full_signal)), full_signal)
        plt.title(f'CSP Channel {i + 1}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.ylim([-10, 10])

    plt.tight_layout()
    plt.show()
    
########################################
####ML Methods
def trainMlModel(X_train, y_train, MLParameters):
    if(MLParameters.modelType == 'SVM'):
        model = svm.SVC(kernel=MLParameters.SVM_kernel, C=MLParameters.SVM_C, gamma=MLParameters.SVM_gamma, probability=True)
        model.fit(X_train, y_train)
    elif(MLParameters.modelType == 'RF'):
        model = RandomForestClassifier(random_state=0, n_estimators=MLParameters.RandomForest_n_estimators, criterion=MLParameters.DecisionTree_criterion )
        model = model.fit(X_train, y_train)
    return model

def quickTest(X_test, y_test, MlModel):
    y_pred = MlModel.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def testML(X_test, y_test, MlModel):
    
    y_pred = MlModel.predict(X_test)
    y_probability = MlModel.predict_proba(X_test)
    y_pred = (y_probability[:, 1] < 0.5).astype(int)
    
    #pick only probability of predicted class
    y_probability_fin=np.zeros(len(y_pred))
    indx=np.where(y_pred==1)
    if (len(indx[0])!=0):
        y_probability_fin[indx]=y_probability[indx,1]

    indx = np.where(y_pred == 0)
    if (len(indx[0])!=0):
        y_probability_fin[indx] = y_probability[indx,0]
    # print("y_probability",y_probability)
    print(confusion_matrix(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    return y_pred, y_probability_fin, report
########################################
####Deep Learning
def DLTraining(trainLoader, learningRate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}.")
    
    # model = CNNnet(dataParameters.channelLen, 2).to(device)
    # model = Net(16,2).to(device)
    model = MultiStreamEEGNet(2, input_channels=16, sample_length=1536, num_classes=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    for epoch in range(25):
        # for data, label in trainLoader:
        #     data, label = data.to(device), label.to(device)
        #     optimizer.zero_grad()
        #     output = model(data)
        for alpha_input, beta_input, labels in trainLoader:
            alpha_input, beta_input, labels = alpha_input.to(device), beta_input.to(device), labels.to(device)
            output = model(alpha_input, beta_input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return model.to('cpu')

def DLTest(model, testLoader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data, label in testLoader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            all_labels.extend(label.numpy())
            all_predictions.extend(predicted.numpy())
    return all_labels, all_predictions, correct, total

def DLevaluate(all_labels, all_predictions, correct, total):
    precision = precision_score(all_labels, all_predictions, average='binary')  
    recall = recall_score(all_labels, all_predictions, average='binary')       
    f1 = f1_score(all_labels, all_predictions, average='binary')        
    accuracy = correct / total
    return accuracy, precision, recall, f1

########################################
####Evaluate Methods
def oriVarStemFigure(standDir, subjects,outPutDir):
    # for filePath in os.listdir(standDir):
    for sub in subjects:
        filePath = 'original_' + sub + '.mat.fif'
        standFileInLoop = standDir + filePath
        standEpochs = mne.read_epochs(os.path.abspath(standFileInLoop), preload=True)
        # variances_raw = np.var(standEpochs.get_data(), axis=(0, 2))
        data = standEpochs.get_data()
        
        events = standEpochs.events
        labels = events[:, -1] 
        mean_var_class_0 = []
        mean_var_class_1 = []
        stemResult = outPutDir +'stemOri_'+ sub + '.png'
        for ch in range(dataParameters.channelLen):
            dataLabel0 = data[labels==0]
            dataLabel1 = data[labels==1]
            
            chData0 = dataLabel0[:,ch,:]
            chData1 = dataLabel1[:,ch,:]
            
            variances_label0 = np.var(chData0, axis=1)
            variances_label1 = np.var(chData1, axis=1)
            mean_var_class_0.append(np.mean(variances_label0, axis=0))
            mean_var_class_1.append(np.mean(variances_label1, axis=0))
        plt.figure()
        plt.stem( mean_var_class_0, 'b',label='Label 0')
        plt.stem( mean_var_class_1, 'r',label='Label 1')
        plt.xlabel('Channels')
        plt.ylabel('Variance')
        # plt.title('Variance by Label')
        plt.title(filePath)
        plt.legend()
        plt.savefig(stemResult)
        # plt.show()


def cspVarStemFigure(cspDir, filePathCsp, NonFeatureColumns, outPut):

    cspFileInLoop = cspDir + filePathCsp
    cspSignal, subjectId, startTime, signalLabel = loadCspSignal(cspFileInLoop)
    
    labels = signalLabel
    mean_var_class_0 = []
    mean_var_class_1 = []
    for ch in range(4):
        dataLabel0 = cspSignal[labels[0]==0]
        dataLabel1 = cspSignal[labels[0]==1]
        
        chData0 = dataLabel0[:,ch,:]
        chData1 = dataLabel1[:,ch,:]
        
        variances_label0 = np.var(chData0, axis=1)
        variances_label1 = np.var(chData1, axis=1)
        mean_var_class_0.append(np.mean(variances_label0, axis=0))
        mean_var_class_1.append(np.mean(variances_label1, axis=0))
    plt.figure()
    plt.stem( mean_var_class_0, 'b',label='Label 0')
    plt.stem( mean_var_class_1, 'r',label='Label 1')
    plt.xlabel('Channels')
    plt.ylabel('Variance')
    plt.title('Variance by Label')
    # plt.title(filePath)
    plt.legend()
    plt.show()

def generateClassifyRes(y_test, y_pred, subID):
    report = classification_report(y_test, y_pred, output_dict=True)
    sen0 = report['0']['recall']
    sen1 = report['1']['recall']
    prec0 = report['0']['precision']
    prec1 = report['1']['precision']
    f1_0 = report['0']['f1-score']
    f1_1 = report['1']['f1-score']
    acc = report['accuracy']
    result = {'subId':subID,
                'sen0':sen0,
                'sen1':sen1,
                'prec0':prec0,
                'prec1':prec1,
                'f1_0':f1_0,
                'f1_1':f1_1,
                'acc':acc              
                }
    return result
    
    
    
    
    # cspFeatureDf = pd.read_csv(os.path.abspath(cspFileInLoop))
    # stemResult = outPut +'stemCspFeature_'+ cspFeatureDf['subjectId'][0] + '.png'
    # for csp in cspFeatureDf.loc[:,~cspFeatureDf.columns.isin(NonFeatureColumns)]:
    #     cspFeature0 = cspFeatureDf[cspFeatureDf['label'] == 0][csp]
    #     cspFeature1 = cspFeatureDf[cspFeatureDf['label'] == 1][csp]
    #     xLabel = ['Feature0', 'Feature1', 'Feature2', 'Feature3']
    #     cspVar0.append(np.var(cspFeature0))
    #     cspVar1.append(np.var(cspFeature1))
    # plt.figure()
    # plt.stem(xLabel,cspVar0,'b',label='label0')
    # plt.stem(xLabel,cspVar1,'r',label='label1')
    # plt.title('CSP Feature in Different Label')
    # plt.xlabel('Feature')
    # plt.ylabel('CSP Feature Value')
    # plt.legend()
    # # plt.show()
    # plt.savefig(stemResult)

def generatePredAnnotation(OriAnnotation, y_pred, y_prob):
    pass
def DL_method():
    pass
def evaluate():
    pass








########################################
####Some backup
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