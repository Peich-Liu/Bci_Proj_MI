import numpy as np
import pandas as pd
import os
import torch
import mne

from mne.decoding import CSP
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

from MethodLib import *
from parameterSetup import dataParameters, filterParameter, MLParameters
from architecture import *




########################################
#Folder Create
dataDir = 'MI_BCI_Data'

outDir = 'resultStore/'
featureDir = outDir + 'Feature/'
visualDir = outDir + 'Visual/'

annotationPath = outDir + 'Annotation.csv'

os.makedirs(os.path.dirname(outDir), exist_ok=True)
os.makedirs(os.path.dirname(featureDir), exist_ok=True)
os.makedirs(os.path.dirname(visualDir), exist_ok=True)
########################################
#Annotation Extraction
createOriAnnotationFile(dataDir, annotationPath)
########################################
###Data Standardize
## create folder
standDir = outDir +  'Signal/original/'
filterDir =outDir +  'Signal/afterFilter/'
cspDir = outDir + 'Signal/afterCsp/'
os.makedirs(os.path.dirname(standDir), exist_ok=True)
os.makedirs(os.path.dirname(filterDir), exist_ok=True)
os.makedirs(os.path.dirname(cspDir), exist_ok=True)
# #######################################
# # ##generate original fif
for fileName in os.listdir(dataDir):
    filePathLoop = dataDir + '/' +fileName
    epochs, dataLabel = generateMneData(filePathLoop, annotationPath, filterParameter.lowCut, filterParameter.highCut)
    epochs.save(os.path.join(standDir, f'original_{fileName}.fif'), overwrite=True)
# ########################################
##Band Power Filter
#generate bp after fif
for fileName in os.listdir(dataDir):
    filePathLoop = dataDir + '/' +fileName
    epochsFilter, dataLabel = generateFilterMneData(filePathLoop, annotationPath, filterParameter.lowCut, filterParameter.highCut)
    epochsFilter.save(os.path.join(filterDir, f'filtered_{fileName}.fif'), overwrite=True)
#Band Power Filter Mu
#generate bp after fif
for fileName in os.listdir(dataDir):
    filePathLoop = dataDir + '/' +fileName
    epochsFilter, dataLabel = generateFilterMneData(filePathLoop, annotationPath, 7.0, 11.0)
    epochsFilter.save(os.path.join(filterDir, f'Mu_{fileName}.fif'), overwrite=True)
##Band Power Filter Beta
#generate bp after fif
for fileName in os.listdir(dataDir):
    filePathLoop = dataDir + '/' +fileName
    epochsFilter, dataLabel = generateFilterMneData(filePathLoop, annotationPath, 14.0, 30.0)
    epochsFilter.save(os.path.join(filterDir, f'Beta_{fileName}.fif'), overwrite=True)
# ########################################
# EEG artifact modeling/rejection
########################################
###Spatial Filters(Feature Extraction)
##create folder
featureOri = featureDir + 'FeatureOri/'
featureDirBand = featureDir + 'FeatureAfterBP/'
os.makedirs(os.path.dirname(featureOri), exist_ok=True)
os.makedirs(os.path.dirname(featureDirBand), exist_ok=True)
########################################k=True)
#CSP filter for original signal
allCspFeature = []
for file in os.listdir(standDir):
    fileFolderInLoop = os.path.join(standDir,file)
    filePathInLoop = os.path.abspath(fileFolderInLoop)
    extractCspFeature(outDir, filePathInLoop, featureOri)
##CSP filter for signal after band pass
allCspFeatureBandPass = []
for file in os.listdir(filterDir):
    fileFolderInLoopBand = os.path.join(filterDir,file)
    filePathInLoopBand = os.path.abspath(fileFolderInLoopBand)
    extractCspFeature(outDir, filePathInLoopBand, featureDirBand)
    CspFilter(outDir, filePathInLoopBand, cspDir)
########################################
####CSP visualization
##Generate the Csp Result Folder
cspOriVisual = visualDir + 'Ori/'
cspBandVisual = visualDir + 'AfterBP/'
os.makedirs(os.path.dirname(cspOriVisual), exist_ok=True)
os.makedirs(os.path.dirname(cspBandVisual), exist_ok=True)
# #######################################
# ##Generate the Original Signal Result-maybe not use
# for file in os.listdir(featureOri):
#     fileInVisual = os.path.abspath(os.path.join(featureOri,file))
#     Csp2DFeatureGenerate(outDir, fileInVisual, cspOriVisual)
# #Generate the Band Pass Signal Result
# for file in os.listdir(featureDirBand):
#     fileInVisual = os.path.abspath(os.path.join(featureDirBand,file))
#     Csp2DFeatureGenerate(outDir, fileInVisual, cspBandVisual)
# ########################################
# ##load All Feature
dataParameters.subject = [os.path.splitext(fileName)[0] for fileName in os.listdir(dataDir)]
dataAllFeatureOri = loadAllFeature(featureOri, dataParameters.subject)
dataAllFeatureBand = loadAllFeature(featureDirBand, dataParameters.subject)
# ########################################
# ##classification
NonFeatureColumns= ['subjectId', 'start_time', 'label']
# # ##classification for Ori Data
# # ########################################
# #### Machine Learning - SVM
# # ##classification for Band Data
meanAcc = []
meanPrec = []
meanF1 = []
meanSen = []
subId = []
allReportSVM = []
classPathSVM = outDir + "SVM_Personal_K_Validation.csv"
for subIdx, sub in enumerate(dataParameters.subject):
    runFeatureOriFile = dataAllFeatureBand[dataAllFeatureBand['subjectId'] == sub]
    runFeatureOri = runFeatureOriFile.loc[:,~runFeatureOriFile.columns.isin(NonFeatureColumns)]   
    X = runFeatureOri.to_numpy()
    y = runFeatureOriFile['label'].to_numpy()
    model = SVC(kernel='rbf')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_predictions = []
    all_labels = []
    # allAcc = []
    # allSen = []
    # allPrec = []
    # allF1 = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_labels.extend(y_test)
        all_predictions.extend(y_pred)
    resultSVM = generateClassifyRes(all_labels, all_predictions, sub)
    allReportSVM.append(resultSVM)
svmDf = pd.DataFrame(allReportSVM)
svmDf.to_csv(classPathSVM)
# # # ########################################
# # # #### Deep Learning - CNN
learningRate = 0.000002

meanAccDl = []
meanPrecDl = []
meanF1Dl = []
meanSenDl = []
subIdDl = []
allReportCNN = []
classPathCNN = outDir + "CNN_Personal_K_Validation.csv"
for subIdx, sub in enumerate(dataParameters.subject):
    all_predictionsCNN = []
    all_labelsCNN = []
    testSubject = sub
    # runBand = filterDir + 'filtered_' +sub + '.mat.fif'
    # runOri = standDir + 'original_' +sub + '.mat.fif'
    runMu = filterDir + 'Mu_' +sub + '.mat.fif'
    runBeta = filterDir + 'Beta_' +sub + '.mat.fif'
    
    runEpochMu = mne.read_epochs(runMu, preload=True)
    runEpochBeta = mne.read_epochs(runBeta, preload=True)
    runSignalMu = runEpochMu.get_data()
    runSignalBeta = runEpochBeta.get_data()
    
    allAcc = []
    allSen = []
    allPrec = []
    allF1 = []
    for cv in range(2):
        length = runSignalMu.shape[0]
        halfLen = int(np.ceil(length*0.5)) 

        if cv == 0:
            trainData = DLSignal(runSignalMu[:halfLen,:,:], runSignalBeta[:halfLen,:,:], runEpochMu.events[:halfLen,2])
            testData = DLSignal(runSignalMu[halfLen:, :, :], runSignalBeta[halfLen:, :, :], runEpochMu.events[halfLen:, 2])

        elif cv == 1:
            testData = DLSignal(runSignalMu[:halfLen,:,:], runSignalBeta[:halfLen,:,:], runEpochMu.events[:halfLen,2])
            trainData = DLSignal(runSignalMu[halfLen:, :, :], runSignalBeta[halfLen:, :, :], runEpochMu.events[halfLen:, 2])

        print("123")
        #train data
        trainLoader = DataLoader(trainData, batch_size=32, shuffle=True)
        # #test data
        testLoader = DataLoader(testData, batch_size=32, shuffle=True)
        
        #Training
        Model = DLTraining(trainLoader, learningRate)
        #Test
        labelsCNN, predictionsCNN, correct, total = DLTest(Model, testLoader)
        all_labelsCNN.extend(labelsCNN)
        all_predictionsCNN.extend(predictionsCNN)
    resultCNN = generateClassifyRes(all_labelsCNN, all_predictionsCNN, sub)
    allReportCNN.append(resultCNN)
svmDf = pd.DataFrame(allReportCNN)
svmDf.to_csv(classPathCNN)
########################################
# #### Evaluate
oriStemDir = cspBandVisual + 'Stem/'
cspStemDir = cspBandVisual + 'Stem/'
os.makedirs(os.path.dirname(oriStemDir), exist_ok=True)
os.makedirs(os.path.dirname(cspStemDir), exist_ok=True)
# ########################################
## calculate the average
## SVM
finalDf = pd.read_csv(classPathSVM)
mean = finalDf.loc[:, finalDf.columns != 'subId'].mean().to_frame().T
mean['subId'] = 'average'
finalDf = pd.concat([finalDf,mean], ignore_index=True)
finalDf.to_csv(classPathSVM, index=False)
##CNN
finalDf = pd.read_csv(classPathCNN)
mean = finalDf.loc[:, finalDf.columns != 'subId'].mean().to_frame().T
mean['subId'] = 'average'
finalDf = pd.concat([finalDf,mean], ignore_index=True)
finalDf.to_csv(classPathCNN, index=False)
#csp
for filePathCsp in os.listdir(cspDir):
    cspVarStemFigure(cspDir, filePathCsp, NonFeatureColumns, cspStemDir)

