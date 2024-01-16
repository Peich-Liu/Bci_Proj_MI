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
# createOriAnnotationFile(dataDir, annotationPath)
########################################
###Data Standardize
## create folder
standDir = outDir +  'Signal/original/'
filterDir =outDir +  'Signal/afterFilter/'
cspDir = outDir + 'Signal/afterCsp/'
os.makedirs(os.path.dirname(standDir), exist_ok=True)
os.makedirs(os.path.dirname(filterDir), exist_ok=True)
os.makedirs(os.path.dirname(cspDir), exist_ok=True)
# # #######################################
# # ##generate original fif
# for fileName in os.listdir(dataDir):
#     filePathLoop = dataDir + '/' +fileName
#     epochs, dataLabel = generateMneData(filePathLoop, annotationPath, filterParameter.lowCut, filterParameter.highCut)
#     epochs.save(os.path.join(standDir, f'original_{fileName}.fif'), overwrite=True)
# # ########################################
# # ##Band Power Filter
# # #generate bp after fif
# for fileName in os.listdir(dataDir):
#     filePathLoop = dataDir + '/' +fileName
#     epochsFilter, dataLabel = generateFilterMneData(filePathLoop, annotationPath, filterParameter.lowCut, filterParameter.highCut)
#     epochsFilter.save(os.path.join(filterDir, f'filtered_{fileName}.fif'), overwrite=True)
# # ########################################
# EEG artifact modeling/rejection
########################################
###Spatial Filters(Feature Extraction)
##create folder
featureOri = featureDir + 'FeatureOri/'
featureDirBand = featureDir + 'FeatureAfterBP/'
os.makedirs(os.path.dirname(featureOri), exist_ok=True)
os.makedirs(os.path.dirname(featureDirBand), exist_ok=True)
########################################k=True)
##CSP filter for original signal
# allCspFeature = []
# for file in os.listdir(standDir):
#     fileFolderInLoop = os.path.join(standDir,file)
#     filePathInLoop = os.path.abspath(fileFolderInLoop)
#     extractCspFeature(outDir, filePathInLoop, featureOri)
# ##CSP filter for signal after band pass
# allCspFeatureBandPass = []
# for file in os.listdir(filterDir):
#     fileFolderInLoopBand = os.path.join(filterDir,file)
#     filePathInLoopBand = os.path.abspath(fileFolderInLoopBand)
#     extractCspFeature(outDir, filePathInLoopBand, featureDirBand)
#     CspFilter(outDir, filePathInLoopBand, cspDir)
########################################
####CSP visualization
##Generate the Csp Result Folder
cspOriVisual = visualDir + 'Ori/'
cspBandVisual = visualDir + 'AfterBP/'
os.makedirs(os.path.dirname(cspOriVisual), exist_ok=True)
os.makedirs(os.path.dirname(cspBandVisual), exist_ok=True)
########################################
# ##Generate the Original Signal Result-maybe not use
# for file in os.listdir(featureOri):
#     fileInVisual = os.path.abspath(os.path.join(featureOri,file))
#     Csp2DFeatureGenerate(outDir, fileInVisual, cspOriVisual)
##Generate the Band Pass Signal Result
# for file in os.listdir(featureDirBand):
#     fileInVisual = os.path.abspath(os.path.join(featureDirBand,file))
#     Csp2DFeatureGenerate(outDir, fileInVisual, cspBandVisual)
# ########################################
# ##load All Feature
dataParameters.subject = [os.path.splitext(fileName)[0] for fileName in os.listdir(dataDir)]
# # dataAllFeatureOri = loadAllFeature(featureOri, dataParameters.subject)
# dataAllFeatureBand = loadAllFeature(featureDirBand, dataParameters.subject)
# ########################################
# ##classification
NonFeatureColumns= ['subjectId', 'start_time', 'label']
# # ##classification for Ori Data
#######################################
#### Machine Learning - SVM
# ##classification for Band Data
reportSVM = []
classPathSVM = outDir + "SVM_Loo_Validation.csv"
# for subIdx, sub in enumerate(dataParameters.subject):
#     trainFeatureOriFile = dataAllFeatureBand[dataAllFeatureBand['subjectId'] != sub]
#     testFeatureOriFile = dataAllFeatureBand[dataAllFeatureBand['subjectId'] == sub]
#     trainFeatureOri = trainFeatureOriFile.loc[:,~trainFeatureOriFile.columns.isin(NonFeatureColumns)]
#     testFeatureOri = testFeatureOriFile.loc[:,~testFeatureOriFile.columns.isin(NonFeatureColumns)]
#     ## data spilt
#     X_train = trainFeatureOri.to_numpy()
#     y_train = trainFeatureOriFile['label'].to_numpy()
#     X_test = testFeatureOri.to_numpy()
#     y_test = testFeatureOriFile['label'].to_numpy()
#     # train model
#     # SVM model
#     MLParameters.modelType = 'SVM'
#     SVMModel = trainMlModel(X_train, y_train, MLParameters)
#     yPredSVM, yProbSVM, report = testML(X_test, y_test, SVMModel)
#     result = generateClassifyRes(y_test, yPredSVM, sub)
    
#     reportSVM.append(result)
# svmDf = pd.DataFrame(reportSVM)
# svmDf.to_csv(classPathSVM)
# print("123")

# ########################################
# #### Deep Learning - CNN
learningRate = 0.0002
allReportCNN = []
classPathCNN = outDir + "CNN_Loo_Validation.csv"
for subIdx, sub in enumerate(dataParameters.subject):
    testSubject = sub
    testFile = filterDir + 'filtered_' +sub + '.mat.fif'
    # testFile = standDir + 'original_' +sub + '.mat.fif'

    trainSubjects = [p for p in dataParameters.subject if p != sub]
    testMu = filterDir + 'Mu_' +testSubject + '.mat.fif'
    testBeta = filterDir + 'Beta_' +testSubject + '.mat.fif'
    testEpochMu = mne.read_epochs(testMu, preload=True)
    testEpochBeta = mne.read_epochs(testBeta, preload=True)
    testSignalMu = testEpochMu.get_data()
    testSignalBeta = testEpochBeta.get_data()
    
    trainSignal = []
    trainLabels = []
    allTrainSignalMu = []
    allTrainSignalBeta = []
    for trSub in trainSubjects:
        trainMu = filterDir + 'Mu_' +trSub + '.mat.fif'
        trainBeta = filterDir + 'Beta_' +trSub + '.mat.fif'
        trainEpochMu = mne.read_epochs(trainMu, preload=True)
        trainEpochBeta = mne.read_epochs(trainBeta, preload=True)
        trainSignalMu = trainEpochMu.get_data()
        trainSignalBeta = trainEpochBeta.get_data()
        # trainFile = filterDir + 'filtered_' + trSub + '.mat.fif'
        # trainEpoch = mne.read_epochs(trainFile, preload=True)
        # trSignal = trainEpoch.get_data()
        # trSignalLabel = trainEpoch.events[:, 2]
        # trainSignal.extend(trSignal.transpose(0,2,1))
        allTrainSignalMu.extend(trainSignalMu)
        allTrainSignalBeta.extend(trainSignalBeta)
    #train data
    trainData = DLSignal(allTrainSignalMu, allTrainSignalBeta, trainEpochMu.events[:, 2])
    trainLoader = DataLoader(trainData, batch_size=32, shuffle=True)
    
    # #test data
    testData = DLSignal(testSignalMu, testSignalBeta, testEpochMu.events[:, 2])
    testLoader = DataLoader(testData, batch_size=32, shuffle=True)
    # testSignal = testEpoch.get_data()
    # testSignalLabel = testEpoch.events[:, 2]
    # testTensor = torch.tensor(testSignal, dtype=torch.float32)
    # testLabelsTensor = torch.tensor(testSignalLabel, dtype=torch.long)
    # testData = TensorDataset(testTensor, testLabelsTensor)
    # testLoader = DataLoader(testData, batch_size=32, shuffle=True)
    # testData = DLSignal(runSignalMu[halfLen:, :, :], runSignalBeta[halfLen:, :, :], runEpochMu.events[halfLen:, 2])
    #Training
    Model = DLTraining(trainLoader, learningRate)
    #Test
    all_labels, all_predictions, correct, total = DLTest(Model, testLoader)
    resultCNN = generateClassifyRes(all_labels, all_predictions, sub)
    allReportCNN.append(resultCNN)
cnnDf = pd.DataFrame(allReportCNN)
cnnDf.to_csv(classPathCNN)
    # accuracy, precision, recall, f1 = DLevaluate(all_labels, all_predictions, correct, total)
########################################
# #### Evaluate
oriStemDir = cspBandVisual + 'Stem/'
cspStemDir = cspBandVisual + 'Stem/'
os.makedirs(os.path.dirname(oriStemDir), exist_ok=True)
os.makedirs(os.path.dirname(cspStemDir), exist_ok=True)
# ########################################
# ## calculate the average
# ## SVM
# finalDf = pd.read_csv(classPathSVM)
# mean = finalDf.loc[:, finalDf.columns != 'subId'].mean().to_frame().T
# mean['subId'] = 'average'
# finalDf = pd.concat([finalDf,mean], ignore_index=True)
# finalDf.to_csv(classPathSVM, index=False)
##CNN
finalDf = pd.read_csv(classPathCNN)
mean = finalDf.loc[:, finalDf.columns != 'subId'].mean().to_frame().T
mean['subId'] = 'average'
finalDf = pd.concat([finalDf,mean], ignore_index=True)
finalDf.to_csv(classPathCNN, index=False)

# for filePathCsp in os.listdir(cspDir):
#     cspVarStemFigure(cspDir, filePathCsp, NonFeatureColumns, cspStemDir)

print("123")