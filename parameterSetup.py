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

class MLParameters:
    modelType = 'SVM'
    
    #SVM parameters
    SVM_kernel = 'linear'  # 'linear', 'rbf','poly'
    SVM_C = 1  # 1,100,1000
    SVM_gamma = 'auto' # 0  # 0,10,100
    
    #DecisionTree and random forest parameters
    DecisionTree_criterion = 'entropy'  # 'gini', 'entropy'
    DecisionTree_splitter = 'best'  # 'best','random'
    DecisionTree_max_depth = 0  # 0, 2, 5,10,20
    RandomForest_n_estimators = 100 #10,50, 100,250
