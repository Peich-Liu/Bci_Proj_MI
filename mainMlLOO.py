import numpy as np
import os

from mne.decoding import CSP

from MethodLib import loadFile, loadAllFile, createOriAnnotationFile, bandPass, loadFilterFile, filterSignal
from parameterSetup import dataParameters, filterParameter

########################################
#Folder Create
dataDir = 'MI_BCI_Data'
outDir = 'resultStore/'
standDir = outDir +  'Standard/'
filterDir = standDir + 'afterFilter/'
featureDir = outDir + 'Feature/'
annotationPath = standDir + 'Annotation.csv'

os.makedirs(os.path.dirname(outDir), exist_ok=True)
os.makedirs(os.path.dirname(standDir), exist_ok=True)
os.makedirs(os.path.dirname(filterDir), exist_ok=True)
os.makedirs(os.path.dirname(featureDir), exist_ok=True)
########################################
####Data Standardize
## signal after band filter

for file in os.listdir(dataDir):
    filePath = dataDir + '/' + file
    standardPath = filterDir + os.path.splitext(file)[0] + '.mat'
    standardPath = os.path.abspath(standardPath)
    filterSignal(filePath, filterParameter.lowCut, filterParameter.highCut, filterParameter.Fs, standardPath)
########################################
#Annotation Extraction
# createOriAnnotationFile(dataDir, annotationPath)
########################################
#data load info
# loadFilterFile('/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/resultStore/Standard/afterFilter/PAT013.mat')
dataParameters.subject = [os.path.splitext(fileName)[0] for fileName in os.listdir(dataDir)]

# data_class_0 = [data for data, label in zip(allData, allLabels) if label == 0]
# data_class_1 = [data for data, label in zip(allData, allLabels) if label == 1]
########################################
####Spatial Filters
##CSP filter
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

########################################
#classification


########################################
#evaluate