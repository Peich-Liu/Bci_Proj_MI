import numpy as np
import pandas as pd
import os
import mne

from mne.decoding import CSP

from MethodLib import loadFile, loadAllFile, createOriAnnotationFile, generateMneData, Csp2DFeatureGenerate, CspFilter, generateFilterMneData
from parameterSetup import dataParameters, filterParameter
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
standDir = outDir +  'Standard/original/'
filterDir =outDir +  'Standard/afterFilter/'
os.makedirs(os.path.dirname(standDir), exist_ok=True)
os.makedirs(os.path.dirname(filterDir), exist_ok=True)
########################################
# ##generate original fif
# for fileName in os.listdir(dataDir):
#     filePathLoop = dataDir + '/' +fileName
#     epochs, dataLabel = generateMneData(filePathLoop, annotationPath, filterParameter.lowCut, filterParameter.highCut)
#     epochs.save(os.path.join(standDir, f'original_{fileName}.fif'), overwrite=True)
# ########################################
# ###Band Power Filter
# ##generate bp after fif
# for fileName in os.listdir(dataDir):
#     filePathLoop = dataDir + '/' +fileName
#     epochsFilter, dataLabel = generateFilterMneData(filePathLoop, annotationPath, filterParameter.lowCut, filterParameter.highCut)
#     epochsFilter.save(os.path.join(filterDir, f'filtered_{fileName}.fif'), overwrite=True)
########################################
###Spatial Filters
##create folder
featureOri = featureDir + 'FeatureOri/'
featureDirBand = featureDir + 'FeatureAfterBP/'
os.makedirs(os.path.dirname(featureOri), exist_ok=True)
os.makedirs(os.path.dirname(featureDirBand), exist_ok=True)
########################################
# ##CSP filter for original signal
# ##
# allCspFeature = []
# for file in os.listdir(standDir):
#     fileFolderInLoop = os.path.join(standDir,file)
#     filePathInLoop = os.path.abspath(fileFolderInLoop)
#     CspFilter(outDir, filePathInLoop, featureOri)
# ##CSP filter for signal after band pass
# allCspFeatureBandPass = []
# for file in os.listdir(filterDir):
#     fileFolderInLoopBand = os.path.join(filterDir,file)
#     filePathInLoopBand = os.path.abspath(fileFolderInLoopBand)
#     CspFilter(outDir, filePathInLoopBand, featureDirBand)
########################################
####CSP visualization
##Generate the Csp Result
cspOriVisual = visualDir + 'Ori/'
cspBandVisual = visualDir + 'AfterBP/'
os.makedirs(os.path.dirname(cspOriVisual), exist_ok=True)
os.makedirs(os.path.dirname(cspBandVisual), exist_ok=True)
##Generate the Original Signal Result
for file in os.listdir(featureOri):
    fileInVisual = os.path.abspath(os.path.join(featureOri,file))
    Csp2DFeatureGenerate(outDir, fileInVisual, cspOriVisual)
##Generate the Band Pass Signal Result
for file in os.listdir(featureDirBand):
    fileInVisual = os.path.abspath(os.path.join(featureDirBand,file))
    Csp2DFeatureGenerate(outDir, fileInVisual, cspBandVisual)
########################################
#classification
dataParameters.subject = [os.path.splitext(fileName)[0] for fileName in os.listdir(dataDir)]

X_train, X_test, y_train, y_test = train_test_split(cspFeatures, dataLabel, test_size=0.2, random_state=42)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
########################################
#evaluate
