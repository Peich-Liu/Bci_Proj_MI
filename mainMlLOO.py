import numpy as np
import os

from mne.decoding import CSP

from MethodLib import loadFile, loadAllFile, createOriAnnotationFile, generateMneData
from parameterSetup import dataParameters, filterParameter
import matplotlib.pyplot as plt
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
# windowInfo, trialsData, trialsLabels = loadFile('/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/MI_BCI_Data/PAT013.mat')
# allData, allLabel = loadAllFile(dataDir)

# for file in os.listdir(dataDir):
#     filePath = dataDir + '/' + file
#     standardPath = filterDir + os.path.splitext(file)[0] + '.mat'
#     standardPath = os.path.abspath(standardPath)
#     filterSignal(filePath, filterParameter.lowCut, filterParameter.highCut, filterParameter.Fs, standardPath)
########################################
#Annotation Extraction
# createOriAnnotationFile(dataDir, annotationPath)
########################################
#data load info
# loadFilterFile('/Users/liu/Documents/22053 Principles of brain computer interface/miniProj/Bci_Proj_MI/resultStore/Standard/afterFilter/PAT013.mat')
# dataParameters.subject = [os.path.splitext(fileName)[0] for fileName in os.listdir(dataDir)]
########################################
####Spatial Filters
##CSP filter
for fileName in os.listdir(dataDir):
    filePathLoop = dataDir + '/' +fileName
    epochs, dataLabel = generateMneData(filePathLoop, annotationPath, filterParameter.lowCut, filterParameter.highCut)
    csp = CSP(n_components=4, norm_trace=False)
    csp.fit(epochs.get_data(), dataLabel)
    csp_features = csp.transform(epochs.get_data())

    feature_1 = csp_features[:, 1]
    feature_2 = csp_features[:, 2]

    plt.figure(figsize=(10, 6))
    plt.scatter(feature_1[dataLabel == 0], feature_2[dataLabel == 0], c='blue', label='Class 0', alpha=0.5)
    plt.scatter(feature_1[dataLabel == 1], feature_2[dataLabel == 1], c='red', label='Class 1', alpha=0.5)

    plt.xlabel('CSP Feature 1')
    plt.ylabel('CSP Feature 2')
    plt.title('2D Plot of CSP Features')
    plt.legend()

    plt.show()
    csp.plot_patterns(epochs.info)
    csp_component_1 = csp_features[:, 0]
    csp_component_2 = csp_features[:, 1]
    plt.figure()
    plt.plot(csp_component_1, label='Label=0')
    plt.plot(csp_component_2, label='Label=1')
    plt.legend()

    plt.title('After CSP')
    plt.xlabel('Samples')
    plt.ylabel('CSP Feature Value')

    plt.show()

    print("123")
########################################
#classification


########################################
#evaluate