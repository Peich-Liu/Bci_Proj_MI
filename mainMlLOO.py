import numpy as np
import os

from mne.decoding import CSP

from MethodLib import *
from parameterSetup import *

########################################
#Folder Create
dataDir = 'MI_BCI_Data'
outDir = 'resultStore/'

annotationDir = outDir + 'standard/'
annotationPath = annotationDir + 'Annotation.csv'

os.makedirs(os.path.dirname(outDir), exist_ok=True)
os.makedirs(os.path.dirname(annotationDir), exist_ok=True)
########################################
#Annotation Extraction
createAnnotationFile(dataDir, annotationPath)
########################################
#data load

########################################
#data preprocessing

# CSP filter
# csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

########################################
#classification


########################################
#evaluate