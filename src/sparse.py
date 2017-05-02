import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sys
from time import time
from sklearn.preprocessing import normalize
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import SparseRandomProjection


classLabelName = sys.argv[3].lower()
numberOfRecords = sys.argv[4].lower()

columnFeatureDict = {}
df = pd.read_csv(
    filepath_or_buffer='../data/dataset_diabetes/diabetic_data.csv',
    header=0,
    sep=',', nrows = int(numberOfRecords))

numberOfArguments = len(sys.argv)
arumentList = str(sys.argv)

isDimensionalityReductionApplied = ""
currentClassificationMethod = ""
print("Number of records", len(df.index))

def setUpProgramEnvironment():
    if numberOfArguments < 5:
        print("Script Should have three arguments : Python <ScriptName> <arg1> <arg2>.")
        print("arg2 would be rd for random Projection Reduction or svd for Singular Value decomposition reduction or no if not to be used")
        print("arg3 would be classification method name either eual to dt or svm or knn")
        print("arg4 is the class label either readmitted or med ")
        print("arg5 is the number of records for which the classifier would run")
        sys.exit("Execution Stopped")
    else :
        global isDimensionalityReductionApplied
        isDimensionalityReductionApplied = sys.argv[1].lower()
        global currentClassificationMethod
        currentClassificationMethod = sys.argv[2].lower()


setUpProgramEnvironment()
if classLabelName == "readmitted":
    cl = df['readmitted'].values
else :
    cl = df['diabetesMed'].values

X = df.ix[:].values

enc = LabelEncoder()
label_encoder = enc.fit(cl)
classLabels = label_encoder.transform(cl) + 1
if classLabelName == "readmitted":
    label_dict = {1: '<30', 2: 'NO', 3 :'>30'}
else :
    label_dict = {1: 'Yes', 2: 'No'}

#sklearn_lda = LDA(n_components=2)

if classLabelName == "readmitted":
    excludedColumns = ['encounter_id','patient_nbr','weight','payer_code','medical_specialty','readmitted']
    numericColumns = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses'];
else :
    excludedColumns = ['encounter_id','patient_nbr','weight','payer_code','medical_specialty','diabetesMed']
    numericColumns = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','number_diagnoses'];

def createFeatureDictionary(data):
    columnFeatureDict = {}
    featureId = 0
    for col in df:
        if col not in excludedColumns:
            uniqueFeatureList = data[col].unique()
            featureDict = {}
            if col in numericColumns:
                featureDict[col] = featureId
                featureId += 1;
            else:
                for feature in uniqueFeatureList:
                    if feature != '?':
                        featureDict[feature] = featureId
                        featureId += 1;
            columnFeatureDict[col] = featureDict
    return columnFeatureDict

def createVector(dict):
    vector = []
    start = True
    for col in df:
        if col in numericColumns:
            numeric =  True
        else:
            numeric =  False
        if col not in excludedColumns:
            i = 0
            for row in df[col]:
                if start == True:
                    vector.append({})
                if row != '?':
                    if numeric:
                        colId = dict[col][col]
                        vector[i][colId] = row
                    else:
                        colId = dict[col][row]
                        vector[i][colId] = 1
                i += 1
            start = False
    return vector

from scipy.sparse import csr_matrix
def createCSR(vector):
    pointer = []
    index = []
    values = []
    currentPointer = 0;
    pointer.append(currentPointer);
    for aRowVector in vector:
        currentPointer = currentPointer + len(aRowVector);
        pointer.append(currentPointer);
        for key,value in aRowVector.items():
            index.append(key);
            values.append(value);

    mat = csr_matrix((values,index,pointer), shape=(len(vector), len(index)));
    mat.sort_indices()
    return mat


from sklearn.random_projection import sparse_random_matrix
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD

def standardizeData(mat):
    min_max_scaler = preprocessing.MaxAbsScaler()
    x_scaled = min_max_scaler.fit_transform(mat)
    return x_scaled

from sklearn.decomposition import TruncatedSVD
def reduceDimensions(mat):
    svd = TruncatedSVD(n_components=50)
    data = svd.fit_transform(mat)
    return data

def reducedProjection(mat):
	rp = SparseRandomProjection(n_components=50)
	data = rp.fit_transform(mat)
	return data

import scipy.sparse as sp
def splitData(mat, cls, fold=1, d=10):
    n = mat.shape[0]
    r = int(np.ceil(n*1.0/d))
    mattr = []
    clstr = []
    for f in range(d):
        if f+1 != fold:
            mattr.append( mat[f*r: min((f+1)*r, n)] )
            clstr.extend( cls[f*r: min((f+1)*r, n)] )
    train = sp.vstack(mattr, format='csr')
    test = mat[(fold-1)*r: min(fold*r, n), :]
    clste = cls[(fold-1)*r: min(fold*r, n)]
    return train, clstr, test, clste

from sklearn import svm
from sklearn import metrics
from sklearn import tree

def nCrossValidation(n, data):
    eachFoldAccuracyList = []
    for f in range(n):
        train, clstr, test, clste = splitData(csr_matrix(data), classLabels, f+1, n)
        if currentClassificationMethod == "svm":
            clf = svm.SVC()
            clf.fit(train, clstr)
        elif currentClassificationMethod == "dt":
            clf = tree.DecisionTreeClassifier()
            clf.fit(train, clstr)
        else :
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(train, clstr)
        predictions = clf.predict(test)
        accuracy = metrics.accuracy_score(predictions,clste)
        eachFoldAccuracyList.append(accuracy)
        print("Accuracy for the fold:",accuracy)
    npAccuracyArray = np.array(eachFoldAccuracyList)
    print ("Mean Accuracy",np.mean(npAccuracyArray))

def csr_idf(mat, copy=False, **kargs):
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]

    return df if copy is False else mat, df

def inputDataForCrossValidation():
    if isDimensionalityReductionApplied == "svd":
        data = reduceDimensions(standardizeData(csrMat))
    elif isDimensionalityReductionApplied == "rd":
        csr_idf(csrMat)
        normalize(csrMat)
        data = reducedProjection(csrMat)
    else :
        if currentClassificationMethod == "knn":
            csr_idf(csrMat)
            normalize(csrMat)
            data = csrMat
        else :
            data = standardizeData(csrMat)
    return data

csrMat = createCSR(createVector(createFeatureDictionary(df)))
startTime = time()
nCrossValidation(10, inputDataForCrossValidation())
elapsedTime = time()-startTime
print("Elapsed Time :",elapsedTime)
