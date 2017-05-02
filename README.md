# DaibetesDataMining
ReadMe.txt 

This project has been developed on Anaconda platform and is supported by python 3.5.2.

To run this script in the terminal you need to provide the following command 

$ python sparse.py <arg1> <arg2> <arg3><arg4>

arg1 - this argument is the type of dimensionality reduction performed. You must provide “svd” or “rd” or “no”

If you do not want to perform any dimensionality reduction technique, you should provide no.

arg2 - this argument is the type of classification algorithm used. The values can be “knn” or “svm” or “dt”

arg3 - The classification has been performed on two different class labels named “readmitted” or “diabetesMed”. Provide readmitted to classify and test for readmitted class label otherwise It would run for “diabetesMed” class label.

arg4 - This argument takes the number of records as an input you want to test the classifier for.

svd  stands for Singular Value Decomposition
rd stands for Random Projection Dimensionality reduction 
knn - K nearest neighbor 
dt - decision tree
svm - support vector machine

A use case :

Navigate to src Folder and run the file using following command :

./run.sh

This will run the script for 1000 records for readmitted class label with Randam projection Reduction technique and Knn classification.


