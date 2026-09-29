import sys
import kagglehub
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
##from sklearn.pipeline import Pipeline
#from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

#Checking to see if we are connecting to kagglehub.
print(sys.executable)
print(kagglehub.__version__)
#{sys.executable} -m pip install --upgrade certifi requests kagglehub

##Part 1: Explore the dataset
print("\nPart 1: Exploring the data.\n")
#To download dataset from Kaggel
path = kagglehub.dataset_download("ethancratchley/email-phishing-dataset")

#Find CSV files in the dataset
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

#Display available CSV files
print(csv_files)

#To load the first CSV file into a python DataFrame
df = pd.read_csv(os.path.join(path, csv_files[0]))

#To preview the data
print("\nHeader: \n",df.head())
print("\nShape: \n", df.shape)
print("\nInfo: \n", df.info())

#To examine the target variable
print(df["label"].value_counts())

print(df.groupby("label").first())

#To change to percentage values
print(df["label"].value_counts(normalize=True) * 100)

##Part 2: Prepare the data
print("\nPart 2: Preparing the data.\n")
#Separate the features from the target
X = df.drop("label", axis=1)
y = df["label"]

#Split the dataset into 80/20 training/testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

#Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##Part 3: Train a Linear SVM
print("\nPart 3: Training a Linear SVM.\n")
linear_svm = LinearSVC(C=1, class_weight="balanced", max_iter=10000, random_state=42)
linear_svm.fit(X_train_scaled, y_train)

#To use the training model to predict the test data
y_pred_linear = linear_svm.predict(X_test_scaled)

#To evaluate the model
print("\nAccuracy: ", accuracy_score(y_test, y_pred_linear))
print("\nClassification Report: \n", classification_report(y_test, y_pred_linear))
print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred_linear))

##Part 4: Linear vs RBF Kernel
print("\nPart 4: Train SVM using RBF Kernel.\n")
#To train another SVM using RBF kernel on a smaller, stratified sample.
X_subset, _, y_subset, _ = train_test_split(X_train_scaled, y_train, train_size=30000, stratify=y_train, random_state=42)

rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", cache_size=4000)
rbf_svm.fit(X_subset, y_subset)
              
y_pred_rbf = rbf_svm.predict(X_test_scaled)

#To evaluate the model
print("\nAccuracy: ", accuracy_score(y_test, y_pred_rbf))
print("\nClassification Report: \n", classification_report(y_test, y_pred_rbf))
print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred_rbf))