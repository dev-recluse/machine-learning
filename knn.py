from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

#Load breast cancer data.
cancer = load_breast_cancer()

#Feature matrix and target vector.
X = cancer.data
y = cancer.target

print("\nFeature columns: ", cancer.feature_names)
print("\nTarget variables: ", cancer.target_names)
print("\nFeature shape: ", X.shape)
print("\nSample shape: ", y.shape)

#Creating the data frame.
df = pd.DataFrame(X, columns=cancer.feature_names)
df["target"] = y
df.head()

#Print data types
print("\nData Types: \n", df.dtypes)

#Counting the missing values in each feature.
print("\nMissing values in each feature: ", df.isnull().sum())

#Inspecting the target values.
print("\nTarget data type: ", y.dtype)
print("\nTarget values: ", np.unique(y))

#Looking at the summary statistics for mean, min, max, and standard deviation for features.
summary_stats = df.describe().T[["mean", "min", "max", "std"]]
print("\nSummary Stats: \n", summary_stats)

#Creating sample histogram of first five features.
features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]

#Creating the plots.
df[features].hist(
    bins=20,
    figsize=(12,6),
    edgecolor="black"
)

plt.suptitle("Distributions of Selected Features")
plt.tight_layout()
plt.show()