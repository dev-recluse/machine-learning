from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression

###Step 1: Getting to know the data.

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

###Step 2: Preprocessing

#Test/Train split for standardization.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Initialize the Standard Scaler.
scaler = StandardScaler()

#Fit the training and test data and transform them.
X_training_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTotal samples: ", X.shape[0])
print("\nTraining size: ", X_train.shape[0])
print("\nTest size: ", X_test.shape[0])
print("\nNumber of features: ", X.shape[1])

###Step 3: Training with Logistic Regression and k-NN.
#Defining the models to train with Logistic Regression and k-NN.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "k-NN (k=3)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=3))
    ]),
    "k-NN (k=5)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "k-NN (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ])
}

#To train and evaluate each model.
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Get classification metrics as a dictionary
    metrics = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        "Model": name,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["weighted avg"]["precision"],
        "Recall": metrics["weighted avg"]["recall"]
    })

results_df = pd.DataFrame(results)
print(results_df)

###Step 4: Cross-Validation
