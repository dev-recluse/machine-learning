from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score
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
#For cross-validation.
cv_results = [] 

for name, model in models.items():
    accuracy_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    precision_scores = cross_val_score(model, X, y, cv=5, scoring="precision")
    recall_scores = cross_val_score(model, X, y, cv=5, scoring="recall")

    #Store the results for each fold.
    for fold in range(5):
        cv_results.append({
            "Model": name,
            "Fold": fold+1,
            "Accuracy": accuracy_scores[fold],
            "Precision": precision_scores[fold],
            "Recall": recall_scores[fold]
        })

#Convert results to data frame.
cv_results = pd.DataFrame(cv_results)
print(cv_results)

#To calculate the average standard deviation across the 5 folds.
cv_std = cv_results.groupby("Model")[["Accuracy", "Precision", "Recall"]].agg(["mean", "std"])
print(cv_std)

#Cross validation for k only.
k_values = list(range(1, 31))
cv_scores = []

for k in k_values:
    pipe_k = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    #Cross validating training data.
    scores = cross_val_score(pipe_k, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_k = k_values[int(np.argmax(cv_scores))]

print("\nBest k by CV on training set: ", best_k)
print("\nBest mean CV accuracy: ", max(cv_scores))

#Make the plots for training k CV.
plt.figure()
plt.plot(k_values, cv_scores, marker="o")
plt.xlabel("k (neighbors)")
plt.ylabel("Mean CV Accuracy (5-fold)")
plt.title("Choosing k via Cross-Validation")
plt.grid(True)
plt.show()

#Fit with best k and evaluate on test data.
best_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k))
])

best_knn.fit(X_train, y_train)
test_predictions = best_knn.predict(X_test)

print("Test-set classification report:")
print(classification_report(y_test, test_predictions, target_names=cancer.target_names))

#To cross validate Logistic Regression.
logistic_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=1000))
])

logistic_scores = cross_val_score(logistic_pipe, X_train, y_train, cv=5, scoring="accuracy")

print("Logistic Regression CV scores: ", logistic_scores)
print("Logistic Regression mean CV accuracy: ", logistic_scores.mean())

#To compare Logistic Regression and k-NN in data frame.
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", f"k-NN ({best_k})"],
    "Mean CV Accuracy": [logistic_scores.mean(), max(cv_scores).std()],
    "CV Accuracy Std": [logistic_scores.std(), np.array(cv_scores).std()]
})

print(comparison)

##ROC Curve and AUC
#To fit logistic regression model
logistic_pipe.fit(X_train, y_train)

#To get probablities
logistic_probabilities = logistic_pipe.predict_proba(X_test)[:,1]
knn_probabilities = best_knn.predict_proba(X_test)[:,1]

#Calculate ROC curves
logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_probabilities)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probabilities)

#Calculate AUC
logistic_auc = roc_auc_score(y_test, logistic_probabilities)
knn_auc = roc_auc_score(y_test, knn_probabilities)

#Plot ROC curves.
plt.figure()
plt.plot(logistic_fpr, logistic_tpr, label="Logistic Regression (AUC)")
plt.plot(knn_fpr, knn_tpr, label="k-NN (AUC)")
plt.plot([0,1], [0,1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

###To stimulate imbalanced data by downsampling one class.
#Combine features and target into one data frame.
imbalanced_df = pd.DataFrame(X, columns=cancer.feature_names)
imbalanced_df["target"] = y

#To separate the two classes.
class0 = imbalanced_df[imbalanced_df["target"]==0]
class1 = imbalanced_df[imbalanced_df["target"]==1]

#To downsample class 1.
class1_downsample = class1.sample(n=len(class1) // 2, random_state=42)

#To recombine the classes.
imbalanced_df = pd.concat([class0, class1_downsample], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

#Seperate features and target again.
X_imbalanced = imbalanced_df.drop(columns="target").values
y_imbalanced = imbalanced_df['target'].values

#To inspect the new class distribution.
print(pd.Series(y_imbalanced).value_counts())
print(pd.Series(y_imbalanced).value_counts(normalize=True))

#Split and evaluate the models on the imbalanced data.
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(X_imbalanced, y_imbalanced, test_size=0.20, random_state=42, stratify=y_imbalanced)
imbalanced_results = []

for name, model in {"Logistic Regression": logistic_pipe, f"k-NN (k={best_k})": best_knn}.items():
    model.fit(X_train_imb, y_train_imb)
    predictions = model.predict(X_test_imb)
    report = classification_report(y_test_imb, predictions, output_dict=True)
    imbalanced_results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"]
    })

imbalanced_results_df = pd.DataFrame(imbalanced_results)
print(imbalanced_results_df)

#To inspect the confusion matrixes.
for name, model in {"Logistic Regression": logistic_pipe, f"k-NN (k={best_k})": best_knn}.items():
    predictions = model.predict(X_test_imb)
    confMatrix = confusion_matrix(y_test_imb, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=confMatrix, display_labels=cancer.target_names)
    display.plot()
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

###To tune hyperparameters further for Logistic Regression.
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
logistic_tuning_results = []

for C in C_values:
    logistic_model = Pipeline([("scaler", StandardScaler()), ("logistic", LogisticRegression(C=C, max_iter=1000))])
    accuracy_scores = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring="accuracy")
    logistic_tuning_results.append({
        "C": C,
        "Mean Accuracy": accuracy_scores.mean(),
        "Std Accuracy": accuracy_scores.std()
    })

logistic_tuning_df = pd.DataFrame(logistic_tuning_results)
print(logistic_tuning_df)

#To print the best regularization value.
best_C = logistic_tuning_df.loc[logistic_tuning_df["Mean Accuracy"].idxmax(), "C"]
print("\nBest regularization value: ", best_C)

###To tune hyperparameters further for k-NN.
distance_metrics = ["euclidean", "manhattan", "minkowski"]
knn_tuning_results = []

for metric in distance_metrics:
    knn_model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=best_k, metric=metric))])
    accuracy_scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring="accuracy")
    knn_tuning_results.append({
        "Distance Metric": metric,
        "Mean Accuracy": accuracy_scores.mean(),
        "Std Accuracy": accuracy_scores.std()
    })

knn_tuning_df = pd.DataFrame(knn_tuning_results)
print(knn_tuning_df)

#To print the best metric.
best_metric = knn_tuning_df.loc[knn_tuning_df["Mean Accuracy"].idxmax(), "Distance Metric"]
print("\nBest metric value: ", best_metric)
        