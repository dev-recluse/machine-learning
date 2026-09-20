from sklearn.datasets import load_breast_cancer

#Loading the breast cancer data.
cancer = load_breast_cancer()

#Feature matrix and target vector.
X = cancer.data
y = cancer.target

print("Featured shape: ", X.shape)
print("Target shape: ", y.shape)
print("Feature names: ", cancer.feature_names)
print("Target names: ", cancer.target_names)