import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Created with help from https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
banknote_headers = ["variance", "skewness", "kurtosis", "entropy", "classification"]
banknotes = pd.read_csv("./data_sets/data_banknote_authentication.txt", header=None, names=banknote_headers)

# Separating the dataset columns into the features and labels
X = banknotes[banknote_headers[:-1]]
y = banknotes.classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)

# The data ranges from around -10 to 10, varying by feature. Thus, normalization is used to improve
# distance function, so that each feature is equally valued.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

accuracy_for_training = []
accuracy_for_testing = []

# Calculating error for K values between 1 and 40
for i in range(1, 200, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    accuracy_for_training.append(knn.score(X_train, y_train))
    accuracy_for_testing.append(knn.score(X_test, y_test))

fig, ax = plt.subplots()
ax.plot(range(1, 200, 10), accuracy_for_testing, marker='o', markersize=10, label="test")
ax.plot(range(1, 200, 10), accuracy_for_training, marker='o', markersize=10, label="train")
ax.set_title('Accuracy vs K value for training and testing sets of banknotes')
ax.set_xlabel('K Value')
ax.set_label('Accuracy')
ax.legend()
plt.show()
