import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Created with help from
# https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4
banknote_headers = ["variance", "skewness", "kurtosis", "entropy", "classification"]
banknotes = pd.read_csv("./data_sets/data_banknote_authentication.txt", header=None, names=banknote_headers)

# Separating the dataset columns into the features and labels
X = banknotes[banknote_headers[:-1]]
y = banknotes.classification

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)


learning_rates = np.arange(0.1, 2.0, 0.1)
classifiers = []
for learning_rate in learning_rates:
    classifier = GradientBoostingClassifier(
        max_depth=2,
        n_estimators=10,
        learning_rate=learning_rate
    )
    classifier.fit(X_train, y_train)
    classifiers.append(classifier)

train_scores = [classifier.score(X_train, y_train) for classifier in classifiers]
test_scores = [classifier.score(X_test, y_test) for classifier in classifiers]

fig, ax = plt.subplots()
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Learning Rate for training and testing sets of banknotes")
ax.plot(learning_rates, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(learning_rates, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

sonar_data = pd.read_csv("./data_sets/sonar.csv", header=None)

# Separating the dataset columns into the features and labels
X = sonar_data.iloc[:, :-1]
y = sonar_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)


learning_rates = np.arange(0.05, 2.0, 0.05)
classifiers = []
for learning_rate in learning_rates:
    classifier = GradientBoostingClassifier(
        max_depth=2,
        n_estimators=10,
        learning_rate=learning_rate
    )
    classifier.fit(X_train, y_train)
    classifiers.append(classifier)

train_scores = [classifier.score(X_train, y_train) for classifier in classifiers]
test_scores = [classifier.score(X_test, y_test) for classifier in classifiers]

fig, ax = plt.subplots()
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Learning Rate for training and testing sonar data")
ax.plot(learning_rates, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(learning_rates, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()