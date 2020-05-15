import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function

import matplotlib.pyplot as plt

# Importing the data from a csv in an adjacent directory
banknote_headers = ["variance", "skewness", "kurtosis", "entropy", "classification"]
banknotes = pd.read_csv("./data_sets/data_banknote_authentication.txt", header=None, names=banknote_headers)

# Separating the dataset columns into the features and labels
X = banknotes[banknote_headers[:-1]]
y = banknotes.classification

# 80%/20% training testing split. The training set has 1097 instances and the test set has 275 instances.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating a decision tree for use
clf = DecisionTreeClassifier()

# Train a initial Decision Tree and generate potential cost complexity pruning values
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-2]  # Experiments show the last two values are large outliers

# This loop calculates accuracy for each alpha value
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Collecting the scores for plotting
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# Preparing the graph displaying accuracy vs alpha value
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets of banknotes")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

sonar_data = pd.read_csv("./data_sets/sonar.csv", header=None)

# Separating the dataset columns into the features and labels
X = sonar_data.iloc[:, :-1]
y = sonar_data.iloc[:, -1]

# 80%/20% training testing split. The training set has 166 instances and the test set has 42 instances.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating a decision tree for use
clf = DecisionTreeClassifier()

# Train a initial Decision Tree and generate potential cost complexity pruning values
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # Experiments show the last value is a large outlier

# This loop calculates accuracy for each alpha value
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Collecting the scores for plotting
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# Preparing the graph displaying accuracy vs alpha value
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets of sonar data")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()
