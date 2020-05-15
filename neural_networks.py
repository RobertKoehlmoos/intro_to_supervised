import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

# Made with help from https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
# Importing our data fom the local csv file
banknote_headers = ["variance", "skewness", "kurtosis", "entropy", "classification"]
banknotes = pd.read_csv("./data_sets/data_banknote_authentication.txt", header=None, names=banknote_headers)

# Separating the data into the features and classifications
X = banknotes.drop("classification", axis=1)
y = banknotes["classification"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)

# The below block normalizes the data to improve neural net convergence time
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
# Applying the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Creating the MLPs, 3 layers with 4 neurons chosen because of the four
# input features
mlps = []
max_iterations = range(4, 150, 2)
for max_iter in max_iterations:
    mlp = MLPClassifier(hidden_layer_sizes=(4, 4, 4), max_iter=max_iter)
    mlp.fit(X_train, y_train)  # Training the data
    mlps.append(mlp)

train_scores = [mlp.score(X_train, y_train) for mlp in mlps]
test_scores = [mlp.score(X_test, y_test) for mlp in mlps]

fig, ax = plt.subplots()
ax.set_xlabel("Iterations")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs Iterations for training and testing sets of banknotes")
ax.plot(max_iterations, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(max_iterations, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

sonar_data = pd.read_csv("./data_sets/sonar.csv", header=None)

# Separating the dataset columns into the features and labels
X = sonar_data.iloc[:, :-1]
y = sonar_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)

# The below block normalizes the data to improve neural net convergence time
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
# Applying the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Creating the MLPs, 3 layers with 4 neurons chosen because of the four
# input features
mlps = []
max_iterations = range(1, 200, 5)
for max_iter in max_iterations:
    mlp = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=(60, 60, 60))
    mlp.fit(X_train, y_train)  # Training the data
    mlps.append(mlp)

train_scores = [mlp.score(X_train, y_train) for mlp in mlps]
test_scores = [mlp.score(X_test, y_test) for mlp in mlps]

fig, ax = plt.subplots()
ax.set_xlabel("Iterations")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Iterations for training and testing sets of sonar data")
ax.plot(max_iterations, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(max_iterations, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()
