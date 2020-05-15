import pandas as pd

from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import svm
from sklearn import metrics


# Importing the data from a csv in an adjacent directory
banknote_headers = ["variance", "skewness", "kurtosis", "entropy", "classification"]
banknotes = pd.read_csv("./data_sets/data_banknote_authentication.txt", header=None, names=banknote_headers)

# Separating the dataset columns into the features and labels
X = banknotes[banknote_headers[:-1]]
y = banknotes.classification

# 80%/20% training testing split. The training set has 1097 instances and the test set has 275 instances.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Creating the model for training
clf = svm.SVC(kernel='linear')

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
