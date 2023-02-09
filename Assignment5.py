import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the glass dataset
glass_data = pd.read_csv("C:/Users/sruja/OneDrive/Desktop/Neural Network/Neural-git/Assignments/NNDL_Code and Data/NNDL_Code and Data/glass.csv")

# Split the data into features and target variables
X = glass_data.drop("Type", axis=1)
y = glass_data["Type"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train the model using the Gaussian Naive Bayes algorithm
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gnb.predict(X_test)

# Evaluate the model using accuracy score
score = accuracy_score(y_test, y_pred)
print("Accuracy: ", score)

# Naive Bayes
print("Naive Bayes Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print()





#Question 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the glass dataset
glass_df = pd.read_csv("C:/Users/sruja/OneDrive/Desktop/Neural Network/Neural-git/Assignments/NNDL_Code and Data/NNDL_Code and Data/glass.csv")

# Split the dataset into features and target variables
features = glass_df.drop("Type", axis=1)
target = glass_df["Type"]

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=0)

# Create a linear SVM model
svm_model = SVC(kernel="linear")

# Train the model on the training data
svm_model.fit(features_train, target_train)

# Make predictions on the test data
target_pred = svm_model.predict(features_test)

# Evaluate the model's performance
print("Accuracy: ", accuracy_score(target_test, target_pred))
print("Support Vector Machine Confusion Matrix:\n", confusion_matrix(target_test, target_pred))
print("Classification Report: \n", classification_report(target_test, target_pred))