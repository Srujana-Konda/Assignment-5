# Assignment-5

Program 1: 
We are loading the given glass.csv file into dataframe using pandas. Then the dataset is divided into two sets one as training and other as testing set using the train_test_split function. A Gaussian Naive Bayes model is then created, trained on the training data, and used to make predictions on the test data. Finally, the model's performance is evaluated using the accuracy score and the classification report.

Program 2:
The first two steps of loading and spliting the data into two datasets remains same. But here we are using SVC class from scikit-learn's svm module to create a linear SVM model. The kernel parameter is set to "linear" to specify that we want to use a linear SVM. The model is then trained on the training data using the fit method and evaluated on the test data using the accuracy_score and classification_report functions from scikit-learn's metrics module.
