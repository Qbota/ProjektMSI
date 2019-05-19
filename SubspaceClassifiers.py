from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from random import sample
from numpy import ndarray
from sklearn import base
from numpy import swapaxes
from statistics import mean


class SubspaceClassifier():
    def __init__(self, numberOfSubspaces=10, classifier=KNeighborsClassifier(), attnumber=3):
        self.numberOfSubspaces = numberOfSubspaces
        self.classifier = classifier
        self.predictions = []
        self.attNumber = attnumber
        self.fittedClassifiers = []
        self.randomIndexes = []
    def fit(self, X, y):
        self.fittedClassifiers = []
        for i in range(self.numberOfSubspaces):
            self.randomIndexes.append(sample(range(X.shape[1]), self.attNumber))
            X_train = ndarray(shape=(X.shape[0], self.attNumber))
            for j in range(self.attNumber):
                X_train[:, j] = X[:, self.randomIndexes[i][j]]
            memberClf = base.clone(self.classifier)
            memberClf.fit(X_train, y)
            self.fittedClassifiers.append(memberClf)
        return self
    def predict(self, X):
        y_pred = []

        for i in range(len(self.fittedClassifiers)):
            X_test = ndarray(shape=(X.shape[0], self.attNumber))
            for j in range(self.attNumber):
                X_test[:, j] = X[:, self.randomIndexes[i][j]]
            self.predictions.append(self.fittedClassifiers[i].predict(X_test))
        
        self.predictions = swapaxes(self.predictions, 0, 1)
        for p in range(len(self.predictions)):
            if mean(self.predictions[p]) >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)

        return y_pred