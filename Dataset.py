from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from numpy import ndarray
from random import sample
from copy import deepcopy
class Dataset:
    def __init__(self, path, name):
        data = genfromtxt(path, delimiter=',')
        X = data[:, :0 - 1]
        y = data[:, -1]
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        self.name = name

    def GetRandomSubspaceDataset(self, attNumber=3):
        data = deepcopy(self)
        #attNumber = round(0.8*self.X.shape[1])
        randomIndexes = sample(range(self.X.shape[1]),attNumber)
        data.X = ndarray(shape=(self.X.shape[0],attNumber))
        for i in range(attNumber):
            data.X[:, i] = self.X[:, randomIndexes[i]]
        data.y = self.y
        data.X_train, data.X_test, data.y_train, data.y_test = train_test_split(data.X, data.y, test_size=0.3, random_state=0)
        return data