from numpy import genfromtxt
from sklearn.model_selection import train_test_split
class Dataset:
    def __init__(self, path, name):
        data = genfromtxt(path, delimiter=',')
        X = data[:, :0 - 1]
        y = data[:, -1]
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        self.name = name

    def GetRandomSubspaceDataset(self):
        data = Dataset()

        return data