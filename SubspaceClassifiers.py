from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class SubspaceClassifiers():
    def __init__(self, numberOfSubspaces=10):
        self.numberOfSubspaces = numberOfSubspaces
        self.datasets = [
            KNeighborsClassifier(),
            GaussianNB(),
            DecisionTreeClassifier(),
            SVC(gamma='auto')
        ]
        self.predictions = []
    def fit(self, dataset):
        for i in range(self.numberOfSubspaces):
            self.datasets.append(dataset.GetRandomSubspaceDataset())
        for c in range(len(self.classifiers)):
            for i in range(len(self.datasets)):