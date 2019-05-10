# Estimators
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from Dataset import Dataset

# Others
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Import data from files

datasets = []
datasets.append(Dataset("datasets\diabetes.csv", "Diabetes set"))
datasets.append(Dataset("datasets\wine.csv", "Wine set"))
datasets.append(Dataset("datasets\german.csv", "German set"))
datasets.append(Dataset("datasets\popfailures.csv","Popfailures set"))
"""
#step 1 - full data approach
scores = []

for i in range(len(datasets)):
    classifiers = [
        KNeighborsClassifier(),
        GaussianNB(),
        DecisionTreeClassifier(),
        SVC(gamma='auto')
    ]
    predictions = []
    for j in range(len(classifiers)):
        classifiers[j].fit(datasets[i].X_train, datasets[i].y_train)
        predictions.append(classifiers[j].predict(datasets[i].X_test))
        scores.append(accuracy_score(datasets[i].y_test, predictions[j]))

print(datasets[0].name)
print(scores[0:4])
print(datasets[1].name)
print(scores[4:8])
print(datasets[2].name)
print(scores[8:12])
print(datasets[3].name)
print(scores[12:16])
"""
#step 2 - random subspace approach


#divide random subspaces of datasets
#calculate accuracy scores for each set
#compare results: normal vs random subspace