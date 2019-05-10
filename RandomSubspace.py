from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from Dataset import Dataset
from statistics import mean



from sklearn.metrics import accuracy_score
#Import data from files

datasets = []
datasets.append(Dataset("datasets\diabetes.csv", "Diabetes set"))
datasets.append(Dataset("datasets\wine.csv", "Wine set"))
datasets.append(Dataset("datasets\german.csv", "German set"))
datasets.append(Dataset("datasets\popfailures.csv","Popfailures set"))

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


#step 2 - random subspace approach
means = []
numberOfSubspaces = 15
for d in range(len(datasets)):
    subspaceDatasets = []
    for i in range(numberOfSubspaces):
        subspaceDatasets.append(datasets[d].GetRandomSubspaceDataset())
    subspaceScores = []

    for i in range(len(subspaceDatasets)):
        classifiers = [
            KNeighborsClassifier(),
            GaussianNB(),
            DecisionTreeClassifier(),
            SVC(gamma='auto')
        ]
        predictions = []
        for j in range(len(classifiers)):
            classifiers[j].fit(subspaceDatasets[i].X_train, subspaceDatasets[i].y_train)
            predictions.append(classifiers[j].predict(subspaceDatasets[i].X_test))
            subspaceScores.append(accuracy_score(subspaceDatasets[i].y_test, predictions[j]))
    for i in range(len(classifiers)):
        means.append(mean(subspaceScores[i*numberOfSubspaces:(i+1)*numberOfSubspaces]))

comparasion = []
for i in range(len(scores)):
    scores[i] = round(scores[i],2)
    means[i] = round(means[i],2)
    comparasion.append(round(scores[i]/means[i],2))

for i in range(len(datasets)):
    print("Results: for " + datasets[i].name)
    print("Normal approach ")
    print("KNN, GaussianNB, Decision Tree, SVC")
    print(scores[i*len(datasets):(i+1)*len(datasets)])
    print("Subspace approach:")
    print("KNN, GaussianNB, Decision Tree, SVC")
    print(means[i*len(datasets):(i+1)*len(datasets)])
    print("Normal approach / Subspace approach")
    print(comparasion[i*len(datasets):(i+1) * len(datasets)])
    print("")
counter = 0
for i in range(len(comparasion)):
    if comparasion[i] <= 1:
        counter = counter +1
print("In " + str(counter) + " of " + str(len(comparasion)) + " tests achieved higher or equal accuracy with random subspace")