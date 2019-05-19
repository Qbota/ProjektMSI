from SubspaceClassifiers import SubspaceClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from Dataset import Dataset
from sklearn import base
from numpy import shape
datasets = []
datasets.append(Dataset("datasets\diabetes.csv", "Diabetes set"))
datasets.append(Dataset("datasets\wine.csv", "Wine set"))
datasets.append(Dataset("datasets\german.csv", "German set"))
datasets.append(Dataset("datasets\popfailures.csv","Popfailures set"))
datasets.append(Dataset("datasets\heart.csv","Heart set"))
datasets.append(Dataset("datasets\liver.csv","Liver set"))
classifiers = [
        KNeighborsClassifier(),
        GaussianNB(),
        DecisionTreeClassifier(),
        SVC(gamma='auto')
    ]

#step 1 - full data approach
scores = []
for i in range(len(datasets)):
    predictions = []
    for j in range(len(classifiers)):
        member_clf = base.clone(classifiers[j])
        member_clf.fit(datasets[i].X_train, datasets[i].y_train)
        predictions.append(member_clf.predict(datasets[i].X_test))
        scores.append(accuracy_score(datasets[i].y_test, predictions[j]))



#step 2 - subspace approach
subspaceScores = []
subspacePredictions = []
numberOfSubspaces = 20
for i in range(len(datasets)):
    subspacePredictions = []
    for j in range(len(classifiers)):
        member_clf = base.clone(classifiers[j])
        sub_clf = SubspaceClassifier(numberOfSubspaces, member_clf)
        sub_clf.fit(datasets[i].X_train, datasets[i].y_train)
        subspacePredictions.append(sub_clf.predict(datasets[i].X_test))
        subspaceScores.append(accuracy_score(datasets[i].y_test, subspacePredictions[j]))


#step 3 comparasion of results
comparasion = []
for i in range(len(scores)):
    comparasion.append(round(scores[i]/subspaceScores[i],2))
    scores[i] = round(scores[i], 2)
    subspaceScores[i] = round(subspaceScores[i], 2)

for i in range(len(datasets)):
    print("Results: for " + datasets[i].name)
    print("Normal approach ")
    print("KNN, GaussianNB, Decision Tree, SVC")
    print(scores[i*len(classifiers):(i+1)*len(classifiers)])
    print("Subspace approach:")
    print("KNN, GaussianNB, Decision Tree, SVC")
    print(subspaceScores[i*len(classifiers):(i+1)*len(classifiers)])
#    print("Normal approach / Subspace approach")
#    print(comparasion[i*len(classifiers):(i+1)*len(classifiers)])
    print("")
counter = 0
for i in range(len(comparasion)):
    if comparasion[i] <= 1:
        counter = counter +1
print("In " + str(counter) + " of " + str(len(comparasion)) + " tests subspace algorithm achieved higher or equal accuracy")