# Estimators
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Others
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Import data from files
#step 1 - full data approach
#Initialize Classifiers
#Calculate accuracy scores
#step 2 - random subspace approach
#divide random subspaces of datasets
#calculate accuracy scores for each set
#compare results: normal vs random subspace