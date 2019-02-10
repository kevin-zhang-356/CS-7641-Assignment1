import csv
import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import sklearn.tree as tree
from scipy.stats import randint as sp_randint
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

#################################################
#Wine data set

data = pd.read_csv('winequality-white.csv')
X = data.loc[:, data.columns != "quality"]
y = data.loc[:, "quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
features = list(X_train.columns.values)

# #Hyperparameter tuning using randomizedSearchCV
# clf = DecisionTreeClassifier()
# param_dist = {"max_depth": [3, 5, 10],
#               "min_samples_leaf": sp_randint(1, 11),
#               "min_samples_split": sp_randint(2, 11)}

# n_iter_search = 20
# kfold = KFold(n_splits = 5, random_state = None)
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search, cv=kfold)
# random_search.fit(X, y)
# print("Best hyperparameters are: ")
# print(random_search.best_params_)

# Plot the learning curve
list1=[]
list2=[]
for i in range(1,95):
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf = 6, min_samples_split= 6)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf = clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list1)),list1,label="Training Accuracy")
plt.plot(range(len(list2)),list2,label="Testing Accuracy")
plt.xlabel("Training sample percentage")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

# visualization of decision tree
# clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf = 6, min_samples_split= 6)
# clf = clf.fit(X_train, y_train)
# test_predict = clf.predict(X_test)
# with open("dt.dot", "w") as f:
#     f = tree.export_graphviz(clf, out_file=f)

#Neural Network

#Hyperparameter tuning using randomizedSearchCV
# clf = MLPClassifier()
# param_dist = {"solver" : ['lbfgs', 'sgd', 'adam'],
#               "activation" : ['identity', 'logistic', 'tanh', 'relu'],
#               "alpha": [1e-5, 1e-4],
#               "hidden_layer_sizes": [(50,50,50), (50,100,50)]
#               }

# n_iter_search = 20
# kfold = KFold(n_splits = 5, random_state = None)
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search, cv=kfold)
# random_search.fit(X, y)
# print("Best hyperparameters are: ")
# print(random_search.best_params_)

#Neural network classifier learning curve
list1=[]
list2=[]
for i in range(1,95):
    clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(5), activation='logistic')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list1)),list1,label="Training Accuracy")
plt.plot(range(len(list2)),list2,label="Testing Accuracy")
plt.xlabel("Training sample percentage")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

#Boosted DT classifier
# #Hyperparameter tuning using randomizedSearchCV
# clf = AdaBoostClassifier()
# param_dist = {"n_estimators" : sp_randint(1, 100),
#               "learning_rate" : [1.0, 5.0, 10.0]
#               }
# n_iter_search = 20
# kfold = KFold(n_splits = 5, random_state = None)
# random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                    n_iter=n_iter_search, cv=kfold)
# random_search.fit(X, y)
# print("Best hyperparameters are: ")
# print(random_search.best_params_)

list1=[]
list2=[]
for i in range(1,95):
    clf = clf = AdaBoostClassifier(n_estimators=98, learning_rate = 1.0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list1)),list1,label="Training Accuracy")
plt.plot(range(len(list2)),list2,label="Testing Accuracy")
plt.xlabel("Training sample percentage")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

#SVM classifier

#Normalize data to increase speed
Xn = normalize(X)

# #GridSearchCV
# SVCtune = svm.SVC()
# SVCparams = [{'kernel': ['rbf'],'C': [0.1, 1, 10]},{'kernel': ['linear'], 'C': [0.1, 1, 10]}]
# SVCgs = GridSearchCV(SVCtune, SVCparams, scoring = "accuracy", cv = 10)
# SVCgs.fit(Xn, y)
# print("Best params: ", SVCgs.best_params_)

#SVM learning curve with RBF kernel
list1=[]
list2=[]
for i in range(1,95):
    clf = svm.SVC(kernel="rbf", C=0.1)
    X_train, X_test, y_train, y_test = train_test_split(Xn, y, random_state=0, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
# plt.ylim(ymin=0,ymax=1.1)
plt.plot(range(len(list1)),list1,label="Training Accuracy")
plt.plot(range(len(list2)),list2,label="Testing Accuracy")
plt.xlabel("Training sample percentage")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

# KNN classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
KNN_list=[]
list2=[]
for K in range(1,50):
    clf = KNeighborsClassifier(K, weights="distance")
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    clf = clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)
    KNN_list.append(accuracy_score(y_test, test_predict))
    list2.append(sum(scores)/len(scores))
plt.plot(range(len(KNN_list)),KNN_list,label="Testing Accuracy")
plt.plot(range(len(list2)),list2,label="Cross Validation")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()

#learning curve of KNN classifier with K=9
list1=[]
list2=[]
for i in range(1,95):
    clf = KNeighborsClassifier(9, weights="distance")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list1)),list1,label="Training Accuracy")
plt.plot(range(len(list2)),list2,label="Testing Accuracy")
plt.xlabel("Training sample percentage")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()