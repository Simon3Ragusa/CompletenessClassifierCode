import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd


def classification(X, y, classifier, param):

    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)
    clf = DecisionTreeClassifier()
    if classifier == "DecisionTree":
        clf = DecisionTreeClassifier(max_depth=int(param), random_state=0)
    elif classifier == "LogisticRegression":
        clf = LogisticRegression(C=param, random_state=0)
    elif classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=int(param))
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(max_depth=int(param), random_state=0)
    elif classifier == "AdaBoost":
        clf = AdaBoostClassifier(n_estimators=int(param), random_state=0)
    elif classifier == "SVC":
        clf = SVC(max_iter=10000, C=param, random_state=0)

    # print("Training for "+classifier+"...")
    cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=0)
    model_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")
    f1_mean = model_scores.mean()
    return f1_mean
