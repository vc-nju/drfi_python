import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from .load_data import load_data


class RandomForest():
    def __init__(self):
        X_train, Y_train, X_test, Y_test = load_data()
        clf = RandomForest.generate_model(X_train, Y_train)
        RandomForest.test_model(clf, X_test, Y_test)

    @staticmethod
    def generate_model(X_train, Y_train):
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=0)
        clf.fit(X_train, Y_train)
        return clf

    @staticmethod
    def test_model(clf, X_test, Y_test):
        Y_predict = clf.predict(X_test)
        Y_prob = clf.predict_proba(X_test)
        auc = metrics.roc_auc_score(Y_test, Y_prob[:,1])
        print(auc)