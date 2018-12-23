import pickle
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from .load_data import load_data


class RandomForest():
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=0, max_features="log2")
        # self.clf = RandomForestClassifier(
        #     n_estimators=10, max_depth=5, random_state=0, max_features="log2")

    def train(self, train_csv_path):
        X_train, Y_train = load_data(train_csv_path)
        self.clf.fit(X_train, Y_train)

    def test(self, test_csv_path):
        X_test, Y_test = load_data(test_csv_path)
        Y_prob = self.clf.predict_proba(X_test)
        auc = metrics.roc_auc_score(Y_test, Y_prob[:, 1])
        print("model's auc is {}".format(auc))

    def predict(self, X):
        Y_prob = self.clf.predict_proba(X)
        return Y_prob

    def load_model(self, model_path):
        with open(model_path, "rb+") as file:
            self.clf = pickle.load(file)

    def save_model(self, model_path):
        with open(model_path, "wb+") as file:
            pickle.dump(self.clf, file)