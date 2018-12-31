import pickle
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

from .load_data import do_rebalance


class MLP():
    """Generate Multiayer Perceptron Model.

    Useing scikit-learn to genearate MLP Model.

    Attributes:
        clf: MLP Classifier.
    """

    def __init__(self):
        self.clf = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(20,20,), max_iter=10000, verbose=True, learning_rate_init=.1)

    def train(self, X_train, Y_train):
        X_train, Y_train = do_rebalance(X_train, Y_train)
        self.clf.fit(X_train, Y_train)

    def test(self, X_test, Y_test):
        Y_prob = self.clf.predict_proba(X_test)
        auc = metrics.roc_auc_score(Y_test, Y_prob[:, 1])
        print("model's auc is {}".format(auc))

    def predict(self, X):
        Y_prob = self.clf.predict_proba(X)[:, 1]
        return Y_prob

    def load_model(self, model_path):
        with open(model_path, "rb+") as file:
            self.clf = pickle.load(file)

    def save_model(self, model_path):
        with open(model_path, "wb+") as file:
            pickle.dump(self.clf, file)