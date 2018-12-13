import numpy as np
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier

from .load_data import load_data


class MLP():
    def __init__(self):
        X_train, Y_train, X_test, Y_test = load_data()
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_train, Y_train,test_size=0.2,random_state=27)
        clf = MLP.generate_model(X_train, Y_train)
        MLP.test_model(clf, X_test, Y_test)

    @staticmethod
    def generate_model(X_train, Y_train):
        mlp = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(
            50, 50), random_state=1, max_iter=100, verbose=10, learning_rate_init=.1)
        mlp.fit(X_train, Y_train)
        return mlp

    @staticmethod
    def test_model(mlp, X_test, Y_test):
        print(mlp.score(X_test, Y_test))
