import math
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'decision-tree'))
from decision_tree import DecisionTreeClassifier_2 # type: ignore

class Stump(DecisionTreeClassifier_2):
    
    def __init__(self, min_samples_split):
        super().__init__(max_depth=1, min_samples_split=min_samples_split)
        self.weight = None



class AdaBoostClassifier:
    
    def __init__(self, n_estimators=100, min_samples_split=2):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.stumps = []
        self.sample_weights = []

    def _build_forest_stump(self, X, y):
        stump = Stump(self.min_samples_split)
        stump.fit(X, y)
        return stump

    def _build_forest(self, X, y):
        self.sample_weights = np.ones(len(X)) / len(X)

        for _ in range(self.n_estimators):
            random_indices = np.random.choice(len(X), len(X), p=self.sample_weights)
            stump = self._build_forest_stump(X[random_indices], y[random_indices])
            y_pred = stump.predict(X)

            misclassified = np.array([int(i) for i in (y_pred != y)])
            error = np.sum(self.sample_weights * misclassified) / np.sum(self.sample_weights)
            stump.weight = 0.5 * math.log((1 - error) / error)

            self.sample_weights[misclassified == 0] *= np.exp(-stump.weight)
            self.sample_weights[misclassified == 1] *= np.exp(stump.weight)
            self.sample_weights /= np.sum(self.sample_weights)

            self.stumps.append(stump)

    def fit(self, X, y):
        self._build_forest(np.array(X), np.array(y))

    def predict(self, X):
        return np.array([self._make_prediction(np.array(x)) for x in np.array(X)])

    def _make_prediction(self, x):
        stump_preds = {}
        for stump in self.stumps:
            pred = stump.predict([x])[0]
            if pred in stump_preds:
                stump_preds[pred] += stump.weight
            else:
                stump_preds[pred] = stump.weight
        return max(stump_preds, key=stump_preds.get)