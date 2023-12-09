from adaboost import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('../data/glass_binned.csv')
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

clf = AdaBoostClassifier(n_estimators=100, min_samples_split=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'{accuracy_score(y_pred, y_test):.2f}')
