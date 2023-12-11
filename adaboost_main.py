from adaboost import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def main():
    df = pd.read_csv('wine_binned.csv')
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    print(type(X_train))
    print(type(y_train))
    clf = AdaBoostClassifier(n_estimators=100, min_samples_split=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)

    print(f'{accuracy_score(y_pred, y_train):.2f}')

if __name__ == '__main__':
    main()
