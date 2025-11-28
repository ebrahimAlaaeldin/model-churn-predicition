from sklearn.tree import DecisionTreeClassifier
from typing import List

def decisionTreeAccuracy(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    pridict_y = clf.predict(X_test)
    accuracy_dt = clf.score(X_test,y_test)
    return accuracy_dt

