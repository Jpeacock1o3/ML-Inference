"""Train and serialize the demo iris classifier."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

OUTPUT_PATH = Path("models/classifier.joblib")


def train():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"Test accuracy : {acc:.4f}")
    print(f"CV accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    artifact = {
        "model": clf,
        "version": "v1",
        "metadata": {
            "algorithm": "RandomForestClassifier",
            "feature_names": list(iris.feature_names),
            "class_names": list(iris.target_names),
            "n_features": X.shape[1],
            "accuracy": round(float(acc), 4),
            "description": "Iris flower species classifier (demo)",
        },
    }
    joblib.dump(artifact, OUTPUT_PATH)
    print(f"Model saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    train()
