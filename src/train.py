import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


def train_model():
    df = pd.read_csv("data/sample.csv")
    X = df[["feature1", "feature2"]]
    y = df["target"]

    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    return model


if __name__ == "__main__":
    train_model()
