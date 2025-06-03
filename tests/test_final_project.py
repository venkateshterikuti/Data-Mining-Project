import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def test_logistic_regression_cross_val():
    data_path = Path(__file__).resolve().parent.parent / "Cancer_Data.csv"
    df = pd.read_csv(data_path)
    # Basic preprocessing similar to final_project.py
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')
    mean_accuracy = np.mean(scores)

    assert mean_accuracy > 0.90
