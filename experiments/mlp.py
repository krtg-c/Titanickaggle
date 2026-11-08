from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Добавляем корневую папку проекта для импорта config.py
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    FEATURES,
    MODEL_PARAMS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TRAIN_PATH
)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Небольшая нейросеть с интерфейсом sklearn."""

    def __init__(
        self,
        epochs=80,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.0001,
        random_state=42
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        """Создаёт и обучает нейросеть."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        torch.manual_seed(self.random_state)

        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        generator = torch.Generator().manual_seed(self.random_state)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator
        )

        input_size = X.shape[1]

        self.model_ = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        for _ in range(self.epochs):
            self.model_.train()

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Возвращает класс 0 или 1."""
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X)

        self.model_.eval()

        with torch.no_grad():
            logits = self.model_(X_tensor)
            probabilities = torch.sigmoid(logits)

        return (
            probabilities.numpy().ravel() >= 0.5
        ).astype(int)


def main():
    """Запускает отдельный CV-эксперимент с MLP."""
    train = pd.read_csv(TRAIN_PATH)
    X = train[FEATURES]
    y = train[TARGET]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            TorchMLPClassifier(**MODEL_PARAMS["mlp"])
        )
    ])

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("MLP RESULTS")

    for fold, score in enumerate(cv_scores, start=1):
        print(f"Fold {fold}: {score * 100:.2f}%")

    print(f"\nMean CV: {cv_scores.mean() * 100:.2f}%")
    print(f"Std CV: {cv_scores.std() * 100:.2f} п.п.")
    print(f"Min CV: {cv_scores.min() * 100:.2f}%")
    print(f"Max CV: {cv_scores.max() * 100:.2f}%")


if __name__ == "__main__":
    main()
