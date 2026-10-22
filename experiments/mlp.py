
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler




# 1. Загружаем данные

BASE_DIR = Path(__file__).resolve().parents[1]
train = pd.read_csv(BASE_DIR / "data" / "train.csv")

features = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

numeric_features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare"
]

categorical_features = ["Sex", "Embarked"]

X = train[features]
y = train["Survived"].to_numpy()


# 2. Настройки эксперимента

EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.001

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scores = []


# 3. Кросс-валидация

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y), start=1
):

    # Данные текущего фолда
    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y[train_idx]
    y_valid = y[valid_idx]

    # Предобработка признаков
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]), categorical_features)
    ])

    # Учим предобработку только на train
    X_train_processed = preprocessor.fit_transform(
        X_train
    ).astype(np.float32)

    X_valid_processed = preprocessor.transform(
        X_valid
    ).astype(np.float32)

    # Преобразуем данные в тензоры
    X_train_tensor = torch.tensor(X_train_processed)
    y_train_tensor = torch.tensor(
        y_train, dtype=torch.float32
    ).reshape(-1, 1)

    X_valid_tensor = torch.tensor(X_valid_processed)

    # Формируем батчи
    dataset = TensorDataset(
        X_train_tensor, y_train_tensor
    )

    generator = torch.Generator().manual_seed(42 + fold)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    # Создаём новую сеть для каждого фолда
    torch.manual_seed(42 + fold)

    input_size = X_train_processed.shape[1]

    model = nn.Sequential(
        nn.Linear(input_size, 32),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(32, 16),
        nn.ReLU(),

        nn.Linear(16, 1)
    )

    # Функция потерь и оптимизатор
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    # 4. Обучаем нейросеть
    for epoch in range(EPOCHS):
        model.train()

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

    # 5. Проверяем точность
    model.eval()

    with torch.no_grad():
        logits = model(X_valid_tensor)
        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities.numpy().ravel() >= 0.5
        ).astype(int)

    score = accuracy_score(y_valid, predictions)
    scores.append(score)

    print(f"Fold {fold}: {score * 100:.2f}%")


# 6. Итоги эксперимента

scores = np.array(scores)

print("MLP RESULTS")
print(f"Mean CV: {scores.mean() * 100:.2f}%")
print(f"Std CV: {scores.std() * 100:.2f} п.п.")
print(f"Min CV: {scores.min() * 100:.2f}%")
print(f"Max CV: {scores.max() * 100:.2f}%")
