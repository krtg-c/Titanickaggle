import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    FEATURES,
    MODEL_PARAMS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TRAIN_PATH
)


# 1. Загружаем данные
train = pd.read_csv(TRAIN_PATH)
X = train[FEATURES]
y = train[TARGET]


# 2. Настраиваем предобработку
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES)
])


# 3. Создаём простой бейзлайн
baseline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            **MODEL_PARAMS["logistic_regression"]
        )
    )
])


# 4. Считаем accuracy на тех же фолдах, что и остальные модели
cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_scores = cross_val_score(
    baseline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print("LOGISTIC REGRESSION BASELINE")
print(f"Mean CV: {cv_scores.mean() * 100:.2f}%")
print(f"Std CV: {cv_scores.std() * 100:.2f} п.п.")
print(f"Min CV: {cv_scores.min() * 100:.2f}%")
print(f"Max CV: {cv_scores.max() * 100:.2f}%")
