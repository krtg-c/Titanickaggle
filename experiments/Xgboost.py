from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder



# 1. Загружаем данные
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "data" / "train.csv"

train = pd.read_csv(TRAIN_PATH)


# 2. Выбираем признаки
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = train[features]
y = train["Survived"]


# 3. Разделяем признаки по типам
numeric_features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare"
]

categorical_features = [
    "Sex", "Embarked"
]


# 4. Создаём предобработку
numeric_transformer = SimpleImputer(
    strategy="median"
)

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 5. Создаём XGBoost
xgboost_model = XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=1
)


# 6. Объединяем обработку и модель
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgboost_model)
])


# 7. Настраиваем кросс-валидацию
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


# 8. Запускаем кросс-валидацию
cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


# 9. Выводим результаты
print("XGBOOST")

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score * 100:.2f}%")

print(f"\nMean CV: {cv_scores.mean() * 100:.2f}%")
print(f"Std CV: {cv_scores.std() * 100:.2f} п.п.")
print(f"Min CV: {cv_scores.min() * 100:.2f}%")
print(f"Max CV: {cv_scores.max() * 100:.2f}%")
