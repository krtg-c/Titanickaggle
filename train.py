from pathlib import Path
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# 1. Пути к файлам
BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_400.joblib"


# 2. Загружаем данные
train = pd.read_csv(TRAIN_PATH)


# 3. Выбираем признаки
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


# 4. Разделяем признаки по типам
numeric_features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare"
]

categorical_features = [
    "Sex", "Embarked"
]


# 5. Настраиваем предобработку
numeric_transformer = SimpleImputer(
    strategy="median"
)

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 6. Создаём XGBoost
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


# 7. Объединяем предобработку и модель
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgboost_model)
])


# 8. Обучаем модель
pipeline.fit(X, y)


# 9. Сохраняем модель
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

model_data = {
    "pipeline": pipeline,
    "features": features
}

joblib.dump(model_data, MODEL_PATH)


print("XGBOOST")
print(f"Обучено на {len(X)} пассажирах")
print(f"Количество деревьев: {xgboost_model.n_estimators}")
print(f"Модель сохранена: {MODEL_PATH}")
