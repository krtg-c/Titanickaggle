from pathlib import Path


# Корневая папка проекта
BASE_DIR = Path(__file__).resolve().parent


# "logistic_regression", "random_forest", "xgboost", "catboost", "mlp"
MODEL_NAME = "catboost"


# Пути к данным и результатам
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / f"{MODEL_NAME}.joblib"
SUBMISSION_PATH = (
    BASE_DIR / "submissions" / f"submission_{MODEL_NAME}.csv"
)


# Признаки и целевая переменная
FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

NUMERIC_FEATURES = [
    "Pclass", "Age", "SibSp", "Parch", "Fare"
]

CATEGORICAL_FEATURES = [
    "Sex", "Embarked"
]

TARGET = "Survived"


# Настройки кросс-валидации
CV_FOLDS = 10
RANDOM_STATE = 42


# Настройки моделей
MODEL_PARAMS = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE
    },
    "random_forest": {
        "n_estimators": 200,
        "random_state": RANDOM_STATE
    },
    "xgboost": {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 1
    },
    "catboost": {
        "iterations": 400,
        "depth": 4,
        "learning_rate": 0.03,
        "loss_function": "Logloss",
        "eval_metric": "Accuracy",
        "random_state": RANDOM_STATE,
        "allow_writing_files": False,
        "verbose": False
    },
    "mlp": {
        "epochs": 80,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "random_state": RANDOM_STATE
    }
}

SUPPORTED_MODELS = list(MODEL_PARAMS.keys())
