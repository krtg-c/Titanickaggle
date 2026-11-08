import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler
)
from xgboost import XGBClassifier

from config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    FEATURES,
    MODEL_NAME,
    MODEL_PARAMS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SUPPORTED_MODELS,
    TARGET,
    TRAIN_PATH
)
from experiments.mlp import TorchMLPClassifier


def load_train_data():
    """Загружает обучающие данные и разделяет признаки и цель."""
    train = pd.read_csv(TRAIN_PATH)
    X = train[FEATURES]
    y = train[TARGET]
    return X, y


def create_preprocessor(scale_numeric=False):
    """Создаёт предобработку для sklearn-моделей."""
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps)

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

    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])


def create_logistic_regression_pipeline():
    """Создаёт pipeline с логистической регрессией."""
    preprocessor = create_preprocessor(scale_numeric=True)
    model = LogisticRegression(
        **MODEL_PARAMS["logistic_regression"]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def create_random_forest_pipeline():
    """Создаёт pipeline с Random Forest."""
    preprocessor = create_preprocessor()
    model = RandomForestClassifier(
        **MODEL_PARAMS["random_forest"]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def create_xgboost_pipeline():
    """Создаёт pipeline с XGBoost."""
    preprocessor = create_preprocessor()

    model = XGBClassifier(**MODEL_PARAMS["xgboost"])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def prepare_catboost_data(data):
    """Заполняет пропуски в категориальных признаках CatBoost."""
    data = data.copy()

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].fillna("Unknown")

    return data


def create_catboost_pipeline():
    """Создаёт pipeline с предобработкой и CatBoost."""
    preprocessor = FunctionTransformer(
        prepare_catboost_data,
        validate=False
    )

    model = CatBoostClassifier(
        cat_features=CATEGORICAL_FEATURES,
        **MODEL_PARAMS["catboost"]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def create_mlp_pipeline():
    """Создаёт pipeline с предобработкой и нейросетью."""
    preprocessor = create_preprocessor(scale_numeric=True)
    model = TorchMLPClassifier(**MODEL_PARAMS["mlp"])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def create_pipeline(model_name):
    """Выбирает модель по значению из config.py."""
    if model_name == "logistic_regression":
        return create_logistic_regression_pipeline()

    if model_name == "random_forest":
        return create_random_forest_pipeline()

    if model_name == "xgboost":
        return create_xgboost_pipeline()

    if model_name == "catboost":
        return create_catboost_pipeline()

    if model_name == "mlp":
        return create_mlp_pipeline()

    raise ValueError(
        f"Неизвестная модель. Доступные модели: {SUPPORTED_MODELS}"
    )


def evaluate_model(pipeline, X, y, model_name=MODEL_NAME):
    """Считает accuracy с помощью стратифицированной CV."""
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # CatBoost нельзя клонировать через cross_val_score
    # в некоторых версиях scikit-learn, поэтому считаем CV вручную.
    if model_name == "catboost":
        scores = []

        for train_index, valid_index in cv.split(X, y):
            fold_pipeline = create_pipeline(model_name)

            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            fold_pipeline.fit(X_train, y_train)
            predictions = fold_pipeline.predict(X_valid)
            scores.append(accuracy_score(y_valid, predictions))

        return pd.Series(scores).to_numpy()

    return cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )


def train_and_save_model(pipeline, X, y):
    """Обучает pipeline на всём train и сохраняет его."""
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "pipeline": pipeline,
        "features": FEATURES,
        "model_name": MODEL_NAME
    }

    joblib.dump(model_data, MODEL_PATH)
    return model_data


if __name__ == "__main__":
    X, y = load_train_data()
    pipeline = create_pipeline(MODEL_NAME)
    train_and_save_model(pipeline, X, y)

    print(MODEL_NAME.upper())
    print(f"Обучено на {len(X)} пассажирах")
    print(f"Модель сохранена: {MODEL_PATH}")
