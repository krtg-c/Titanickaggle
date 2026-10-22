from pathlib import Path
import pandas as pd
from catboost import Pool, cv as catboost_cv
from sklearn.model_selection import StratifiedKFold



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

categorical_features = ["Sex", "Embarked"]

X = train[features].copy()
y = train["Survived"]


# 3. Обрабатываем пропуски в категориях
X["Embarked"] = X["Embarked"].fillna("Unknown")


# 4. Создаём Pool для CatBoost
train_pool = Pool(
    data=X,
    label=y,
    cat_features=categorical_features
)


# 5. Параметры CatBoost
catboost_params = {
    "iterations": 500,
    "depth": 4,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "random_seed": 42,
    "allow_writing_files": False
}


# 6. Настраиваем кросс-валидацию
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

folds = list(cv.split(X, y))


# 7. Проводим кросс-валидацию
cv_results = catboost_cv(
    pool=train_pool,
    params=catboost_params,
    folds=folds,
    verbose=False
)


# 8. Выводим результат после 500 деревьев
catboost_mean = cv_results["test-Accuracy-mean"].iloc[-1]
catboost_std = cv_results["test-Accuracy-std"].iloc[-1]

print("CATBOOST")
print(f"Mean CV: {catboost_mean * 100:.2f}%")
print(f"Std CV: {catboost_std * 100:.2f} п.п.")


# 9. Сравниваем количество деревьев
print("NUMBER OF TREES")

for n in [50, 100, 200, 300, 400, 450, 500]:
    score = cv_results["test-Accuracy-mean"].iloc[n - 1]
    std = cv_results["test-Accuracy-std"].iloc[n - 1]

    print(
        f"{n} деревьев: "
        f"Accuracy = {score * 100:.2f}%, "
        f"Std = {std * 100:.2f} п.п."
    )


# 10. Находим лучшую итерацию
best_idx = cv_results["test-Accuracy-mean"].idxmax()

best_iteration = int(
    cv_results.loc[best_idx, "iterations"]
) + 1

best_score = cv_results.loc[
    best_idx, "test-Accuracy-mean"
]

print("BEST ITERATION")
print(f"Лучшая итерация: {best_iteration}")
print(f"Максимальная Mean CV: {best_score * 100:.2f}%")
