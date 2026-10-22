from pathlib import Path
import pandas as pd
import joblib


# 1. Пути к файлам
BASE_DIR = Path(__file__).resolve().parent

TEST_PATH = BASE_DIR / "data" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_400.joblib"

SUBMISSION_PATH = (
    BASE_DIR / "submissions" / "submission_xgboost.csv"
)


# 2. Загружаем тестовые данные
test = pd.read_csv(TEST_PATH)


# 3. Загружаем сохранённую модель
model_data = joblib.load(MODEL_PATH)

pipeline = model_data["pipeline"]
features = model_data["features"]


# 4. Подготавливаем признаки
X_test = test[features]


# 5. Получаем предсказания
predictions = pipeline.predict(X_test)


# 6. Формируем submission
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions.astype(int)
})


# 7. Проверяем результат
assert submission.shape == (418, 2)
assert submission["Survived"].isin([0, 1]).all()


# 8. Сохраняем файл
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

submission.to_csv(
    SUBMISSION_PATH,
    index=False
)


print("XGBOOST SUBMISSION")
print(submission.head())
print(f"\nКоличество предсказаний: {len(submission)}")
print(f"Файл сохранён: {SUBMISSION_PATH}")
