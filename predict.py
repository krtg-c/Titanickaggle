import joblib
import pandas as pd

from config import MODEL_PATH, SUBMISSION_PATH, TEST_PATH


def create_submission(model_data=None):
    """Создаёт Kaggle submission с помощью обученной модели."""
    test = pd.read_csv(TEST_PATH)

    if model_data is None:
        model_data = joblib.load(MODEL_PATH)

    pipeline = model_data["pipeline"]
    features = model_data["features"]
    model_name = model_data["model_name"]

    predictions = pipeline.predict(test[features])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions.astype(int)
    })

    assert submission.shape == (len(test), 2)
    assert submission["Survived"].isin([0, 1]).all()

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"\n{model_name.upper()} SUBMISSION")
    print(submission.head())
    print(f"Количество предсказаний: {len(submission)}")
    print(f"Файл сохранён: {SUBMISSION_PATH}")

    return submission


if __name__ == "__main__":
    create_submission()
