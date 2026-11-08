from config import MODEL_NAME, MODEL_PATH
from predict import create_submission
from train import (
    create_pipeline,
    evaluate_model,
    load_train_data,
    train_and_save_model
)

def main():
    # 1. Загружаем данные и создаём выбранную модель
    X, y = load_train_data()
    pipeline = create_pipeline(MODEL_NAME)

    # 2. Оцениваем эту модель с помощью кросс-валидации
    cv_scores = evaluate_model(pipeline, X, y, MODEL_NAME)

    print(MODEL_NAME.upper())

    for fold, score in enumerate(cv_scores, start=1):
        print(f"Fold {fold}: {score * 100:.2f}%")

    print(f"\nMean CV: {cv_scores.mean() * 100:.2f}%")
    print(f"Std CV: {cv_scores.std() * 100:.2f} п.п.")

    # 3. Обучаем ту же модель на всём train
    model_data = train_and_save_model(pipeline, X, y)

    print(f"\nОбучено на {len(X)} пассажирах")
    print(f"Модель сохранена: {MODEL_PATH}")

    # 4. Создаём submission этой же моделью
    create_submission(model_data)


if __name__ == "__main__":
    main()
