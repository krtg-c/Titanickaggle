from pathlib import Path
import sys


# Добавляем корневую папку проекта, чтобы использовать общий pipeline
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from train import (  # noqa: E402
    create_pipeline,
    evaluate_model,
    load_train_data
)


# Используем ту же модель и те же фолды, что и в main.py
X, y = load_train_data()
pipeline = create_pipeline("catboost")
cv_scores = evaluate_model(pipeline, X, y, "catboost")


print("CATBOOST")

for fold, score in enumerate(cv_scores, start=1):
    print(f"Fold {fold}: {score * 100:.2f}%")

print(f"\nMean CV: {cv_scores.mean() * 100:.2f}%")
print(f"Std CV: {cv_scores.std() * 100:.2f} п.п.")
print(f"Min CV: {cv_scores.min() * 100:.2f}%")
print(f"Max CV: {cv_scores.max() * 100:.2f}%")
