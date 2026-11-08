# Titanic — Machine Learning from Disaster

Учебный проект по задаче классификации с Kaggle. Модель должна определить,
выжил пассажир «Титаника» или нет.

## Данные и признаки

Обучающие и тестовые данные находятся в папке `data/`.
В основном pipeline используются признаки:

- класс билета (`Pclass`);
- пол (`Sex`);
- возраст (`Age`);
- количество родственников (`SibSp`, `Parch`);
- стоимость билета (`Fare`);
- порт посадки (`Embarked`).

Целевая переменная — `Survived`. Основная метрика — accuracy.

## Установка

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать его в Windows:

```bash
.venv\Scripts\activate
```

В Linux или macOS:

```bash
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

## Основной запуск

```bash
python main.py
```

За один запуск программа:

1. загружает `train.csv`;
2. считает accuracy выбранной модели на 10 фолдах;
3. обучает эту же модель на всём train;
4. сохраняет модель в папку `models/`;
5. создаёт готовый Kaggle submission в папке `submissions/`.

Вся необходимая предобработка входит в pipeline модели. Для XGBoost это
заполнение пропусков и кодирование категорий. Во время CV предобработка
обучается отдельно внутри каждого фолда.

## Выбор модели

Модель выбирается в `config.py`:

```python
MODEL_NAME = "xgboost"
```

Доступны пять моделей:

```python
MODEL_NAME = "logistic_regression"
MODEL_NAME = "random_forest"
MODEL_NAME = "xgboost"
MODEL_NAME = "catboost"
MODEL_NAME = "mlp"
```

В этом же файле находятся параметры моделей, число фолдов и random state.
После изменения настройки нужно снова запустить `python main.py`.

`experiments/01_family_size.py` проверяет добавление нового признака
`FamilySize` к Random Forest. Это не отдельный тип модели, поэтому в конфиге
для него нет отдельного названия: сама модель по-прежнему Random Forest.

Особенности предобработки:

- Logistic Regression и MLP используют заполнение пропусков,
  OneHotEncoder и StandardScaler;
- Random Forest и XGBoost используют заполнение пропусков и OneHotEncoder;
- CatBoost получает категориальные признаки напрямую.

## Результаты

Для оценки используется `StratifiedKFold` с 10 фолдами, перемешиванием
и `random_state=42`.

1. | Модель | Mean CV | Std CV |
2. | Logistic Regression| 79.57% | 3.11 п.п. |
3. | Random Forest | 82.04% | 1.95 п.п. |
4. | CatBoost | 82.15% | 2.06 п.п. |
5. | MLP | 82.60% | 2.22 п.п. |
6. | XGBoost | **83.16%** | 2.42 п.п. |

Подробные результаты остальных экспериментов находятся в `experiments.md`.
По умолчанию выбран XGBoost, так как он показал лучший Mean CV.

## Структура проекта

- `main.py` — полный запуск CV, обучения и создания submission;
- `config.py` — выбор модели и все основные настройки;
- `train.py` — загрузка данных, создание pipeline, CV и обучение;
- `predict.py` — создание submission;
- `baseline.py` — отдельная оценка простого бейзлайна Logistic Regression;
- `EDA.ipynb` — исследовательский анализ данных с графиками и выводами;
- `experiments/mlp.py` — реализация PyTorch-нейросети и её CV-эксперимент;
- `experiments/` — остальные эксперименты с моделями и признаками;
- `experiments.md` — таблица результатов экспериментов.
