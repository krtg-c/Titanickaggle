from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:.2f}".format)

# Корневая папка проекта
BASE_DIR = Path(__file__).resolve().parent

# Пути к данным
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"

# Загружаем данные
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Размер обучающей выборки:", train.shape)
print("Размер тестовой выборки:", test.shape)

print("\nПервые 5 строк train.csv:")
print(train.head())

print("\nИнформация о колонках:")
train.info()

print("\nПропущенные значения:")
print(train.isna().sum())


# 1. Сколько пассажиров выжило
print("\nРаспределение выживаемости:")
print(train["Survived"].value_counts())

# 2. Процент выживших
print("\nПроцент выживших:")
print(train["Survived"].value_counts(normalize=True) * 100)

# 3. Выживаемость в зависимости от пола
print("\nВыживаемость по полу:")
print(train.groupby("Sex")["Survived"].mean() * 100)

# 4. Выживаемость в зависимости от класса
print("\nВыживаемость по классам:")
print(train.groupby("Pclass")["Survived"].mean() * 100)

# 5. Статистика возраста
print("\nСтатистика возраста:")
print(train["Age"].describe())

# 6. Пропуски в тестовой выборке
print("\nПропуски в test.csv:")
print(test.isna().sum())
