from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "train.csv"

train = pd.read_csv(TRAIN_PATH)
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

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare"
]

categorical_features = [
    "Sex", "Embarked"
]

numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_valid)

accuracy = accuracy_score(y_valid, predictions)


cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)





train_experiment = train.copy()

train_experiment["FamilySize"] = (
    train_experiment["SibSp"] +
    train_experiment["Parch"] + 1
)

features_experiment = features + ["FamilySize"]

X_experiment = train_experiment[features_experiment]

numeric_features_experiment = numeric_features + ["FamilySize"]

preprocessor_experiment = ColumnTransformer([
    ("num", numeric_transformer, numeric_features_experiment),
    ("cat", categorical_transformer, categorical_features)
])

model_experiment = Pipeline([
    ("preprocessor", preprocessor_experiment),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

experiment_scores = cross_val_score(
    model_experiment,
    X_experiment,
    y,
    cv=cv,
    scoring="accuracy"
)

baseline_mean = cv_scores.mean() * 100
experiment_mean = experiment_scores.mean() * 100

difference = experiment_mean - baseline_mean

print("FAMILY SIZE")
print(f"Baseline CV: {baseline_mean:.2f}%")
print(f"FamilySize CV: {experiment_mean:.2f}%")
print(f"Изменение: {difference:+.2f} п.п.")
print(f"Std: {experiment_scores.std() * 100:.2f} п.п.")