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


BASE_DIR = Path(__file__).resolve().parent

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


print("Random Forest")
print(f"\nMean CV: {cv_scores.mean() * 100:.2f}%")
print(f"Std CV: {cv_scores.std() * 100:.2f} п.п.")
print(f"Min CV: {cv_scores.min() * 100:.2f}%")
print(f"Max CV: {cv_scores.max() * 100:.2f}%")


