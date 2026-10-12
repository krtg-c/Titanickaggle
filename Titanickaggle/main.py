from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / 'data' / 'train.csv'
TEST_DIR = BASE_DIR / 'data' / 'test.csv'

train = pd.read_csv(TRAIN_DIR)
test = pd.read_csv(TEST_DIR)

print('train size', train.shape, 'test size', test.shape)

print('\nFirst 5 string\n', train.head(5))

print(train.info)

print(train.isna().sum())
