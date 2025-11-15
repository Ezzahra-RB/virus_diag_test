import pandas as pd
from utils.preprocessing import preprocess_data

def load_dataset(path):
    df = pd.read_csv(path)
    X, y = preprocess_data(df)
    return X, y
