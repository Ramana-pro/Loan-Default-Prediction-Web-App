from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_models

df = load_data("data/loan_data.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)

train_models(X_train, X_test, y_train, y_test)
