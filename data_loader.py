import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def load_data():
    """Загрузка датасета Iris, возвращает DataFrame"""
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

def preprocess_data(data):
    """Предобработка данных"""
    # Данные уже чистые, но для общности пускай будет
    return data

def split_data(data):
    """Разделение данных на признаки и целевую переменную, а затем на train/test"""
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test, path="data/"):
    """Сохранение данных в указанную директорию"""
    os.makedirs(path, exist_ok=True)
    X_train.to_csv(os.path.join(path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(path, "y_test.csv"), index=False)

if __name__ == "__main__":
    df = load_data()
    df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    save_data(X_train, X_test, y_train, y_test)
    print("Данные загружены!")