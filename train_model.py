import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import os

def load_train_data(path="data/"):
    """Загружает тренировочные данные из папки"""
    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(path, "y_train.csv"))
    return X_train, y_train

def load_test_data(path="data/"):
    """Загружает тестовые данные из папки"""
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(path, "y_test.csv"))
    return X_test, y_test

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test, params=None):
    """Обучает модель и логирует все параметры, метрики и саму модель в MLflow"""
    if params is None:
        params = {}

    with mlflow.start_run(run_name=model_name):
        # Логируем параметры
        mlflow.log_params(params)

        # Обучаем модель
        model.set_params(**params)
        model.fit(X_train, y_train.values.ravel()) # .ravel() для преобразования y в 1D array

        # Делаем предсказания
        y_pred = model.predict(X_test)

        # Вычисляем метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Логируем метрики
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Логируем модель как артефакт
        mlflow.sklearn.log_model(model, name="model")

        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return model

if __name__ == "__main__":
    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()

    mlflow.set_tracking_uri("file:///"+ os.path.abspath("mlruns"))
    mlflow.set_experiment("Iris_Classification")

    # Модель 1: Логистическая регрессия
    lr_params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 42}
    lr_model = LogisticRegression()
    train_and_log_model(lr_model, "Logistic Regression", X_train, y_train, X_test, y_test, lr_params)

    # Модель 2: Случайный лес с разными параметрами
    rf_params_1 = {"n_estimators": 50, "random_state": 42}
    rf_model_1 = RandomForestClassifier()
    train_and_log_model(rf_model_1, "Random Forest (n_estimators=50)", X_train, y_train, X_test, y_test, rf_params_1)

    # Модель 3: Случайный лес с другими параметрами
    rf_params_2 = {"n_estimators": 100, "random_state": 42}
    rf_model_2 = RandomForestClassifier()
    train_and_log_model(rf_model_2, "Random Forest (n_estimators=100)", X_train, y_train, X_test, y_test, rf_params_2)

    print("Обучение завершено! Результаты записаны в MLflow.")