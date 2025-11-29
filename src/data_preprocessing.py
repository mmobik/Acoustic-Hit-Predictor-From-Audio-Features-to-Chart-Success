"""
Data Preprocessing Module
Функции для предобработки данных датасета Spotify
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """
    Загрузка данных из CSV файла
    
    Args:
        filepath: Путь к CSV файлу
        
    Returns:
        DataFrame с загруженными данными
    """
    df = pd.read_csv(filepath)
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверка пропущенных значений
    
    Args:
        df: Исходный DataFrame
        
    Returns:
        DataFrame с информацией о пропусках
    """
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2)
    })
    return missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаление дубликатов
    
    Args:
        df: Исходный DataFrame
        
    Returns:
        DataFrame без дубликатов
    """
    initial_shape = df.shape[0]
    df_clean = df.drop_duplicates()
    removed = initial_shape - df_clean.shape[0]
    
    if removed > 0:
        print(f"Удалено {removed} дубликатов ({removed/initial_shape*100:.2f}%)")
    else:
        print("Дубликаты не обнаружены")
    
    return df_clean


def handle_outliers(df: pd.DataFrame, columns: list, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Обработка выбросов
    
    Args:
        df: Исходный DataFrame
        columns: Список колонок для обработки
        method: Метод обнаружения ('iqr' или 'zscore')
        threshold: Порог для определения выбросов
        
    Returns:
        DataFrame с обработанными выбросами
    """
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            outliers = (z_scores > threshold).sum()
            df_clean = df_clean[z_scores <= threshold]
        
        if outliers > 0:
            print(f"{col}: удалено {outliers} выбросов")
    
    return df_clean


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = 'standard') -> tuple:
    """
    Масштабирование признаков
    
    Args:
        X_train: Обучающая выборка
        X_test: Тестовая выборка
        method: Метод масштабирования ('standard' или 'minmax')
        
    Returns:
        Кортеж (X_train_scaled, X_test_scaled, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method должен быть 'standard' или 'minmax'")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Преобразование обратно в DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Разделение данных на train и test
    
    Args:
        df: Исходный DataFrame
        target_col: Название целевой колонки
        test_size: Размер тестовой выборки
        random_state: Seed для воспроизводимости
        
    Returns:
        Кортеж (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def get_numeric_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Получение списка числовых колонок
    
    Args:
        df: DataFrame
        exclude: Список колонок для исключения
        
    Returns:
        Список числовых колонок
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude:
        numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    return numeric_cols


def get_categorical_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Получение списка категориальных колонок
    
    Args:
        df: DataFrame
        exclude: Список колонок для исключения
        
    Returns:
        Список категориальных колонок
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if exclude:
        categorical_cols = [col for col in categorical_cols if col not in exclude]
    
    return categorical_cols
