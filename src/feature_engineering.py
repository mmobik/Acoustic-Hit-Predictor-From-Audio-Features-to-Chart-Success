"""
Feature Engineering Module
Создание производных признаков для датасета Spotify
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def create_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков на основе длительности трека
    
    Args:
        df: DataFrame с колонкой duration_ms
        
    Returns:
        DataFrame с новыми признаками
    """
    df_new = df.copy()
    
    # Длительность в минутах
    df_new['duration_min'] = df_new['duration_ms'] / 60000
    
    # Длительность в секундах
    df_new['duration_sec'] = df_new['duration_ms'] / 1000
    
    # Категоризация длительности
    df_new['duration_category'] = pd.cut(
        df_new['duration_min'],
        bins=[0, 2, 3, 4, 5, np.inf],
        labels=['very_short', 'short', 'medium', 'long', 'very_long']
    )
    
    return df_new


def create_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков на основе энергии и связанных параметров
    
    Args:
        df: DataFrame с акустическими признаками
        
    Returns:
        DataFrame с новыми признаками
    """
    df_new = df.copy()
    
    # Соотношение энергии и танцевальности
    df_new['energy_dance_ratio'] = df_new['energy'] / (df_new['danceability'] + 1e-6)
    
    # Произведение энергии и темпа
    df_new['tempo_energy_product'] = df_new['tempo'] * df_new['energy']
    
    # Баланс энергии и акустичности
    df_new['acoustic_energy_balance'] = df_new['acousticness'] * (1 - df_new['energy'])
    
    # Энергетический индекс (комбинация энергии и громкости)
    df_new['energy_loudness_index'] = df_new['energy'] * (df_new['loudness'] + 60) / 60
    
    return df_new


def create_mood_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков настроения
    
    Args:
        df: DataFrame с признаками valence и energy
        
    Returns:
        DataFrame с новыми признаками
    """
    df_new = df.copy()
    
    # Взаимодействие позитивности и энергии
    df_new['valence_energy_interaction'] = df_new['valence'] * df_new['energy']
    
    # Категоризация настроения
    def categorize_mood(row):
        if row['valence'] > 0.5 and row['energy'] > 0.5:
            return 'happy_energetic'
        elif row['valence'] > 0.5 and row['energy'] <= 0.5:
            return 'happy_calm'
        elif row['valence'] <= 0.5 and row['energy'] > 0.5:
            return 'sad_energetic'
        else:
            return 'sad_calm'
    
    df_new['mood_category'] = df_new.apply(categorize_mood, axis=1)
    
    # Индекс позитивности (комбинация valence и danceability)
    df_new['positivity_index'] = (df_new['valence'] + df_new['danceability']) / 2
    
    return df_new


def create_acoustic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков на основе акустических характеристик
    
    Args:
        df: DataFrame с акустическими признаками
        
    Returns:
        DataFrame с новыми признаками
    """
    df_new = df.copy()
    
    # Индекс инструментальности
    df_new['instrumental_acoustic_index'] = df_new['instrumentalness'] * df_new['acousticness']
    
    # Баланс речи и музыки
    df_new['speech_music_balance'] = df_new['speechiness'] / (1 - df_new['speechiness'] + 1e-6)
    
    # Индекс "живого" звучания
    df_new['live_acoustic_index'] = df_new['liveness'] * df_new['acousticness']
    
    return df_new


def create_tempo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков на основе темпа
    
    Args:
        df: DataFrame с колонкой tempo
        
    Returns:
        DataFrame с новыми признаками
    """
    df_new = df.copy()
    
    # Категоризация темпа
    df_new['tempo_category'] = pd.cut(
        df_new['tempo'],
        bins=[0, 90, 120, 140, 180, np.inf],
        labels=['slow', 'moderate', 'upbeat', 'fast', 'very_fast']
    )
    
    # Нормализованный темп
    df_new['tempo_normalized'] = (df_new['tempo'] - df_new['tempo'].min()) / \
                                  (df_new['tempo'].max() - df_new['tempo'].min())
    
    # Взаимодействие темпа и танцевальности
    df_new['tempo_dance_interaction'] = df_new['tempo'] * df_new['danceability']
    
    return df_new


def create_loudness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков на основе громкости
    
    Args:
        df: DataFrame с колонкой loudness
        
    Returns:
        DataFrame с новыми признаками
    """
    df_new = df.copy()
    
    # Нормализованная громкость (от 0 до 1)
    df_new['loudness_normalized'] = (df_new['loudness'] + 60) / 60
    
    # Категоризация громкости
    df_new['loudness_category'] = pd.cut(
        df_new['loudness'],
        bins=[-np.inf, -20, -10, -5, 0],
        labels=['quiet', 'moderate', 'loud', 'very_loud']
    )
    
    # Взаимодействие громкости и энергии
    df_new['loudness_energy_interaction'] = df_new['loudness_normalized'] * df_new['energy']
    
    return df_new


def create_polynomial_features(df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
    """
    Создание полиномиальных признаков
    
    Args:
        df: Исходный DataFrame
        columns: Список колонок для полиномиальных признаков
        degree: Степень полинома
        
    Returns:
        DataFrame с полиномиальными признаками
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[columns])
    
    # Получение названий признаков
    feature_names = poly.get_feature_names_out(columns)
    
    # Создание DataFrame
    df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Удаление исходных признаков (они уже есть в df)
    df_poly = df_poly.drop(columns=columns)
    
    return pd.concat([df, df_poly], axis=1)


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание всех производных признаков
    
    Args:
        df: Исходный DataFrame
        
    Returns:
        DataFrame со всеми новыми признаками
    """
    df_new = df.copy()
    
    # Проверка наличия необходимых колонок
    required_cols = ['duration_ms', 'energy', 'danceability', 'acousticness', 
                     'tempo', 'valence', 'loudness', 'instrumentalness', 
                     'speechiness', 'liveness']
    
    missing_cols = [col for col in required_cols if col not in df_new.columns]
    if missing_cols:
        print(f"Предупреждение: отсутствуют колонки {missing_cols}")
        return df_new
    
    # Создание признаков
    if 'duration_ms' in df_new.columns:
        df_new = create_duration_features(df_new)
    
    if all(col in df_new.columns for col in ['energy', 'danceability', 'acousticness', 'tempo', 'loudness']):
        df_new = create_energy_features(df_new)
    
    if all(col in df_new.columns for col in ['valence', 'energy', 'danceability']):
        df_new = create_mood_features(df_new)
    
    if all(col in df_new.columns for col in ['instrumentalness', 'acousticness', 'speechiness', 'liveness']):
        df_new = create_acoustic_features(df_new)
    
    if 'tempo' in df_new.columns:
        df_new = create_tempo_features(df_new)
    
    if 'loudness' in df_new.columns:
        df_new = create_loudness_features(df_new)
    
    print(f"Создано {len(df_new.columns) - len(df.columns)} новых признаков")
    print(f"Итого признаков: {len(df_new.columns)}")
    
    return df_new


def get_feature_correlations(df: pd.DataFrame, target_col: str, threshold: float = 0.1) -> pd.DataFrame:
    """
    Получение корреляций признаков с целевой переменной
    
    Args:
        df: DataFrame с признаками
        target_col: Название целевой колонки
        threshold: Минимальный порог корреляции для отображения
        
    Returns:
        DataFrame с корреляциями, отсортированный по убыванию
    """
    # Выбираем только числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col not in numeric_cols:
        raise ValueError(f"{target_col} не является числовой колонкой")
    
    # Вычисляем корреляции
    correlations = df[numeric_cols].corr()[target_col].drop(target_col)
    
    # Фильтруем по порогу
    correlations = correlations[abs(correlations) >= threshold]
    
    # Сортируем по абсолютному значению
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    
    # Создаем DataFrame
    corr_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values,
        'Abs_Correlation': abs(correlations.values)
    })
    
    return corr_df
