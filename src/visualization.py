"""
Visualization Module
Функции для визуализации данных и результатов моделей
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional


# Настройка стиля по умолчанию
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_distribution(df: pd.DataFrame, column: str, figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Визуализация распределения признака
    
    Args:
        df: DataFrame
        column: Название колонки
        figsize: Размер фигуры
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Гистограмма
    axes[0].hist(df[column].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Распределение {column}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Частота')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df[column].dropna(), vert=True)
    axes[1].set_title(f'Box Plot: {column}', fontsize=14, fontweight='bold')
    axes[1].set_ylabel(column)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Статистика
    print(f"\n{'='*50}")
    print(f"Статистика для {column}:")
    print(f"{'='*50}")
    print(df[column].describe())
    print(f"\nКоличество пропусков: {df[column].isnull().sum()}")
    print(f"Процент пропусков: {df[column].isnull().sum() / len(df) * 100:.2f}%")


def plot_multiple_distributions(df: pd.DataFrame, columns: List[str], ncols: int = 3, figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Визуализация распределений нескольких признаков
    
    Args:
        df: DataFrame
        columns: Список колонок
        ncols: Количество колонок в сетке
        figsize: Размер фигуры
    """
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes
    
    for idx, col in enumerate(columns):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(col, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Частота')
        axes[idx].grid(True, alpha=0.3)
    
    # Скрыть лишние оси
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (14, 12), annot: bool = False) -> None:
    """
    Визуализация матрицы корреляций
    
    Args:
        df: DataFrame с числовыми признаками
        figsize: Размер фигуры
        annot: Показывать ли значения корреляций
    """
    # Выбираем только числовые колонки
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Вычисляем корреляции
    corr_matrix = numeric_df.corr()
    
    # Создаем маску для верхнего треугольника
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Визуализация
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Матрица корреляций признаков', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_target_correlation(df: pd.DataFrame, target_col: str, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Визуализация корреляций признаков с целевой переменной
    
    Args:
        df: DataFrame
        target_col: Название целевой колонки
        top_n: Количество топ признаков для отображения
        figsize: Размер фигуры
    """
    # Выбираем только числовые колонки
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Вычисляем корреляции с таргетом
    correlations = numeric_df.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    
    # Берем топ-N
    top_correlations = correlations.head(top_n)
    
    # Визуализация
    plt.figure(figsize=figsize)
    colors = ['green' if x > 0 else 'red' for x in top_correlations.values]
    plt.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7, edgecolor='black')
    plt.yticks(range(len(top_correlations)), top_correlations.index)
    plt.xlabel('Корреляция с таргетом', fontsize=12)
    plt.title(f'Топ-{top_n} признаков по корреляции с {target_col}', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    # Вывод значений
    print(f"\n{'='*60}")
    print(f"Топ-{top_n} признаков по корреляции с {target_col}:")
    print(f"{'='*60}")
    for feature, corr in top_correlations.items():
        print(f"{feature:40s}: {corr:+.4f}")


def plot_scatter_with_target(df: pd.DataFrame, x_col: str, y_col: str, target_col: str, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Scatter plot двух признаков с цветовой кодировкой по таргету
    
    Args:
        df: DataFrame
        x_col: Колонка для оси X
        y_col: Колонка для оси Y
        target_col: Целевая переменная для цвета
        figsize: Размер фигуры
    """
    plt.figure(figsize=figsize)
    scatter = plt.scatter(df[x_col], df[y_col], c=df[target_col], 
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label=target_col)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f'{y_col} vs {x_col} (цвет: {target_col})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pairplot(df: pd.DataFrame, columns: List[str], target_col: Optional[str] = None, figsize: Tuple[int, int] = (12, 12)) -> None:
    """
    Pairplot для выбранных признаков
    
    Args:
        df: DataFrame
        columns: Список колонок для визуализации
        target_col: Целевая переменная для цветовой кодировки
        figsize: Размер фигуры
    """
    if target_col and target_col not in columns:
        columns = columns + [target_col]
    
    sns.pairplot(df[columns], hue=target_col if target_col else None, 
                 diag_kind='hist', plot_kws={'alpha': 0.6})
    plt.suptitle('Pairplot признаков', y=1.01, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances: pd.Series, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Визуализация важности признаков
    
    Args:
        importances: Series с важностью признаков
        top_n: Количество топ признаков
        figsize: Размер фигуры
    """
    # Сортируем и берем топ-N
    top_importances = importances.sort_values(ascending=False).head(top_n)
    
    # Визуализация
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_importances)), top_importances.values, 
             color='steelblue', alpha=0.7, edgecolor='black')
    plt.yticks(range(len(top_importances)), top_importances.index)
    plt.xlabel('Важность признака', fontsize=12)
    plt.title(f'Топ-{top_n} признаков по важности', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, metric: str = 'MAE', figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Сравнение моделей по метрикам
    
    Args:
        results_df: DataFrame с результатами моделей
        metric: Метрика для визуализации
        figsize: Размер фигуры
    """
    plt.figure(figsize=figsize)
    
    models = results_df['Model']
    values = results_df[metric]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = plt.bar(range(len(models)), values, color=colors, alpha=0.7, edgecolor='black')
    
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel(metric, fontsize=12)
    plt.title(f'Сравнение моделей по метрике {metric}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values), 
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Визуализация предсказаний vs реальных значений
    
    Args:
        y_true: Реальные значения
        y_pred: Предсказанные значения
        figsize: Размер фигуры
    """
    plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
    
    # Идеальная линия
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальные предсказания')
    
    plt.xlabel('Реальные значения', fontsize=12)
    plt.ylabel('Предсказанные значения', fontsize=12)
    plt.title('Предсказания vs Реальные значения', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, figsize: Tuple[int, int] = (12, 5)) -> None:
    """
    Визуализация остатков (residuals)
    
    Args:
        y_true: Реальные значения
        y_pred: Предсказанные значения
        figsize: Размер фигуры
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Предсказанные значения', fontsize=12)
    axes[0].set_ylabel('Остатки', fontsize=12)
    axes[0].set_title('Остатки vs Предсказания', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Остатки', fontsize=12)
    axes[1].set_ylabel('Частота', fontsize=12)
    axes[1].set_title('Распределение остатков', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Статистика остатков
    print(f"\n{'='*50}")
    print("Статистика остатков:")
    print(f"{'='*50}")
    print(f"Среднее: {residuals.mean():.4f}")
    print(f"Медиана: {np.median(residuals):.4f}")
    print(f"Стд. отклонение: {residuals.std():.4f}")
    print(f"Мин: {residuals.min():.4f}")
    print(f"Макс: {residuals.max():.4f}")
