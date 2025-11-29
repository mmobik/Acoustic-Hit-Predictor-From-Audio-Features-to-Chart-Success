"""
Models Module
Обертки и утилиты для обучения и оценки ML моделей
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Вычисление метрик регрессии
    
    Args:
        y_true: Реальные значения
        y_pred: Предсказанные значения
        
    Returns:
        Словарь с метриками
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE может быть undefined если есть нули в y_true
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def train_linear_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Обучение линейных моделей
    
    Args:
        X_train, X_test: Признаки
        y_train, y_test: Целевая переменная
        
    Returns:
        Словарь с результатами
    """
    results = {}
    
    # Linear Regression
    print("Обучение Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['Linear Regression'] = {
        'model': lr,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }
    
    # Ridge Regression
    print("Обучение Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    results['Ridge'] = {
        'model': ridge,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }
    
    # Lasso Regression
    print("Обучение Lasso Regression...")
    lasso = Lasso(alpha=0.1, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    results['Lasso'] = {
        'model': lasso,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }
    
    return results


def train_tree_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                     y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Обучение древовидных моделей
    
    Args:
        X_train, X_test: Признаки
        y_train, y_test: Целевая переменная
        
    Returns:
        Словарь с результатами
    """
    results = {}
    
    # Decision Tree
    print("Обучение Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    results['Decision Tree'] = {
        'model': dt,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }
    
    # Random Forest
    print("Обучение Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'model': rf,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred),
        'feature_importances': pd.Series(rf.feature_importances_, index=X_train.columns)
    }
    
    # Extra Trees
    print("Обучение Extra Trees...")
    et = ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    et.fit(X_train, y_train)
    y_pred = et.predict(X_test)
    results['Extra Trees'] = {
        'model': et,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred),
        'feature_importances': pd.Series(et.feature_importances_, index=X_train.columns)
    }
    
    return results


def train_boosting_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Обучение моделей градиентного бустинга
    
    Args:
        X_train, X_test: Признаки
        y_train, y_test: Целевая переменная
        
    Returns:
        Словарь с результатами
    """
    results = {}
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("Обучение XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        results['XGBoost'] = {
            'model': xgb_model,
            'predictions': y_pred,
            'metrics': calculate_metrics(y_test, y_pred),
            'feature_importances': pd.Series(xgb_model.feature_importances_, index=X_train.columns)
        }
    else:
        print("XGBoost не установлен, пропускаем...")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("Обучение LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        y_pred = lgb_model.predict(X_test)
        results['LightGBM'] = {
            'model': lgb_model,
            'predictions': y_pred,
            'metrics': calculate_metrics(y_test, y_pred),
            'feature_importances': pd.Series(lgb_model.feature_importances_, index=X_train.columns)
        }
    else:
        print("LightGBM не установлен, пропускаем...")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        print("Обучение CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        y_pred = cat_model.predict(X_test)
        results['CatBoost'] = {
            'model': cat_model,
            'predictions': y_pred,
            'metrics': calculate_metrics(y_test, y_pred),
            'feature_importances': pd.Series(cat_model.feature_importances_, index=X_train.columns)
        }
    else:
        print("CatBoost не установлен, пропускаем...")
    
    return results


def train_neural_network(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Обучение нейронной сети
    
    Args:
        X_train, X_test: Признаки
        y_train, y_test: Целевая переменная
        
    Returns:
        Словарь с результатами
    """
    results = {}
    
    print("Обучение MLP Regressor...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    results['MLP Regressor'] = {
        'model': mlp,
        'predictions': y_pred,
        'metrics': calculate_metrics(y_test, y_pred)
    }
    
    return results


def train_all_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series) -> Dict:
    """
    Обучение всех моделей
    
    Args:
        X_train, X_test: Признаки
        y_train, y_test: Целевая переменная
        
    Returns:
        Словарь со всеми результатами
    """
    all_results = {}
    
    print("\n" + "="*60)
    print("ЛИНЕЙНЫЕ МОДЕЛИ")
    print("="*60)
    all_results.update(train_linear_models(X_train, X_test, y_train, y_test))
    
    print("\n" + "="*60)
    print("ДРЕВОВИДНЫЕ МОДЕЛИ")
    print("="*60)
    all_results.update(train_tree_models(X_train, X_test, y_train, y_test))
    
    print("\n" + "="*60)
    print("ГРАДИЕНТНЫЙ БУСТИНГ")
    print("="*60)
    all_results.update(train_boosting_models(X_train, X_test, y_train, y_test))
    
    print("\n" + "="*60)
    print("НЕЙРОННЫЕ СЕТИ")
    print("="*60)
    all_results.update(train_neural_network(X_train, X_test, y_train, y_test))
    
    return all_results


def create_results_table(results: Dict) -> pd.DataFrame:
    """
    Создание таблицы с результатами моделей
    
    Args:
        results: Словарь с результатами
        
    Returns:
        DataFrame с метриками
    """
    rows = []
    for model_name, result in results.items():
        row = {'Model': model_name}
        row.update(result['metrics'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('MAE')  # Сортируем по MAE
    
    return df


def perform_cross_validation(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
    """
    Кросс-валидация модели
    
    Args:
        model: Модель для валидации
        X: Признаки
        y: Целевая переменная
        cv: Количество фолдов
        
    Returns:
        Словарь с результатами CV
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # MAE
    mae_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    
    # RMSE
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error'))
    
    # R2
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    results = {
        'MAE_mean': mae_scores.mean(),
        'MAE_std': mae_scores.std(),
        'RMSE_mean': rmse_scores.mean(),
        'RMSE_std': rmse_scores.std(),
        'R2_mean': r2_scores.mean(),
        'R2_std': r2_scores.std()
    }
    
    print(f"\nРезультаты {cv}-Fold Cross-Validation:")
    print(f"{'='*50}")
    print(f"MAE:  {results['MAE_mean']:.4f} ± {results['MAE_std']:.4f}")
    print(f"RMSE: {results['RMSE_mean']:.4f} ± {results['RMSE_std']:.4f}")
    print(f"R²:   {results['R2_mean']:.4f} ± {results['R2_std']:.4f}")
    
    return results


def get_feature_importance(model, feature_names: List[str], top_n: int = 20) -> pd.Series:
    """
    Получение важности признаков
    
    Args:
        model: Обученная модель
        feature_names: Названия признаков
        top_n: Количество топ признаков
        
    Returns:
        Series с важностью признаков
    """
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, 'coef_'):
        importances = pd.Series(np.abs(model.coef_), index=feature_names)
    else:
        raise ValueError("Модель не поддерживает извлечение важности признаков")
    
    return importances.sort_values(ascending=False).head(top_n)
