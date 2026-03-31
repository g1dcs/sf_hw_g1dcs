"""
Модуль для ML-прогнозирования продаж продуктов питания.

Author: g1dcs
Date: 2026-03-30
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class SalesPredictor:
    """Класс для прогнозирования продаж с использованием ML."""

    def __init__(self, df: pd.DataFrame):
        """Инициализация предиктора.

        Args:
            df (pd.DataFrame): DataFrame с данными о продажах
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создает признаки для модели.

        Args:
            df (pd.DataFrame): Исходный DataFrame

        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        df_features = df.copy()

        # Временные признаки
        df_features['month'] = df_features['date'].dt.month
        df_features['day_of_week'] = df_features['date'].dt.dayofweek
        df_features['day_of_year'] = df_features['date'].dt.dayofyear
        df_features['quarter'] = df_features['date'].dt.quarter
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)

        # Признаки сезона
        season_map = {'весна': 0, 'лето': 1, 'осень': 2, 'зима': 3}
        df_features['season_num'] = df_features['season'].map(season_map)

        # Кодирование категориальных признаков
        categorical_cols = ['category', 'product', 'store', 'region']

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_features[col])
            else:
                # Для новых данных используем уже обученный encoder
                df_features[f'{col}_encoded'] = self.label_encoders[col].transform(df_features[col])

        # Лаговые признаки (продажи за предыдущие периоды)
        df_features = df_features.sort_values(['product', 'store', 'date'])
        df_features['revenue_lag_7'] = df_features.groupby(['product', 'store'])['revenue'].shift(7)
        df_features['revenue_lag_30'] = df_features.groupby(['product', 'store'])['revenue'].shift(30)
        df_features['quantity_lag_7'] = df_features.groupby(['product', 'store'])['quantity'].shift(7)

        # Скользящие средние
        df_features['revenue_ma_7'] = df_features.groupby(['product', 'store'])['revenue'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df_features['revenue_ma_30'] = df_features.groupby(['product', 'store'])['revenue'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )

        # Заполняем NaN
        df_features = df_features.fillna(0)

        return df_features

    def get_feature_columns(self) -> list:
        """Возвращает список признаков для модели.

        Returns:
            list: Список названий признаков
        """
        return [
            'month', 'day_of_week', 'day_of_year', 'quarter', 'is_weekend',
            'season_num', 'price',
            'category_encoded', 'product_encoded', 'store_encoded', 'region_encoded',
            'revenue_lag_7', 'revenue_lag_30', 'quantity_lag_7',
            'revenue_ma_7', 'revenue_ma_30'
        ]

    def train_models(self, target_col: str = 'revenue', test_size: float = 0.2) -> dict:
        """Обучает несколько ML моделей.

        Args:
            target_col (str): Целевая переменная
            test_size (float): Размер тестовой выборки

        Returns:
            dict: Результаты обучения
        """
        print("🔧 Подготовка признаков...")
        df_prepared = self.prepare_features(self.df)

        # Удаляем строки с пропусками в целевой переменной
        df_prepared = df_prepared.dropna(subset=[target_col])

        # Выбираем признаки
        feature_cols = self.get_feature_columns()
        X = df_prepared[feature_cols]
        y = df_prepared[target_col]

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        # Модели
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        results = {}
        best_score = -np.inf

        print("\n🤖 Обучение моделей...")
        print("-" * 60)

        for name, model in models.items():
            print(f"\n📊 {name}:")

            # Обучение
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Метрики
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # MAPE
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'predictions': y_pred
            }

            print(f"   MAE:  {mae:,.0f} руб.")
            print(f"   RMSE: {rmse:,.0f} руб.")
            print(f"   R²:   {r2:.4f}")
            print(f"   MAPE: {mape:.2f}%")

            # Выбираем лучшую модель по R²
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = name

        self.models = results

        print("\n" + "=" * 60)
        print(f"🏆 Лучшая модель: {self.best_model_name} (R² = {best_score:.4f})")
        print("=" * 60)

        return results

    def predict(self, df_future: pd.DataFrame) -> np.ndarray:
        """Делает прогноз для новых данных.

        Args:
            df_future (pd.DataFrame): DataFrame с будущими датами

        Returns:
            np.ndarray: Предсказанные значения
        """
        if self.best_model is None:
            raise ValueError("Сначала обучите модели с помощью train_models()")

        df_prepared = self.prepare_features(df_future)
        feature_cols = self.get_feature_columns()
        X_future = df_prepared[feature_cols]

        if self.best_model_name == 'Linear Regression':
            X_future_scaled = self.scalers['main'].transform(X_future)
            predictions = self.best_model.predict(X_future_scaled)
        else:
            predictions = self.best_model.predict(X_future)

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """Возвращает важность признаков.

        Returns:
            pd.DataFrame: DataFrame с важностью признаков
        """
        if self.best_model is None:
            raise ValueError("Сначала обучите модели")

        feature_cols = self.get_feature_columns()

        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_)
        else:
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def forecast_future_sales(self, periods: int = 90) -> pd.DataFrame:
        """Прогнозирует продажи на будущие периоды.

        Args:
            periods (int): Количество дней для прогноза

        Returns:
            pd.DataFrame: DataFrame с прогнозом
        """
        # Получаем последнюю дату
        last_date = self.df['date'].max()

        # Создаем будущие даты
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')

        # Создаем шаблон для каждого продукта и магазина
        products_stores = self.df[['product', 'store', 'category', 'region', 'price']].drop_duplicates()

        future_data = []
        for _, row in products_stores.iterrows():
            for date in future_dates:
                season = self._get_season(date)
                future_data.append({
                    'date': date,
                    'product': row['product'],
                    'store': row['store'],
                    'category': row['category'],
                    'region': row['region'],
                    'price': row['price'],
                    'season': season,
                    'quantity': 0,  # placeholder
                    'revenue': 0    # placeholder
                })

        df_future = pd.DataFrame(future_data)

        # Делаем прогноз
        predictions = self.predict(df_future)

        df_future['predicted_revenue'] = predictions
        df_future['predicted_quantity'] = (predictions / df_future['price']).round().astype(int)

        return df_future

    @staticmethod
    def _get_season(date: pd.Timestamp) -> str:
        """Определяет сезон по дате."""
        month = date.month
        if month in [12, 1, 2]:
            return 'зима'
        elif month in [3, 4, 5]:
            return 'весна'
        elif month in [6, 7, 8]:
            return 'лето'
        else:
            return 'осень'


if __name__ == '__main__':
    print("📚 Модуль SalesPredictor загружен")
    print("💡 Использование:")
    print("   predictor = SalesPredictor(df)")
    print("   predictor.train_models()")
    print("   forecast = predictor.forecast_future_sales(periods=30)")
