"""
Модуль для генерации и загрузки данных о продажах продуктов питания.

Author: g1dcs
Date: 2026-03-30
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
import os


# Категории продуктов с сезонностью
PRODUCT_CATEGORIES = {
    'фрукты': {
        'products': ['яблоки', 'груши', 'бананы', 'апельсины', 'виноград', 'киви', 'манго', 'ананасы'],
        'seasonality': {
            'зима': ['апельсины', 'мандарины', 'бананы', 'киви'],
            'весна': ['яблоки', 'груши', 'бананы'],
            'лето': ['виноград', 'арбузы', 'дыни', 'черешня', 'персики'],
            'осень': ['яблоки', 'груши', 'виноград', 'гранаты']
        },
        'base_price_range': (50, 300)
    },
    'овощи': {
        'products': ['помидоры', 'огурцы', 'картофель', 'морковь', 'лук', 'капуста', 'перец', 'баклажаны'],
        'seasonality': {
            'зима': ['картофель', 'морковь', 'лук', 'капуста', 'свекла'],
            'весна': ['огурцы', 'помидоры', 'перец', 'кабачки'],
            'лето': ['помидоры', 'огурцы', 'перец', 'баклажаны', 'кабачки'],
            'осень': ['картофель', 'морковь', 'капуста', 'тыква', 'свекла']
        },
        'base_price_range': (30, 200)
    },
    'молочные продукты': {
        'products': ['молоко', 'йогурт', 'сыр', 'творог', 'сливки', 'кефир', 'ряженка'],
        'seasonality': {
            'зима': ['молоко', 'кефир', 'творог', 'сыр'],
            'весна': ['молоко', 'йогурт', 'творог'],
            'лето': ['мороженое', 'йогурт', 'сливки'],
            'осень': ['молоко', 'кефир', 'сыр']
        },
        'base_price_range': (60, 500)
    },
    'мясо': {
        'products': ['курица', 'свинина', 'говядина', 'индейка', 'баранина'],
        'seasonality': {
            'зима': ['свинина', 'говядина', 'баранина'],
            'весна': ['курица', 'индейка', 'свинина'],
            'лето': ['курица', 'индейка', 'свинина'],
            'осень': ['говядина', 'свинина', 'курица']
        },
        'base_price_range': (200, 800)
    },
    'напитки': {
        'products': ['вода', 'сок', 'газировка', 'чай', 'кофе', 'компот'],
        'seasonality': {
            'зима': ['чай', 'кофе', 'компот', 'вода'],
            'весна': ['вода', 'сок', 'чай'],
            'лето': ['вода', 'газировка', 'сок', 'лимонад', 'морс'],
            'осень': ['чай', 'кофе', 'сок', 'компот']
        },
        'base_price_range': (40, 400)
    },
    'выпечка': {
        'products': ['хлеб', 'булочки', 'пироги', 'печенье', 'торты'],
        'seasonality': {
            'зима': ['хлеб', 'пироги', 'торты', 'пряники'],
            'весна': ['хлеб', 'булочки', 'пироги'],
            'лето': ['хлеб', 'булочки', 'пирожные'],
            'осень': ['хлеб', 'пироги', 'торты', 'пряники']
        },
        'base_price_range': (30, 600)
    }
}


def get_season(date: datetime) -> str:
    """Определяет сезон по дате.

    Args:
        date (datetime): Дата

    Returns:
        str: Сезон (зима, весна, лето, осень)
    """
    month = date.month
    if month in [12, 1, 2]:
        return 'зима'
    elif month in [3, 4, 5]:
        return 'весна'
    elif month in [6, 7, 8]:
        return 'лето'
    else:
        return 'осень'


def get_holiday_multiplier(date: datetime) -> float:
    """Возвращает множитель продаж для праздничных дат.

    Args:
        date (datetime): Дата

    Returns:
        float: Множитель продаж
    """
    month, day = date.month, date.day

    # Новый год
    if month == 12 and day >= 25:
        return 2.0
    # Масленица (февраль)
    elif month == 2 and 15 <= day <= 25:
        return 1.5
    # 8 марта
    elif month == 3 and 5 <= day <= 10:
        return 1.6
    # Пасха (апрель)
    elif month == 4 and 15 <= day <= 25:
        return 1.5
    # 1 мая
    elif month == 4 and day >= 28:
        return 1.4
    # 9 мая
    elif month == 5 and 7 <= day <= 12:
        return 1.4
    # День знаний (сентябрь)
    elif month == 9 and 1 <= day <= 5:
        return 1.3
    # Новогодние предзаказы (ноябрь)
    elif month == 11 and day >= 20:
        return 1.4

    return 1.0


def generate_sales_data(
    start_date: str = '2023-01-01',
    end_date: str = '2025-12-31',
    n_stores: int = 5,
    random_seed: int = 42
) -> pd.DataFrame:
    """Генерирует данные о продажах продуктов питания.

    Args:
        start_date (str): Начальная дата в формате 'YYYY-MM-DD'
        end_date (str): Конечная дата в формате 'YYYY-MM-DD'
        n_stores (int): Количество магазинов
        random_seed (int): Seed для воспроизводимости

    Returns:
        pd.DataFrame: DataFrame с данными о продажах
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Создаем даты
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    sales_data = []

    # Названия магазинов
    store_names = [f'Магазин_{i+1}' for i in range(n_stores)]
    store_regions = ['Москва', 'СПб', 'Казань', 'Новосибирск', 'Екатеринбург'][:n_stores]

    for date in date_range:
        season = get_season(date)

        # Добавляем выходные эффект (больше продаж)
        is_weekend = date.weekday() >= 5
        weekend_multiplier = 1.3 if is_weekend else 1.0

        # Добавляем праздничные эффекты
        holiday_multiplier = get_holiday_multiplier(date)

        for store_idx, store_name in enumerate(store_names):
            for category, info in PRODUCT_CATEGORIES.items():
                for product in info['products']:
                    # Базовая цена
                    base_price = random.randint(*info['base_price_range'])

                    # Сезонный множитель
                    seasonal_products = info['seasonality'].get(season, [])
                    if product in seasonal_products:
                        season_multiplier = np.random.uniform(1.2, 1.8)
                    else:
                        season_multiplier = np.random.uniform(0.6, 1.0)

                    # Случайные колебания
                    random_factor = np.random.uniform(0.8, 1.2)

                    # Итоговое количество продаж
                    base_quantity = np.random.poisson(50)
                    quantity = int(base_quantity * season_multiplier * weekend_multiplier * 
                                  holiday_multiplier * random_factor)

                    # Цена с небольшими колебаниями
                    price = base_price * np.random.uniform(0.9, 1.1)

                    sales_data.append({
                        'date': date,
                        'store': store_name,
                        'region': store_regions[store_idx],
                        'category': category,
                        'product': product,
                        'quantity': quantity,
                        'price': round(price, 2),
                        'revenue': round(quantity * price, 2),
                        'season': season
                    })

    return pd.DataFrame(sales_data)


def load_or_generate_data(filepath: str = None, **kwargs) -> pd.DataFrame:
    """Загружает данные из файла или генерирует новые.

    Args:
        filepath (str): Путь к CSV файлу
        **kwargs: Параметры для generate_sales_data()

    Returns:
        pd.DataFrame: DataFrame с данными о продажах
    """
    if filepath and os.path.exists(filepath):
        print(f"📂 Загружаем данные из {filepath}")
        return pd.read_csv(filepath, parse_dates=['date'])
    else:
        print("🎲 Генерируем синтетические данные о продажах...")
        df = generate_sales_data(**kwargs)
        if filepath:
            df.to_csv(filepath, index=False)
            print(f"💾 Данные сохранены в {filepath}")
        return df


if __name__ == '__main__':
    # Генерация тестовых данных
    df = generate_sales_data(n_stores=3)
    print(f"✅ Сгенерировано {len(df):,} записей")
    print(f"📊 Период: {df['date'].min().date()} - {df['date'].max().date()}")
    print(f"🏪 Магазинов: {df['store'].nunique()}")
    print(f"📦 Продуктов: {df['product'].nunique()}")
    print(f"💰 Общая выручка: {df['revenue'].sum():,.0f} руб.")
