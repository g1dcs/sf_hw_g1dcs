"""
Модуль для анализа сезонных продаж продуктов питания.

Author: g1dcs
Date: 2026-03-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


class SeasonalSalesAnalyzer:
    """Класс для анализа сезонных продаж продуктов питания."""

    def __init__(self, df: pd.DataFrame):
        """Инициализация анализатора.

        Args:
            df (pd.DataFrame): DataFrame с данными о продажах
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['month'] = self.df['date'].dt.month
        self.df['year'] = self.df['date'].dt.year
        self.df['month_name'] = self.df['date'].dt.month_name()

    def get_basic_info(self) -> Dict:
        """Возвращает базовую информацию о данных.

        Returns:
            Dict: Словарь с базовой информацией
        """
        info = {
            'total_records': len(self.df),
            'date_range': (self.df['date'].min(), self.df['date'].max()),
            'total_revenue': self.df['revenue'].sum(),
            'total_quantity': self.df['quantity'].sum(),
            'n_stores': self.df['store'].nunique(),
            'n_products': self.df['product'].nunique(),
            'n_categories': self.df['category'].nunique()
        }
        return info

    def analyze_by_season(self) -> pd.DataFrame:
        """Анализирует продажи по сезонам.

        Returns:
            pd.DataFrame: Агрегированные данные по сезонам
        """
        season_analysis = self.df.groupby('season').agg({
            'revenue': ['sum', 'mean'],
            'quantity': ['sum', 'mean'],
            'product': 'nunique'
        }).round(2)

        season_analysis.columns = ['_'.join(col).strip() for col in season_analysis.columns]
        season_analysis = season_analysis.reset_index()

        # Добавляем процент от общей выручки
        total_revenue = self.df['revenue'].sum()
        season_analysis['revenue_pct'] = (season_analysis['revenue_sum'] / total_revenue * 100).round(2)

        # Сортируем по сезонам
        season_order = ['весна', 'лето', 'осень', 'зима']
        season_analysis['season'] = pd.Categorical(season_analysis['season'], categories=season_order, ordered=True)
        season_analysis = season_analysis.sort_values('season')

        return season_analysis

    def get_top_products_by_season(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """Возвращает топ продуктов по каждому сезону.

        Args:
            top_n (int): Количество топ продуктов

        Returns:
            Dict[str, pd.DataFrame]: Словарь с DataFrame для каждого сезона
        """
        result = {}
        seasons = ['весна', 'лето', 'осень', 'зима']

        for season in seasons:
            season_df = self.df[self.df['season'] == season]

            top_products = season_df.groupby(['category', 'product']).agg({
                'revenue': 'sum',
                'quantity': 'sum'
            }).reset_index()

            top_products = top_products.sort_values('revenue', ascending=False).head(top_n)
            top_products['rank'] = range(1, len(top_products) + 1)

            result[season] = top_products

        return result

    def get_most_profitable_products(self, top_n: int = 20) -> pd.DataFrame:
        """Возвращает самые прибыльные продукты за весь период.

        Args:
            top_n (int): Количество продуктов

        Returns:
            pd.DataFrame: Топ прибыльных продуктов
        """
        profitable = self.df.groupby(['category', 'product']).agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'date': 'count'
        }).reset_index()

        profitable.columns = ['category', 'product', 'total_revenue', 'total_quantity', 'sales_days']
        profitable = profitable.sort_values('total_revenue', ascending=False).head(top_n)
        profitable['rank'] = range(1, len(profitable) + 1)

        return profitable

    def analyze_category_seasonality(self) -> pd.DataFrame:
        """Анализирует сезонность по категориям продуктов.

        Returns:
            pd.DataFrame: Сезонный анализ по категориям
        """
        category_season = self.df.groupby(['category', 'season']).agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()

        # Добавляем процент внутри категории
        category_totals = self.df.groupby('category')['revenue'].sum().to_dict()
        category_season['revenue_pct_in_category'] = category_season.apply(
            lambda row: (row['revenue'] / category_totals[row['category']] * 100), axis=1
        ).round(2)

        return category_season

    def get_seasonal_recommendations(self) -> Dict[str, List[str]]:
        """Генерирует рекомендации по ассортименту для каждого сезона.

        Returns:
            Dict[str, List[str]]: Рекомендации по сезонам
        """
        recommendations = {}

        top_by_season = self.get_top_products_by_season(top_n=5)

        for season, df in top_by_season.items():
            products = df['product'].tolist()
            recommendations[season] = products

        return recommendations

    def plot_seasonal_revenue(self, save_path: str = None) -> None:
        """Строит график выручки по сезонам.

        Args:
            save_path (str, optional): Путь для сохранения
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # График 1: Общая выручка по сезонам
        season_data = self.analyze_by_season()

        ax1 = axes[0]
        bars = ax1.bar(season_data['season'], season_data['revenue_sum'], 
                       color=['#90EE90', '#FFD700', '#FF8C00', '#87CEEB'],
                       edgecolor='black', linewidth=1.5)
        ax1.set_title('Выручка по сезонам', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Сезон')
        ax1.set_ylabel('Выручка (руб.)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/1e6:.1f}M',
                    ha='center', va='bottom', fontweight='bold')

        # График 2: Процент от общей выручки
        ax2 = axes[1]
        colors = ['#90EE90', '#FFD700', '#FF8C00', '#87CEEB']
        wedges, texts, autotexts = ax2.pie(season_data['revenue_pct'], 
                                            labels=season_data['season'],
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90,
                                            explode=[0.02]*4)
        ax2.set_title('Распределение выручки по сезонам', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")

        plt.show()

    def plot_category_seasonality(self, save_path: str = None) -> None:
        """Строит тепловую карту сезонности по категориям.

        Args:
            save_path (str, optional): Путь для сохранения
        """
        category_season = self.analyze_category_seasonality()

        # Создаем сводную таблицу
        pivot = category_season.pivot(index='category', columns='season', values='revenue_pct_in_category')

        # Сортируем сезоны
        season_order = ['весна', 'лето', 'осень', 'зима']
        pivot = pivot.reindex(columns=season_order)

        # Строим тепловую карту
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': '% выручки в категории'},
                   linewidths=0.5, ax=ax)

        ax.set_title('Сезонность продаж по категориям (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Сезон')
        ax.set_ylabel('Категория')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")

        plt.show()

    def plot_monthly_trend(self, save_path: str = None) -> None:
        """Строит график помесячной динамики продаж.

        Args:
            save_path (str, optional): Путь для сохранения
        """
        monthly = self.df.groupby(['year', 'month']).agg({
            'revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()

        monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # График выручки
        ax1 = axes[0]
        ax1.plot(monthly['date'], monthly['revenue'], marker='o', linewidth=2, 
                markersize=6, color='#2E86AB')
        ax1.fill_between(monthly['date'], monthly['revenue'], alpha=0.3, color='#2E86AB')
        ax1.set_title('Динамика выручки по месяцам', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Дата')
        ax1.set_ylabel('Выручка (руб.)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        ax1.grid(True, alpha=0.3)

        # График количества
        ax2 = axes[1]
        ax2.plot(monthly['date'], monthly['quantity'], marker='s', linewidth=2,
                markersize=6, color='#A23B72')
        ax2.fill_between(monthly['date'], monthly['quantity'], alpha=0.3, color='#A23B72')
        ax2.set_title('Динамика продаж по количеству', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Дата')
        ax2.set_ylabel('Количество')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")

        plt.show()

    def plot_top_products(self, season: str = None, top_n: int = 15, save_path: str = None) -> None:
        """Строит график топ продуктов.

        Args:
            season (str, optional): Сезон для фильтрации
            top_n (int): Количество продуктов
            save_path (str, optional): Путь для сохранения
        """
        if season:
            df_filtered = self.df[self.df['season'] == season]
            title_suffix = f' ({season})'
        else:
            df_filtered = self.df
            title_suffix = ' (все сезоны)'

        top_products = df_filtered.groupby('product')['revenue'].sum().sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.Spectral(np.linspace(0, 1, len(top_products)))
        bars = ax.barh(range(len(top_products)), top_products.values, color=colors, edgecolor='black')

        ax.set_yticks(range(len(top_products)))
        ax.set_yticklabels(top_products.index)
        ax.invert_yaxis()
        ax.set_xlabel('Выручка (руб.)')
        ax.set_title(f'Топ-{top_n} продуктов по выручке{title_suffix}', fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

        # Добавляем значения
        for i, (bar, value) in enumerate(zip(bars, top_products.values)):
            ax.text(value + value*0.01, i, f'{value/1e6:.2f}M', 
                   va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")

        plt.show()

    def generate_full_report(self) -> str:
        """Генерирует полный текстовый отчет.

        Returns:
            str: Текст отчета
        """
        info = self.get_basic_info()
        season_data = self.analyze_by_season()
        top_products = self.get_most_profitable_products(top_n=10)
        recommendations = self.get_seasonal_recommendations()

        report = []
        report.append("=" * 70)
        report.append("📊 ОТЧЕТ О СЕЗОННЫХ ПРОДАЖАХ ПРОДУКТОВ ПИТАНИЯ")
        report.append("=" * 70)
        report.append("")
        report.append(f"📅 Период анализа: {info['date_range'][0].date()} - {info['date_range'][1].date()}")
        report.append(f"🏪 Количество магазинов: {info['n_stores']}")
        report.append(f"📦 Уникальных продуктов: {info['n_products']}")
        report.append(f"📂 Категорий: {info['n_categories']}")
        report.append("")
        report.append(f"💰 ОБЩАЯ ВЫРУЧКА: {info['total_revenue']:,.0f} руб.")
        report.append(f"📦 ОБЩЕЕ КОЛИЧЕСТВО ПРОДАННЫХ ТОВАРОВ: {info['total_quantity']:,.0f} шт.")
        report.append("")

        # Анализ по сезонам
        report.append("-" * 70)
        report.append("📈 АНАЛИЗ ПО СЕЗОНАМ")
        report.append("-" * 70)
        report.append("")

        for _, row in season_data.iterrows():
            report.append(f"🌿 {row['season'].upper()}")
            report.append(f"   Выручка: {row['revenue_sum']:,.0f} руб. ({row['revenue_pct']}% от общей)")
            report.append(f"   Количество продаж: {row['quantity_sum']:,.0f} шт.")
            report.append("")

        # Топ продукты
        report.append("-" * 70)
        report.append("🏆 ТОП-10 САМЫХ ПРИБЫЛЬНЫХ ПРОДУКТОВ")
        report.append("-" * 70)
        report.append("")

        for _, row in top_products.iterrows():
            report.append(f"{row['rank']:2d}. {row['product']} ({row['category']})")
            report.append(f"     Выручка: {row['total_revenue']:,.0f} руб. | "
                         f"Продано: {row['total_quantity']:,.0f} шт.")
            report.append("")

        # Рекомендации по сезонам
        report.append("-" * 70)
        report.append("💡 РЕКОМЕНДАЦИИ ПО АССОРТИМЕНТУ")
        report.append("-" * 70)
        report.append("")

        season_emojis = {'весна': '🌸', 'лето': '☀️', 'осень': '🍂', 'зима': '❄️'}

        for season, products in recommendations.items():
            emoji = season_emojis.get(season, '')
            report.append(f"{emoji} {season.upper()} - увеличить запасы:")
            for i, product in enumerate(products[:5], 1):
                report.append(f"   {i}. {product}")
            report.append("")

        report.append("=" * 70)
        report.append("✅ Конец отчета")
        report.append("=" * 70)

        return "\n".join(report)


if __name__ == '__main__':
    print("📚 Модуль SeasonalSalesAnalyzer загружен")
    print("💡 Использование:")
    print("   analyzer = SeasonalSalesAnalyzer(df)")
    print("   analyzer.plot_seasonal_revenue()")
    print("   print(analyzer.generate_full_report())")
