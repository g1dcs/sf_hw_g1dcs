"""
Модуль для анализа цен и маржинальности продуктов питания.

Author: g1dcs
Date: 2026-03-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MarginAnalyzer:
    """Класс для анализа цен и маржинальности."""

    # Себестоимость как % от цены (примерные данные)
    CATEGORY_COST_RATIO = {
        'фрукты': 0.55,
        'овощи': 0.50,
        'молочные продукты': 0.65,
        'мясо': 0.70,
        'напитки': 0.40,
        'выпечка': 0.45
    }

    def __init__(self, df: pd.DataFrame):
        """Инициализация анализатора.

        Args:
            df (pd.DataFrame): DataFrame с данными о продажах
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Добавляем расчет себестоимости и маржи
        self._calculate_margin()

    def _calculate_margin(self) -> None:
        """Рассчитывает себестоимость и маржинальность."""
        # Себестоимость единицы товара
        self.df['cost_price'] = self.df.apply(
            lambda row: row['price'] * self.CATEGORY_COST_RATIO.get(row['category'], 0.5), 
            axis=1
        )

        # Маржа на единицу
        self.df['margin_per_unit'] = self.df['price'] - self.df['cost_price']

        # Общая маржа
        self.df['total_margin'] = self.df['margin_per_unit'] * self.df['quantity']

        # Маржинальность в %
        self.df['margin_percent'] = (self.df['margin_per_unit'] / self.df['price'] * 100).round(2)

        # Рентабельность продаж (ROS)
        self.df['ros'] = (self.df['total_margin'] / self.df['revenue'] * 100).round(2)

    def analyze_margin_by_category(self) -> pd.DataFrame:
        """Анализирует маржинальность по категориям.

        Returns:
            pd.DataFrame: Сводная таблица по категориям
        """
        category_margin = self.df.groupby('category').agg({
            'revenue': 'sum',
            'total_margin': 'sum',
            'margin_percent': 'mean',
            'quantity': 'sum',
            'price': 'mean'
        }).round(2)

        category_margin['margin_ratio'] = (
            category_margin['total_margin'] / category_margin['revenue'] * 100
        ).round(2)

        category_margin = category_margin.sort_values('total_margin', ascending=False)

        return category_margin.reset_index()

    def analyze_margin_by_season(self) -> pd.DataFrame:
        """Анализирует маржинальность по сезонам.

        Returns:
            pd.DataFrame: Сводная таблица по сезонам
        """
        season_margin = self.df.groupby('season').agg({
            'revenue': 'sum',
            'total_margin': 'sum',
            'margin_percent': 'mean',
            'quantity': 'sum'
        }).round(2)

        season_margin['margin_ratio'] = (
            season_margin['total_margin'] / season_margin['revenue'] * 100
        ).round(2)

        # Сортируем по сезонам
        season_order = ['весна', 'лето', 'осень', 'зима']
        season_margin = season_margin.reindex(season_order)

        return season_margin.reset_index()

    def analyze_price_dynamics(self) -> pd.DataFrame:
        """Анализирует динамику цен по сезонам.

        Returns:
            pd.DataFrame: Средние цены по сезонам
        """
        price_dynamics = self.df.groupby(['category', 'season'])['price'].mean().unstack()

        # Сортируем сезоны
        season_order = ['весна', 'лето', 'осень', 'зима']
        price_dynamics = price_dynamics.reindex(columns=season_order)

        # Добавляем изменение цены зимой относительно лета
        if 'лето' in price_dynamics.columns and 'зима' in price_dynamics.columns:
            price_dynamics['change_winter_summer'] = (
                (price_dynamics['зима'] - price_dynamics['лето']) / price_dynamics['лето'] * 100
            ).round(2)

        return price_dynamics.reset_index()

    def get_most_profitable_products(self, top_n: int = 15) -> pd.DataFrame:
        """Возвращает продукты с наибольшей маржой.

        Args:
            top_n (int): Количество продуктов

        Returns:
            pd.DataFrame: Топ продуктов по марже
        """
        product_margin = self.df.groupby(['category', 'product']).agg({
            'total_margin': 'sum',
            'margin_percent': 'mean',
            'revenue': 'sum',
            'quantity': 'sum'
        }).round(2)

        product_margin['margin_ratio'] = (
            product_margin['total_margin'] / product_margin['revenue'] * 100
        ).round(2)

        product_margin = product_margin.sort_values('total_margin', ascending=False).head(top_n)
        product_margin = product_margin.reset_index()
        product_margin['rank'] = range(1, len(product_margin) + 1)

        return product_margin

    def analyze_seasonal_margin_trends(self) -> pd.DataFrame:
        """Анализирует тренды маржинальности по сезонам.

        Returns:
            pd.DataFrame: Маржинальность по категориям и сезонам
        """
        seasonal_margin = self.df.groupby(['category', 'season']).agg({
            'total_margin': 'sum',
            'margin_percent': 'mean',
            'revenue': 'sum'
        }).round(2)

        seasonal_margin['margin_ratio'] = (
            seasonal_margin['total_margin'] / seasonal_margin['revenue'] * 100
        ).round(2)

        return seasonal_margin.reset_index()

    def plot_margin_by_category(self, save_path: str = None) -> None:
        """Строит график маржинальности по категориям.

        Args:
            save_path (str, optional): Путь для сохранения
        """
        margin_data = self.analyze_margin_by_category()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # График 1: Абсолютная маржа
        ax1 = axes[0]
        bars1 = ax1.bar(margin_data['category'], margin_data['total_margin'], 
                        color='steelblue', edgecolor='black')
        ax1.set_title('Общая маржа по категориям', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Категория')
        ax1.set_ylabel('Маржа (руб.)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
        ax1.tick_params(axis='x', rotation=45)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/1e6:.1f}M',
                    ha='center', va='bottom', fontsize=9)

        # График 2: Маржинальность в %
        ax2 = axes[1]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(margin_data)))
        bars2 = ax2.bar(margin_data['category'], margin_data['margin_percent'], 
                        color=colors, edgecolor='black')
        ax2.set_title('Средняя маржинальность по категориям (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Категория')
        ax2.set_ylabel('Маржинальность (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Целевая 30%')
        ax2.legend()

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")

        plt.show()

    def plot_margin_by_season(self, save_path: str = None) -> None:
        """Строит график маржинальности по сезонам.

        Args:
            save_path (str, optional): Путь для сохранения
        """
        season_data = self.analyze_margin_by_season()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # График 1: Выручка vs Маржа
        ax1 = axes[0]
        x = np.arange(len(season_data))
        width = 0.35

        bars1 = ax1.bar(x - width/2, season_data['revenue'], width, 
                        label='Выручка', color='skyblue', edgecolor='black')
        bars2 = ax1.bar(x + width/2, season_data['total_margin'], width,
                        label='Маржа', color='lightgreen', edgecolor='black')

        ax1.set_title('Выручка и маржа по сезонам', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Сезон')
        ax1.set_ylabel('Сумма (руб.)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(season_data['season'])
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))

        # График 2: Маржинальность в %
        ax2 = axes[1]
        colors = ['#90EE90', '#FFD700', '#FF8C00', '#87CEEB']
        bars = ax2.bar(season_data['season'], season_data['margin_percent'], 
                       color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_title('Маржинальность по сезонам (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Сезон')
        ax2.set_ylabel('Маржинальность (%)')
        ax2.axhline(y=season_data['margin_percent'].mean(), color='red', 
                   linestyle='--', alpha=0.7, label=f'Средняя: {season_data["margin_percent"].mean():.1f}%')
        ax2.legend()

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")

        plt.show()

    def plot_price_dynamics(self, save_path: str = None) -> None:
        """Строит график динамики цен по сезонам.

        Args:
            save_path (str, optional): Путь для сохранения
        """
        price_data = self.analyze_price_dynamics()

        # Создаем сводную таблицу
        season_cols = ['весна', 'лето', 'осень', 'зима']
        pivot_data = price_data.set_index('category')[season_cols]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Строим группированный бар-чарт
        pivot_data.plot(kind='bar', ax=ax, width=0.8, 
                       color=['#90EE90', '#FFD700', '#FF8C00', '#87CEEB'],
                       edgecolor='black')

        ax.set_title('Средние цены по категориям и сезонам', fontsize=14, fontweight='bold')
        ax.set_xlabel('Категория')
        ax.set_ylabel('Средняя цена (руб.)')
        ax.legend(title='Сезон', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")

        plt.show()

    def plot_margin_heatmap(self, save_path: str = None) -> None:
        """Строит тепловую карту маржинальности.

        Args:
            save_path (str, optional): Путь для сохранения
        """
        margin_data = self.analyze_seasonal_margin_trends()

        # Создаем сводную таблицу
        pivot = margin_data.pivot(index='category', columns='season', values='margin_percent')

        # Сортируем сезоны
        season_order = ['весна', 'лето', 'осень', 'зима']
        pivot = pivot.reindex(columns=season_order)

        # Строим тепловую карту
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=35, vmin=20, vmax=50,
                   cbar_kws={'label': 'Маржинальность (%)'},
                   linewidths=0.5, ax=ax)

        ax.set_title('Маржинальность по категориям и сезонам (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Сезон')
        ax.set_ylabel('Категория')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ График сохранен: {save_path}")

        plt.show()

    def generate_margin_report(self) -> str:
        """Генерирует отчет по маржинальности.

        Returns:
            str: Текстовый отчет
        """
        category_margin = self.analyze_margin_by_category()
        season_margin = self.analyze_margin_by_season()
        top_products = self.get_most_profitable_products(top_n=10)

        report = []
        report.append("=" * 70)
        report.append("💰 ОТЧЕТ ПО МАРЖИНАЛЬНОСТИ")
        report.append("=" * 70)
        report.append("")

        # Общая информация
        total_revenue = self.df['revenue'].sum()
        total_margin = self.df['total_margin'].sum()
        avg_margin_pct = self.df['margin_percent'].mean()

        report.append(f"📊 ОБЩИЕ ПОКАЗАТЕЛИ:")
        report.append(f"   Общая выручка: {total_revenue:,.0f} руб.")
        report.append(f"   Общая маржа: {total_margin:,.0f} руб.")
        report.append(f"   Средняя маржинальность: {avg_margin_pct:.2f}%")
        report.append("")

        # По категориям
        report.append("-" * 70)
        report.append("📈 МАРЖИНАЛЬНОСТЬ ПО КАТЕГОРИЯМ:")
        report.append("-" * 70)
        report.append("")

        for _, row in category_margin.iterrows():
            report.append(f"📂 {row['category'].upper()}")
            report.append(f"   Маржа: {row['total_margin']:,.0f} руб.")
            report.append(f"   Маржинальность: {row['margin_percent']:.2f}%")
            report.append(f"   Доля в общей марже: {row['total_margin']/total_margin*100:.1f}%")
            report.append("")

        # По сезонам
        report.append("-" * 70)
        report.append("🌿 МАРЖИНАЛЬНОСТЬ ПО СЕЗОНАМ:")
        report.append("-" * 70)
        report.append("")

        season_emojis = {'весна': '🌸', 'лето': '☀️', 'осень': '🍂', 'зима': '❄️'}

        for _, row in season_margin.iterrows():
            emoji = season_emojis.get(row['season'], '')
            report.append(f"{emoji} {row['season'].upper()}")
            report.append(f"   Маржа: {row['total_margin']:,.0f} руб.")
            report.append(f"   Маржинальность: {row['margin_percent']:.2f}%")
            report.append("")

        # Топ продукты
        report.append("-" * 70)
        report.append("🏆 ТОП-10 ПРОДУКТОВ ПО МАРЖЕ:")
        report.append("-" * 70)
        report.append("")

        for _, row in top_products.iterrows():
            report.append(f"{row['rank']:2d}. {row['product']} ({row['category']})")
            report.append(f"     Маржа: {row['total_margin']:,.0f} руб. | "
                         f"Маржинальность: {row['margin_percent']:.2f}%")
            report.append("")

        # Рекомендации
        report.append("-" * 70)
        report.append("💡 РЕКОМЕНДАЦИИ:")
        report.append("-" * 70)
        report.append("")

        # Находим категории с высокой и низкой маржинальностью
        high_margin = category_margin[category_margin['margin_percent'] > 40]['category'].tolist()
        low_margin = category_margin[category_margin['margin_percent'] < 30]['category'].tolist()

        if high_margin:
            report.append(f"✅ Высокомаржинальные категории (>40%): {', '.join(high_margin)}")
            report.append("   → Рекомендуется расширять ассортимент")
            report.append("")

        if low_margin:
            report.append(f"⚠️ Низкомаржинальные категории (<30%): {', '.join(low_margin)}")
            report.append("   → Требуется оптимизация цен или снижение себестоимости")
            report.append("")

        # Сезонные рекомендации
        best_season = season_margin.loc[season_margin['margin_percent'].idxmax(), 'season']
        worst_season = season_margin.loc[season_margin['margin_percent'].idxmin(), 'season']

        report.append(f"📅 Лучшая маржинальность в сезон: {best_season}")
        report.append(f"📅 Низшая маржинальность в сезон: {worst_season}")
        report.append("")

        report.append("=" * 70)
        report.append("✅ Конец отчета")
        report.append("=" * 70)

        return "\n".join(report)


if __name__ == '__main__':
    print("📚 Модуль MarginAnalyzer загружен")
    print("💡 Использование:")
    print("   analyzer = MarginAnalyzer(df)")
    print("   print(analyzer.generate_margin_report())")
