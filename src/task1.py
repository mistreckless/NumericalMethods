import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.utils import save_figure


def calculate_summary_stats(data: np.ndarray) -> pd.DataFrame:
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    var = np.var(data, ddof=1)
    sem = stats.sem(data)
    cv = (std / mean) * 100
    q25, median, q75 = np.percentile(data, [25, 50, 75])
    mode_val = stats.mode(data, keepdims=True).mode[0]
    skew = stats.skew(data, bias=False)
    kurt = stats.kurtosis(data, bias=False)

    stnd_skew = skew / np.sqrt(6 / n)
    stnd_kurt = kurt / np.sqrt(24 / n)

    summary = pd.DataFrame({
        'Показатель (STATGRAPHICS)': [
            'Count (Объем выборки)', 'Average (Среднее)', 'Median (Медиана)',
            'Mode (Мода)', 'Variance (Дисперсия)', 'Standard deviation (СКО)',
            'Standard error (Станд. ошибка)', 'Coeff. of variation (Коэф. вариации, %)',
            'Minimum (Минимум)', 'Maximum (Максимум)', 'Range (Размах)',
            'Lower quartile (Нижний квартиль Q1)', 'Upper quartile (Верхний квартиль Q3)',
            'Interquartile range (Межквартильный размах)',
            'Skewness (Асимметрия)', 'Stnd. skewness (Станд. асимметрия)',
            'Kurtosis (Эксцесс)', 'Stnd. kurtosis (Станд. эксцесс)'
        ],
        'Значение': [
            n, mean, median, mode_val, var, std, sem, cv,
            np.min(data), np.max(data), np.ptp(data),
            q25, q75, q75 - q25, skew, stnd_skew, kurt, stnd_kurt
        ]
    })
    return summary


def save_stats_table_png(df: pd.DataFrame, filename: str) -> None:
    formatted_df = df.copy()
    formatted_df['Значение'] = formatted_df.apply(
        lambda row: f"{int(row['Значение'])}" if 'Count' in row[
            'Показатель (STATGRAPHICS)'] else f"{row['Значение']:.5f}",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.axis('off')

    table = ax.table(
        cellText=formatted_df.values,
        colLabels=formatted_df.columns,
        loc='center',
        cellLoc='left',
        colWidths=[0.72, 0.28]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            cell.set_facecolor('#2b5c8f')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor('#f4f7fa' if row % 2 == 0 else 'white')

        if col == 1:
            cell.get_text().set_horizontalalignment('right')

    ax.set_title('Суммарные статистики (Summary Statistics)', fontsize=12, weight='bold', pad=12)
    fig.tight_layout()
    save_figure(fig, filename, bbox_inches='tight')


def plot_histogram(data: np.ndarray, filename) -> None:
    mean = np.mean(data)
    median = np.median(data)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data, bins=5, kde=True, color='steelblue', edgecolor='black', ax=ax)
    ax.axvline(mean, color='red', linestyle='--', linewidth=1.5, label=f'Среднее = {mean:.3f}')
    ax.axvline(median, color='green', linestyle=':', linewidth=1.5, label=f'Медиана = {median:.3f}')

    ax.set_title('Гистограмма (Frequency Histogram)')
    ax.set_xlabel('Емкость, пФ')
    ax.set_ylabel('Частота')
    ax.legend()

    fig.tight_layout()
    save_figure(fig, filename)


def plot_qq(data: np.ndarray, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    stats.probplot(data, dist="norm", plot=ax)

    ax.get_lines()[0].set_markerfacecolor('steelblue')
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[1].set_color('red')

    ax.set_title('Квантильный график (Normal Q-Q Plot)')
    ax.set_xlabel('Теоретические квантили')
    ax.set_ylabel('Упорядоченные значения (пФ)')

    fig.tight_layout()
    save_figure(fig, filename)


def plot_scatter(data: np.ndarray, filename: str) -> None:
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    indices = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(indices, data, color='steelblue', edgecolor='black', s=60, zorder=3)
    ax.axhline(mean, color='red', linestyle='--', linewidth=1.5, label='Среднее')
    ax.axhline(mean + 2 * std, color='orange', linestyle=':', label='±2 СКО')
    ax.axhline(mean - 2 * std, color='orange', linestyle=':')

    ax.set_title('Диаграмма рассеивания (Scatterplot)')
    ax.set_xlabel('Номер измерения (индекс)')
    ax.set_ylabel('Емкость, пФ')
    ax.set_xticks(range(1, n + 1, 2))
    ax.legend()

    fig.tight_layout()
    save_figure(fig, filename)


def main():
    sns.set_theme(style="whitegrid")

    capacitance_data = np.array([
        4.45, 4.40, 4.42, 4.45, 4.38, 4.42, 4.36, 4.35, 4.40, 4.45,
        4.42, 4.44, 4.36, 4.42, 4.44, 4.38, 4.39, 4.40, 4.42, 4.45
    ])

    stats_df = calculate_summary_stats(capacitance_data)
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    save_stats_table_png(stats_df, "../output/task1/summary_table.png")
    plot_histogram(capacitance_data, "../output/task1/histogram.png")
    plot_qq(capacitance_data, "../output/task1/qq_plot.png")
    plot_scatter(capacitance_data, "../output/task1/scatterplot.png")


if __name__ == "__main__":
    main()
