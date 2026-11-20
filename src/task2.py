from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_figure


def analyze_f_sample(sample: np.ndarray, n1: int, n2: int) -> pd.DataFrame:
    """Вычисляет числовые характеристики выборки (как в Mathcad) и теоретические значения."""
    # Выборочные характеристики (по аналогии с mean, median, var, stdev, min, max в Mathcad)
    s_mean = np.mean(sample)
    s_median = np.median(sample)
    s_var = np.var(sample, ddof=1)
    s_std = np.std(sample, ddof=1)
    s_min, s_max = np.min(sample), np.max(sample)

    # Теоретические моменты F-распределения (существуют при n2 > 2 и n2 > 4)
    t_mean = n2 / (n2 - 2) if n2 > 2 else np.nan
    t_var = (2 * n2**2 * (n1 + n2 - 2)) / (n1 * (n2 - 2)**2 * (n2 - 4)) if n2 > 4 else np.nan

    return pd.DataFrame({
        'Характеристика': [
            'Объем выборки (m)', 'Степени свободы (n1, n2)',
            'Выборочное среднее (mean)', 'Теоретическое матожидание',
            'Медиана (median)', 'Выборочная дисперсия (var)',
            'Теоретическая дисперсия', 'СКО (stdev)',
            'Минимум (min)', 'Максимум (max)', 'Размах (R)'
        ],
        'Значение': [
            f"{len(sample)}", f"({n1}, {n2})",
            f"{s_mean:.4f}", f"{t_mean:.4f}",
            f"{s_median:.4f}", f"{s_var:.4f}",
            f"{t_var:.4f}", f"{s_std:.4f}",
            f"{s_min:.4f}", f"{s_max:.4f}", f"{s_max - s_min:.4f}"
        ]
    })


def save_table_png(df: pd.DataFrame, filename: str) -> None:
    """Сохраняет таблицу характеристик в PNG."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left', colWidths=[0.68, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

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

    ax.set_title('Числовые характеристики F-распределения', fontsize=11, weight='bold', pad=10)
    fig.tight_layout()
    save_figure(fig, filename, bbox_inches='tight')


def plot_pdf_cdf_ppf(n1: int, n2: int, out_dir: str) -> None:
    """Строит графики плотности dF(x), функции распределения pF(x) и процентилей qF(alpha)."""
    x = np.linspace(0.001, 5.0, 500)
    alpha = np.linspace(0.01, 0.99, 500)

    # 1. Плотность вероятности dF(x, n1, n2)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x, stats.f.pdf(x, n1, n2), color='steelblue', lw=2)
    ax.set_title(f'Плотность вероятности dF(x, {n1}, {n2})')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    fig.tight_layout()
    save_figure(fig, f"{out_dir}/1_pdf_dF.png")

    # 2. Функция распределения pF(x, n1, n2)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x, stats.f.cdf(x, n1, n2), color='seagreen', lw=2)
    ax.set_title(f'Функция распределения pF(x, {n1}, {n2})')
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    fig.tight_layout()
    save_figure(fig, f"{out_dir}/2_cdf_pF.png")

    # 3. Функция значений процентилей qF(alpha, n1, n2)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(alpha, stats.f.ppf(alpha, n1, n2), color='darkorange', lw=2)
    ax.set_title(f'Функция процентилей qF(α, {n1}, {n2})')
    ax.set_xlabel('Вероятность α')
    ax.set_ylabel('Квантиль x_α')
    fig.tight_layout()
    save_figure(fig, f"{out_dir}/3_ppf_qF.png")


def plot_sample_hist(sample: np.ndarray, n1: int, n2: int, filename: str) -> None:
    """Строит гистограмму смоделированной выборки и накладывает теоретическую плотность."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.histplot(sample, bins=10, stat='density', color='steelblue', edgecolor='black', alpha=0.6, label='Выборка (m=100)', ax=ax)

    x = np.linspace(0.001, np.max(sample) * 1.05, 300)
    ax.plot(x, stats.f.pdf(x, n1, n2), color='red', lw=2, label=f'Теор. плотность dF(x, {n1}, {n2})')

    ax.set_title(f'Гистограмма выборки rF(100, {n1}, {n2})')
    ax.set_xlabel('Значение x')
    ax.set_ylabel('Плотность частоты')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, filename)


def main():
    sns.set_theme(style="whitegrid")
    out_dir = "../output/task2"

    # Задаем параметры F-распределения (n1, n2 > 4, чтобы существовали среднее и дисперсия)
    n1, n2 = 5, 10
    m = 100

    # Моделируем случайную выборку rF(m, n1, n2) с фиксированным seed для воспроизводимости
    np.random.seed(42)
    sample = stats.f.rvs(n1, n2, size=m)

    # Вычисляем и сохраняем числовые характеристики
    summary_df = analyze_f_sample(sample, n1, n2)
    print(summary_df.to_string(index=False))
    save_table_png(summary_df, f"{out_dir}/0_f_stats_table.png")

    # Строим и сохраняем все 4 графика
    plot_pdf_cdf_ppf(n1, n2, out_dir)
    plot_sample_hist(sample, n1, n2, f"{out_dir}/4_sample_hist.png")
    print(f"\nВсе файлы сохранены в папку: {out_dir}")


if __name__ == "__main__":
    main()
