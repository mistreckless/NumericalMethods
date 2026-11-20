import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.utils import save_figure


def urand_generator(seed: int = 12345):
    """Классический генератор равномерных чисел URAND на интервале (0, 1)."""
    state = seed
    mod = 2**31
    a = 843314861
    c = 453816693
    while True:
        state = (a * state + c) % mod
        u = state / mod
        if 0.0 < u < 1.0:
            yield u


def generate_gamma_full(sample_size: int, alpha: float, beta: float, seed: int = 42) -> np.ndarray:
    """
    Моделирование выборки гамма-распределения для любого нецелого alpha > 0
    строго по полной системе формул из методички (через URAND).
    """
    if alpha <= 0 or float(alpha).is_integer():
        raise ValueError("Параметр alpha должен быть положительным нецелым числом (например, 2.6).")

    rng = urand_generator(seed)
    sample = np.zeros(sample_size)

    # Разложение alpha на целую и дробную части по формулам методички:
    m = int(np.floor(alpha + 1)) - 1   # число множителей [alpha + 1] - 1
    alpha_1 = alpha - np.floor(alpha)  # дробная часть alpha - [alpha] в (0, 1)

    for k in range(sample_size):
        # 1. Моделирование x для дробной части alpha_1 (метод отбора)
        while True:
            r1 = next(rng)
            r2 = next(rng)
            s1 = r1 ** (1.0 / alpha_1)
            s2 = r2 ** (1.0 / (1.0 - alpha_1))
            if s1 + s2 <= 1.0:
                r3 = next(rng)
                x = -beta * (s1 / (s1 + s2)) * np.log(r3)
                break

        # 2. Учет целой части через произведение m равномерных чисел r_i
        prod_r = 1.0
        for _ in range(m):
            prod_r *= next(rng)

        y = x - beta * np.log(prod_r) if m > 0 else x
        sample[k] = y

    return sample


def compare_samples_table(custom_sample: np.ndarray, builtin_sample: np.ndarray, alpha: float, beta: float) -> pd.DataFrame:
    """Сравнивает выборку по формуле (URAND), встроенную (аналог rgamma) и теорию."""
    t_mean = alpha * beta
    t_var = alpha * (beta ** 2)

    return pd.DataFrame({
        'Показатель': [
            'Объем выборки', 'Параметры (α, β)',
            'Среднее значение', 'Дисперсия',
            'СКО', 'Медиана', 'Минимум', 'Максимум'
        ],
        'По формуле (URAND)': [
            f"{len(custom_sample)}", f"({alpha}, {beta})",
            f"{np.mean(custom_sample):.4f}", f"{np.var(custom_sample, ddof=1):.4f}",
            f"{np.std(custom_sample, ddof=1):.4f}", f"{np.median(custom_sample):.4f}",
            f"{np.min(custom_sample):.4f}", f"{np.max(custom_sample):.4f}"
        ],
        'Встроенный (rgamma)': [
            f"{len(builtin_sample)}", f"({alpha}, {beta})",
            f"{np.mean(builtin_sample):.4f}", f"{np.var(builtin_sample, ddof=1):.4f}",
            f"{np.std(builtin_sample, ddof=1):.4f}", f"{np.median(builtin_sample):.4f}",
            f"{np.min(builtin_sample):.4f}", f"{np.max(builtin_sample):.4f}"
        ],
        'Теория': [
            '—', f"({alpha}, {beta})",
            f"{t_mean:.4f}", f"{t_var:.4f}",
            f"{np.sqrt(t_var):.4f}", '—', '0.0000', '∞'
        ]
    })


def save_comparison_table_png(df: pd.DataFrame, filename: str) -> None:
    """Сохраняет сравнительную таблицу в PNG."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colWidths=[0.31, 0.25, 0.25, 0.19])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if row == 0:
            cell.set_facecolor('#2b5c8f')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor('#f4f7fa' if row % 2 == 0 else 'white')
        if col == 0:
            cell.get_text().set_horizontalalignment('left')

    ax.set_title('Сравнение выборок Гамма-распределения (N = 100)', fontsize=11, weight='bold', pad=10)
    fig.tight_layout()
    save_figure(fig, filename, bbox_inches='tight')


def plot_comparison(custom_sample: np.ndarray, builtin_sample: np.ndarray, alpha: float, beta: float, out_dir: str) -> None:
    """Строит графики последовательностей и сравнительную гистограмму плотностей."""
    indices = np.arange(1, len(custom_sample) + 1)

    # 1. Графики самих выборок (по номеру испытания)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    axes.plot(indices, custom_sample, marker='o', markersize=4, linestyle='-', lw=1, color='steelblue')
    axes.axhline(np.mean(custom_sample), color='red', linestyle='--', label=f'Среднее = {np.mean(custom_sample):.3f}')
    axes.set_title(f'По формуле методички (URAND), α={alpha}, β={beta}')
    axes.set_xlabel('Номер испытания')
    axes.set_ylabel('Значение y')
    axes.legend()

    axes.plot(indices, builtin_sample, marker='s', markersize=4, linestyle='-', lw=1, color='seagreen')
    axes.axhline(np.mean(builtin_sample), color='red', linestyle='--', label=f'Среднее = {np.mean(builtin_sample):.3f}')
    axes.set_title(f'Встроенный генератор (rgamma), α={alpha}, β={beta}')
    axes.set_xlabel('Номер испытания')
    axes.legend()

    fig.tight_layout()
    save_figure(fig, f"{out_dir}/1_samples_sequences.png")

    # 2. Сравнительная гистограмма с теоретической кривой плотности
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = np.linspace(0, max(np.max(custom_sample), np.max(builtin_sample)), 11)

    ax.hist(custom_sample, bins=bins, density=True, alpha=0.5, color='steelblue', edgecolor='black', label='По формуле (URAND)')
    ax.hist(builtin_sample, bins=bins, density=True, alpha=0.5, color='seagreen', edgecolor='black', label='Встроенный (rgamma)')

    x_grid = np.linspace(0.01, bins[-1] * 1.05, 300)
    ax.plot(x_grid, stats.gamma.pdf(x_grid, a=alpha, scale=beta), color='red', lw=2, label=f'Теор. плотность Gamma({alpha}, {beta})')

    ax.set_title('Сравнение распределений двух выборок')
    ax.set_xlabel('Значение y')
    ax.set_ylabel('Плотность вероятности')
    ax.legend()

    fig.tight_layout()
    save_figure(fig, f"{out_dir}/2_samples_hist_comparison.png")


def main():
    sns.set_theme(style="whitegrid")
    out_dir = "../output/task3"

    # Параметры: нецелое alpha > 0 и параметр масштаба beta > 0
    alpha = 2.6
    beta = 1.5
    sample_size = 100

    # 1. Генерация по полной формуле методички (URAND)
    custom_sample = generate_gamma_full(sample_size=sample_size, alpha=alpha, beta=beta, seed=101)

    # 2. Генерация встроенной функцией (аналог rgamma в Mathcad)
    np.random.seed(101)
    builtin_sample = stats.gamma.rvs(a=alpha, scale=beta, size=sample_size)

    # 3. Таблица и графики
    comp_df = compare_samples_table(custom_sample, builtin_sample, alpha, beta)
    print(comp_df.to_string(index=False))
    save_comparison_table_png(comp_df, f"{out_dir}/0_gamma_comparison_table.png")
    plot_comparison(custom_sample, builtin_sample, alpha, beta, out_dir)
    print(f"\nГотово! Файлы сохранены в: {out_dir}")


if __name__ == "__main__":
    main()
