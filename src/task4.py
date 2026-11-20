import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_figure


def estimate_f_parameters(sample: np.ndarray, n1_true: int, n2_true: int, conf_level: float = 0.95) -> tuple[pd.DataFrame, tuple[float, float]]:
    """
    Вычисляет точечные и интервальные оценки матожидания и дисперсии (beta = 0.95),
    а также оценивает параметры F-распределения (n1, n2) методом максимального правдоподобия (ММП).
    """
    m = len(sample)
    alpha = 1.0 - conf_level

    mean_est = np.mean(sample)
    var_est = np.var(sample, ddof=1)
    std_est = np.std(sample, ddof=1)

    mean_true = n2_true / (n2_true - 2) if n2_true > 2 else np.nan
    var_true = (2 * n2_true**2 * (n1_true + n2_true - 2)) / (n1_true * (n2_true - 2)**2 * (n2_true - 4)) if n2_true > 4 else np.nan

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=m - 1)
    mean_margin = t_crit * (std_est / np.sqrt(m))
    mean_ci = (mean_est - mean_margin, mean_est + mean_margin)

    chi2_lower = stats.chi2.ppf(alpha / 2.0, df=m - 1)
    chi2_upper = stats.chi2.ppf(1.0 - alpha / 2.0, df=m - 1)
    var_ci = ((m - 1) * var_est / chi2_upper, (m - 1) * var_est / chi2_lower)

    n1_mle, n2_mle, _, _ = stats.f.fit(sample, floc=0, fscale=1)

    mean_low, mean_high = mean_ci
    var_low, var_high = var_ci

    summary_df = pd.DataFrame({
        'Оцениваемый параметр': [
            'Математическое ожидание M[X]',
            'Дисперсия D[X]',
            'Параметр формы n1 (степени свободы 1)',
            'Параметр формы n2 (степени свободы 2)'
        ],
        'Истинное (теория)': [
            f"{mean_true:.4f}",
            f"{var_true:.4f}",
            f"{n1_true}",
            f"{n2_true}"
        ],
        'Точечная оценка / ММП': [
            f"{mean_est:.4f}",
            f"{var_est:.4f}",
            f"{n1_mle:.4f} (≈ {round(n1_mle)})",
            f"{n2_mle:.4f} (≈ {round(n2_mle)})"
        ],
        f'Доверительный интервал (β={conf_level})': [
            f"({mean_low:.4f}; {mean_high:.4f})",
            f"({var_low:.4f}; {var_high:.4f})",
            '— (оценка ММП)',
            '— (оценка ММП)'
        ]
    })

    return summary_df, (n1_mle, n2_mle)


def save_estimates_table_png(df: pd.DataFrame, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.34, 0.16, 0.22, 0.28]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)

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

    ax.set_title('Точечные, интервальные (β = 0.95) и ММП-оценки F-распределения', fontsize=11, weight='bold', pad=10)
    fig.tight_layout()
    save_figure(fig, filename, bbox_inches='tight')


def plot_mle_fit(sample: np.ndarray, n1_true: int, n2_true: int, n1_mle: float, n2_mle: float, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.histplot(sample, bins=12, stat='density', color='steelblue', edgecolor='black', alpha=0.5, label=f'Выборка (m={len(sample)})', ax=ax)

    x_grid = np.linspace(0.01, np.max(sample) * 1.05, 400)
    ax.plot(x_grid, stats.f.pdf(x_grid, n1_true, n2_true), color='red', lw=2, linestyle='--',
            label=f'Истинная плотность dF(x, {n1_true}, {n2_true})')
    ax.plot(x_grid, stats.f.pdf(x_grid, n1_mle, n2_mle), color='darkgreen', lw=2,
            label=f'Оценка ММП dF(x, {n1_mle:.2f}, {n2_mle:.2f})')

    ax.set_title('Сравнение теоретической плотности и модели по ММП')
    ax.set_xlabel('Значение x')
    ax.set_ylabel('Плотность вероятности')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, filename)


def main():
    sns.set_theme(style="whitegrid")
    out_dir = "../output/task4"

    n1_true, n2_true = 5, 10
    m = 100
    conf_level = 0.95

    np.random.seed(42)
    sample = stats.f.rvs(n1_true, n2_true, size=m)

    estimates_df, (n1_mle, n2_mle) = estimate_f_parameters(sample, n1_true, n2_true, conf_level)
    print(estimates_df.to_string(index=False))

    save_estimates_table_png(estimates_df, f"{out_dir}/0_f_estimates_table.png")
    plot_mle_fit(sample, n1_true, n2_true, n1_mle, n2_mle, f"{out_dir}/1_f_mle_fit.png")


if __name__ == "__main__":
    main()
