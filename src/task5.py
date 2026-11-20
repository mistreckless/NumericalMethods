import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.utils import save_figure


def test_six_hypotheses(x: np.ndarray, y: np.ndarray, m0: float, d0: float, alpha: float = 0.1) -> pd.DataFrame:
    """Проверяет все 6 статистических гипотез при уровне значимости alpha."""
    nx, ny = len(x), len(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    t_stat_1, p_val_1 = stats.ttest_1samp(x, popmean=m0)

    chi2_stat = (nx - 1) * var_x / d0
    cdf_chi2 = stats.chi2.cdf(chi2_stat, df=nx - 1)
    p_val_2 = 2 * min(cdf_chi2, 1.0 - cdf_chi2)

    sw_stat, p_val_3 = stats.shapiro(x)

    f_stat = var_x / var_y
    cdf_f = stats.f.cdf(f_stat, dfn=nx - 1, dfd=ny - 1)
    p_val_4 = 2 * min(cdf_f, 1.0 - cdf_f)

    t_stat_2, p_val_5 = stats.ttest_ind(x, y, equal_var=(p_val_4 >= alpha))

    u_stat, p_val_6 = stats.mannwhitneyu(x, y, alternative='two-sided')

    results = [
        ("1. H0: m_X = 3 (t-тест)", f"t = {t_stat_1:.4f}", p_val_1),
        ("2. H0: D_X = 8 (χ²-тест)", f"χ² = {chi2_stat:.4f}", p_val_2),
        ("3. H0: X ~ N(m, D) (Шапиро-Уилк)", f"W = {sw_stat:.4f}", p_val_3),
        ("4. H0: D_X = D_Y (F-тест Фишера)", f"F = {f_stat:.4f}", p_val_4),
        ("5. H0: m_X = m_Y (2-выб. t-тест)", f"t = {t_stat_2:.4f}", p_val_5),
        ("6. H0: Med_X = Med_Y (Манн-Уитни)", f"U = {u_stat:.1f}", p_val_6),
    ]

    rows = []
    for name, stat_str, p_val in results:
        decision = "Принимается H0" if p_val >= alpha else "Отвергается H0"
        rows.append({
            "Проверяемая гипотеза": name,
            "Статистика": stat_str,
            "p-value": f"{p_val:.4f}",
            f"Решение (α = {alpha})": decision
        })

    return pd.DataFrame(rows)


def save_hypotheses_table_png(df: pd.DataFrame, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.38, 0.18, 0.16, 0.28]
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
            if col == 3:
                txt = cell.get_text().get_text()
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#1b7a3d' if 'Принимается' in txt else '#b82525')
        if col == 0:
            cell.get_text().set_horizontalalignment('left')

    ax.set_title('Результаты проверки статистических гипотез (подразд. 5.3 и 5.4)', fontsize=11, weight='bold', pad=10)
    fig.tight_layout()
    save_figure(fig, filename, bbox_inches='tight')


def plot_two_samples(x: np.ndarray, y: np.ndarray, filename: str) -> None:
    """Строит сравнительные графики для выборок X и Y (аналог Box-and-Whisker и плотностей в STATGRAPHICS)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    sns.histplot(x, bins=10, stat='density', kde=True, color='steelblue', alpha=0.45, label=f'X ~ N(3, 8), n={len(x)}', ax=ax1)
    sns.histplot(y, bins=12, stat='density', kde=True, color='darkorange', alpha=0.45, label=f'Y ~ N(4.5, 11), n={len(y)}', ax=ax1)
    ax1.axvline(np.mean(x), color='steelblue', linestyle='--', lw=1.5)
    ax1.axvline(np.mean(y), color='darkorange', linestyle='--', lw=1.5)
    ax1.set_title('Распределения выборок X и Y')
    ax1.set_xlabel('Значение')
    ax1.set_ylabel('Плотность')
    ax1.legend()

    bp_data = pd.DataFrame({
        'Выборка': ['Выборка X (n=50)'] * len(x) + ['Выборка Y (n=100)'] * len(y),
        'Значение': np.concatenate([x, y])
    })
    sns.boxplot(data=bp_data, x='Выборка', y='Значение', hue='Выборка', palette=['steelblue', 'darkorange'], width=0.45, legend=False, ax=ax2)
    ax2.set_title('Сравнение выборок (Box-and-Whisker Plot)')
    ax2.set_xlabel('')
    ax2.set_ylabel('Значение')

    fig.tight_layout()
    save_figure(fig, filename)



def main():
    sns.set_theme(style="whitegrid")
    out_dir = "../output/task5"

    # Параметры (март = 3, номер в журнале = 8)
    mx, dx, nx = 3.0, 8.0, 50
    my, dy, ny = mx + 1.5, dx + 3.0, 100
    alpha = 0.1

    np.random.seed(42)
    sample_x = np.random.normal(loc=mx, scale=np.sqrt(dx), size=nx)
    sample_y = np.random.normal(loc=my, scale=np.sqrt(dy), size=ny)

    print(f"Выборка X: среднее = {np.mean(sample_x):.4f}, дисперсия = {np.var(sample_x, ddof=1):.4f}")
    print(f"Выборка Y: среднее = {np.mean(sample_y):.4f}, дисперсия = {np.var(sample_y, ddof=1):.4f}\n")

    hyp_df = test_six_hypotheses(sample_x, sample_y, m0=mx, d0=dx, alpha=alpha)
    print(hyp_df.to_string(index=False))

    save_hypotheses_table_png(hyp_df, f"{out_dir}/0_hypotheses_table.png")
    plot_two_samples(sample_x, sample_y, f"{out_dir}/1_samples_comparison.png")


if __name__ == "__main__":
    main()
