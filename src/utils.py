import matplotlib.pyplot as plt
from pathlib import Path


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300, **kwargs) -> None:
    """Создает недостающие директории по пути filename и сохраняет график."""
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, **kwargs)
    plt.close(fig)


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