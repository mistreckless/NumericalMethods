import matplotlib.pyplot as plt
from pathlib import Path


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300, **kwargs) -> None:
    """Создает недостающие директории по пути filename и сохраняет график."""
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, **kwargs)
    plt.close(fig)