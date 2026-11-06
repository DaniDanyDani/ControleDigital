"""Configuração visual comum para figuras técnicas no padrão IEEE."""

import matplotlib.pyplot as plt


def configurar_estilo_ieee() -> None:
    """Define tipografia serifada e tamanhos legíveis para figuras exportadas."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
