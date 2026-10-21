import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
ZERO_FA_PLOT = 0.03  # a log x-axis cannot show zero: null false alarms are drawn here (graphics only)
SCEN_LABEL = {"none": "nessun distacco", "slowdown": "rallenta", "stop": "si ferma"}


def style_axes(ax, title_color: bool = False) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    if title_color:
        ax.title.set_color(INK)


def save_figure(fig, path, dpi: int, tight_layout: bool = True, **savefig_kwargs) -> None:
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, **savefig_kwargs)
    plt.close(fig)
