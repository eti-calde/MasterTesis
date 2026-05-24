"""
Figura de comparación de los tres diseños de arquitectura para la Sección
Metodología: A1 (propuesta, dos redes), A2 (monolítica, estilo Dazzi) y
A3 (una red por campo, estilo Ohara).

Esquema cualitativo: NO muestra conteos de parámetros, porque cada diseño se
evalúa en un estudio de escalamiento (presupuestos pequeño/medio/grande).
Las tres arquitecturas resuelven el mismo problema inverso de batimetría.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "architecture_designs.png"

C_IN = "#2980b9"    # entradas
C_OUT = "#f39c12"   # salidas
C_A1 = "#27ae60"    # redes de A1 (verde)
C_A2 = "#e74c3c"    # red de A2 (rojo)
C_A3 = "#8e44ad"    # redes de A3 (morado)


def box(ax, x, y, w, h, label, color, fs=9, tc="white"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.9,
                               edgecolor="black", lw=1.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333333"))


def setup(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


def panel_two_net(ax, title, net_color, flow_label, bath_label, footer):
    """Diseño de dos redes (flujo arriba, batimetría abajo)."""
    setup(ax, title)
    box(ax, 0.2, 5.0, 1.5, 1.3, "$x,\\,t$", C_IN)
    box(ax, 0.2, 1.7, 1.5, 1.3, "$x$", C_IN)
    box(ax, 2.7, 4.3, 3.4, 2.7, flow_label, net_color)
    box(ax, 2.7, 1.0, 3.4, 2.7, bath_label, net_color)
    box(ax, 7.1, 6.0, 1.4, 1.0, "$h$", C_OUT, fs=11, tc="black")
    box(ax, 7.1, 4.6, 1.4, 1.0, "$u$", C_OUT, fs=11, tc="black")
    box(ax, 7.1, 1.85, 1.4, 1.0, "$z_b$", C_OUT, fs=11, tc="black")
    arrow(ax, 1.7, 5.65, 2.7, 5.65)
    arrow(ax, 1.7, 2.35, 2.7, 2.35)
    arrow(ax, 6.1, 5.9, 7.1, 6.5)
    arrow(ax, 6.1, 5.4, 7.1, 5.1)
    arrow(ax, 6.1, 2.35, 7.1, 2.35)
    ax.text(5.0, 0.2, footer, ha="center", fontsize=8.5, style="italic")


def panel_mono(ax, title, footer):
    """Diseño monolítico (una sola red)."""
    setup(ax, title)
    box(ax, 0.2, 3.7, 1.5, 1.3, "$x,\\,t$", C_IN)
    box(ax, 2.7, 1.4, 3.4, 5.6,
        "MLP único\n$(x,t)\\rightarrow$\n$(h,u,z_b)$\nentradas crudas", C_A2)
    box(ax, 7.1, 5.5, 1.4, 1.0, "$h$", C_OUT, fs=11, tc="black")
    box(ax, 7.1, 3.85, 1.4, 1.0, "$u$", C_OUT, fs=11, tc="black")
    box(ax, 7.1, 2.2, 1.4, 1.0, "$z_b$", C_OUT, fs=11, tc="black")
    arrow(ax, 1.7, 4.35, 2.7, 4.35)
    arrow(ax, 6.1, 4.7, 7.1, 6.0)
    arrow(ax, 6.1, 4.2, 7.1, 4.35)
    arrow(ax, 6.1, 3.7, 7.1, 2.7)
    ax.text(5.0, 0.45, footer, ha="center", fontsize=8.5, style="italic")


def panel_per_field(ax, title, net_color, footer):
    """Diseño de una red por campo (estilo Ohara): una FNN independiente por
    cada campo de salida."""
    setup(ax, title)
    rows = [(6.6, "$h$", "$x,\\,t$"),
            (4.85, "$u$", "$x,\\,t$"),
            (3.1, "$v$", "$x,\\,t$"),
            (1.35, "$z_b$", "$x$")]
    for y, out, inp in rows:
        box(ax, 0.3, y, 1.5, 1.2, inp, C_IN, fs=9)
        box(ax, 2.7, y, 3.4, 1.2, "FNN", net_color, fs=10)
        box(ax, 7.1, y + 0.1, 1.3, 1.0, out, C_OUT, fs=11, tc="black")
        arrow(ax, 1.8, y + 0.6, 2.7, y + 0.6)
        arrow(ax, 6.1, y + 0.6, 7.1, y + 0.6)
    ax.text(5.0, 0.5, footer, ha="center", fontsize=8.5, style="italic")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

panel_two_net(
    axes[0], "A1 — Propuesta: dos redes", C_A1,
    "SolNet\n$(x,t)\\rightarrow(h,u)$\nFourier",
    "BathNet\n$x\\rightarrow z_b$\nFourier",
    "$\\partial z_b/\\partial t = 0$ estructural (BathNet sin $t$)")

panel_mono(
    axes[1], "A2 — Monolítica (estilo Dazzi)",
    "$\\partial z_b/\\partial t = 0$ como término de pérdida")

panel_per_field(
    axes[2], "A3 — Una red por campo (estilo Ohara)", C_A3,
    "$\\partial z_b/\\partial t = 0$ estructural ($z_b$ sin entrada $t$)")

fig.suptitle("Diseños de arquitectura comparados (inversión de batimetría)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
