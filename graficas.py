from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

##############################################################
# DISTANCIA MEDIA VS TIEMPO MEDIO:
##############################################################

# ============================================================
# CONFIGURACIÓN
# ============================================================

path_csv = Path("resultados_experimentos/resumen_estadistico.csv")
path_xlsx = Path("resultados_experimentos/resumen_estadistico.xlsx")

output_dir = Path("resultados_experimentos/figuras")
output_dir.mkdir(parents=True, exist_ok=True)

path_salida = output_dir / "fig_distancia_vs_tiempo.png"


# ============================================================
# LECTURA DE DATOS
# ============================================================

if path_csv.exists():
    df = pd.read_csv(path_csv)
    print(f"Datos leídos desde: {path_csv}")
elif path_xlsx.exists():
    df = pd.read_excel(path_xlsx)
    print(f"Datos leídos desde: {path_xlsx}")
else:
    raise FileNotFoundError(
        "No se ha encontrado resumen_estadistico.csv ni resumen_estadistico.xlsx"
    )


# ============================================================
# PREPARACIÓN
# ============================================================

nombres_algoritmos = {
    "ACO_AS": "AS",
    "ACO_EAS": "EAS",
    "ACO_AS_RANK": "ASrank",
    "ACO_MMAS": "MMAS",
    "ACO_ACS": "ACS",
    "SA": "SA"
}

df["algoritmo_label"] = df["algoritmo"].map(nombres_algoritmos).fillna(df["algoritmo"])


# ============================================================
# GRÁFICA DISTANCIA MEDIA VS TIEMPO MEDIO
# ============================================================

plt.figure(figsize=(7, 4.5))

for escenario in sorted(df["escenario"].unique()):
    df_esc = df[df["escenario"] == escenario]

    plt.scatter(
        df_esc["media_tiempo"],
        df_esc["media_distancia"],
        s=70,
        label=f"Escenario {escenario}"
    )

    for _, row in df_esc.iterrows():
        plt.annotate(
            row["algoritmo_label"],
            (row["media_tiempo"], row["media_distancia"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

plt.xlabel("Tiempo medio de ejecución (s)")
plt.ylabel("Distancia media (u)")
plt.title("Relación entre distancia media y tiempo medio")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig(path_salida, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figura guardada en: {path_salida}")

##############################################################
# JUNTAR DOS IMÁGENES:
##############################################################

# ============================================================
# RUTAS DE ENTRADA Y SALIDA
# ============================================================

path_img_a = Path("resultados_experimentos/figuras/fig_distancia_vs_tiempo.png")
path_img_b = Path("resultados_experimentos/figuras/ruta_as_rank_esc1_recortada.png")

output_dir = Path("resultados_experimentos/figuras")
output_dir.mkdir(parents=True, exist_ok=True)

path_salida = output_dir / "fig_resultados_ruta_combinada.png"


# ============================================================
# COMPROBACIONES
# ============================================================

if not path_img_a.exists():
    raise FileNotFoundError(f"No se ha encontrado la imagen: {path_img_a}")

if not path_img_b.exists():
    raise FileNotFoundError(f"No se ha encontrado la imagen: {path_img_b}")


# ============================================================
# LECTURA DE IMÁGENES
# ============================================================

img_a = mpimg.imread(path_img_a)
img_b = mpimg.imread(path_img_b)


# ============================================================
# CREACIÓN DE FIGURA COMBINADA
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4.2),
    gridspec_kw={
        "width_ratios": [1, 1],
        "wspace": 0.04
    }
)

# ------------------------------------------------------------
# Imagen (a)
# ------------------------------------------------------------

axes[0].imshow(img_a)
axes[0].axis("off")
axes[0].set_aspect("equal")

axes[0].text(
    0.5,
    -0.06,
    "(a) Distancia media frente a tiempo medio.",
    transform=axes[0].transAxes,
    ha="center",
    va="top",
    fontsize=10
)

# ------------------------------------------------------------
# Imagen (b)
# ------------------------------------------------------------

axes[1].imshow(img_b)
axes[1].axis("off")
axes[1].set_aspect("equal")

axes[1].text(
    0.5,
    -0.06,
    r"(b) Ruta $AS_{rank}$ sobre un escenario.",
    transform=axes[1].transAxes,
    ha="center",
    va="top",
    fontsize=10
)


# ============================================================
# AJUSTE Y GUARDADO
# ============================================================

plt.subplots_adjust(
    left=0.01,
    right=0.99,
    top=0.98,
    bottom=0.14,
    wspace=0.04
)

plt.savefig(
    path_salida,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()

print(f"Figura combinada guardada en: {path_salida}")