"""
retail-sales-rfm-analysis
=========================
Análisis de ventas retail: limpieza, estadística descriptiva, estacionalidad
y segmentación de clientes con RFM (Recency, Frequency, Monetary).

Datasets compatibles (descargar de Kaggle y guardar en la carpeta data/):
  - Superstore:   "Sample - Superstore.csv"
  - E-Commerce:   "data.csv" (Online Retail, columnas InvoiceNo, StockCode, ...)

Uso:
    python retail_rfm_analysis.py --input data/"Sample - Superstore.csv"
    python retail_rfm_analysis.py --input data/data.csv --output outputs
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # permite guardar gráficos sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# --------------------------------------------------------------------------
# 1. Carga y estandarización de columnas
# --------------------------------------------------------------------------
COLUMN_MAP = {
    # Superstore
    "Order ID": "order_id",
    "Order Date": "date",
    "Customer ID": "customer_id",
    "Sales": "sales",
    "Quantity": "quantity",
    "Category": "category",
    "Region": "region",
    "Profit": "profit",
    # Online Retail / E-Commerce Data
    "InvoiceNo": "order_id",
    "InvoiceDate": "date",
    "CustomerID": "customer_id",
    "Description": "category",
    "Country": "region",
}


def load_data(path: Path) -> pd.DataFrame:
    """Lee el CSV probando utf-8 y luego latin-1 (Superstore usa latin-1)."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    print(f"[carga] {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    # Online Retail no trae 'Sales': se calcula como cantidad * precio unitario
    if "sales" not in df.columns and {"quantity", "UnitPrice"} <= set(df.columns):
        df["sales"] = df["quantity"] * df["UnitPrice"]
    required = {"order_id", "date", "customer_id", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias: {missing}")
    return df


# --------------------------------------------------------------------------
# 2. Limpieza
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    report = {}

    report["duplicados"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    report["sin_customer_id"] = int(df["customer_id"].isna().sum())
    df = df.dropna(subset=["customer_id"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    report["fechas_invalidas"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    # Devoluciones / cancelaciones / montos no positivos
    # En Online Retail las cancelaciones son facturas tipo "C536379" (C + dígitos).
    # Se usa regex para NO confundirlas con IDs de Superstore como "CA-2016-152156".
    df["order_id"] = df["order_id"].astype(str)
    mask_ok = (df["sales"] > 0) & ~df["order_id"].str.match(r"^C\d")
    if "quantity" in df.columns:
        mask_ok &= df["quantity"] > 0
    report["ventas_no_positivas"] = int((~mask_ok).sum())
    df = df[mask_ok].copy()

    df["customer_id"] = df["customer_id"].astype(str)
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    print("[limpieza] resumen:", report)
    print(f"[limpieza] {n0:,} -> {len(df):,} filas")
    return df


# --------------------------------------------------------------------------
# 3. Estadística descriptiva
# --------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame, out: Path) -> None:
    num_cols = [c for c in ["sales", "quantity", "profit"] if c in df.columns]
    desc = df[num_cols].describe().T
    desc["cv"] = desc["std"] / desc["mean"]  # coeficiente de variación
    desc["asimetria"] = df[num_cols].skew()
    print("\n[descriptiva]\n", desc.round(2))
    desc.round(3).to_csv(out / "estadistica_descriptiva.csv")

    # Distribución de ventas (original y log) -> suele ser muy asimétrica
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["sales"], bins=60, color="#4C72B0")
    axes[0].set_title("Distribución de ventas por transacción")
    axes[0].set_xlabel("Ventas")
    axes[1].hist(np.log1p(df["sales"]), bins=60, color="#55A868")
    axes[1].set_title("Distribución de log(1 + ventas)")
    axes[1].set_xlabel("log(1 + ventas)")
    fig.tight_layout()
    fig.savefig(out / "fig_01_distribucion_ventas.png", dpi=150)
    plt.close(fig)

    if "category" in df.columns and df["category"].nunique() <= 20:
        by_cat = df.groupby("category")["sales"].sum().sort_values()
        ax = by_cat.plot(kind="barh", figsize=(7, 4), color="#4C72B0")
        ax.set_title("Ventas totales por categoría")
        ax.set_xlabel("Ventas")
        plt.tight_layout()
        plt.savefig(out / "fig_02_ventas_por_categoria.png", dpi=150)
        plt.close()


# --------------------------------------------------------------------------
# 4. Estacionalidad
# --------------------------------------------------------------------------
def seasonality(df: pd.DataFrame, out: Path) -> None:
    monthly = df.groupby("year_month")["sales"].sum()
    monthly.to_csv(out / "ventas_mensuales.csv")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(monthly.index, monthly.values, marker="o", label="Ventas mensuales")
    ax.plot(
        monthly.index,
        monthly.rolling(3, center=True).mean(),
        linestyle="--",
        label="Media móvil 3 meses",
    )
    ax.set_title("Ventas mensuales y tendencia")
    ax.set_ylabel("Ventas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_03_serie_mensual.png", dpi=150)
    plt.close(fig)

    # Índice estacional: promedio del mes / promedio general (1.0 = promedio)
    tmp = monthly.to_frame("sales")
    tmp["mes"] = tmp.index.month
    seasonal_index = tmp.groupby("mes")["sales"].mean() / tmp["sales"].mean()
    print("\n[estacionalidad] índice por mes (1.0 = promedio):")
    print(seasonal_index.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(seasonal_index.index, seasonal_index.values, color="#C44E52")
    ax.axhline(1, color="black", linewidth=1)
    ax.set_xticks(range(1, 13))
    ax.set_title("Índice de estacionalidad por mes")
    ax.set_xlabel("Mes")
    fig.tight_layout()
    fig.savefig(out / "fig_04_indice_estacional.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Análisis RFM
# --------------------------------------------------------------------------
def segment_customer(row) -> str:
    r, f = row["R"], row["F"]
    if r >= 4 and f >= 4:
        return "Campeones"
    if r >= 3 and f >= 3:
        return "Clientes leales"
    if r >= 4 and f <= 2:
        return "Nuevos / prometedores"
    if r <= 2 and f >= 4:
        return "En riesgo (eran buenos)"
    if r <= 2 and f <= 2:
        return "Perdidos"
    return "Necesitan atención"


def rfm_analysis(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    snapshot = df["date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("date", lambda x: (snapshot - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("sales", "sum"),
    )

    # Puntajes 1-5 (5 = mejor). rank(method="first") evita errores por empates.
    rfm["R"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["segmento"] = rfm.apply(segment_customer, axis=1)

    summary = (
        rfm.groupby("segmento")
        .agg(
            clientes=("monetary", "size"),
            recencia_media=("recency", "mean"),
            frecuencia_media=("frequency", "mean"),
            gasto_total=("monetary", "sum"),
            gasto_medio=("monetary", "mean"),
        )
        .sort_values("gasto_total", ascending=False)
    )
    summary["% clientes"] = 100 * summary["clientes"] / summary["clientes"].sum()
    summary["% ventas"] = 100 * summary["gasto_total"] / summary["gasto_total"].sum()
    print("\n[RFM] resumen por segmento:\n", summary.round(1))

    rfm.to_csv(out / "rfm_clientes.csv")
    summary.round(2).to_csv(out / "rfm_resumen_segmentos.csv")

    # Gráfico: clientes vs ventas por segmento
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["% clientes"], width=0.4, label="% de clientes")
    ax.bar(x + 0.2, summary["% ventas"], width=0.4, label="% de ventas")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=25, ha="right")
    ax.set_title("Peso de cada segmento RFM: clientes vs ventas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_05_segmentos_rfm.png", dpi=150)
    plt.close(fig)

    # Principio de Pareto: ¿qué % de las ventas aporta el 20 % de clientes?
    sorted_m = rfm["monetary"].sort_values(ascending=False)
    top20 = sorted_m.head(int(np.ceil(0.2 * len(sorted_m)))).sum() / sorted_m.sum()
    print(f"\n[Pareto] el 20% de clientes top concentra {top20:.1%} de las ventas")
    return rfm


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis de ventas retail y RFM")
    parser.add_argument("--input", required=True, help="Ruta al CSV")
    parser.add_argument("--output", default="outputs", help="Carpeta de resultados")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.input))
    df = standardize(df)
    df = clean_data(df)
    descriptive_stats(df, out)
    seasonality(df, out)
    rfm_analysis(df, out)
    print(f"\nListo. Resultados guardados en: {out.resolve()}")


if __name__ == "__main__":
    main()
