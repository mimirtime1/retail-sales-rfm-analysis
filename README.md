# Retail Sales & RFM Customer Segmentation

Análisis de ventas retail en Python: limpieza de datos, estadística descriptiva, estacionalidad y segmentación de clientes con **RFM** (Recency, Frequency, Monetary).

## Dataset
Descargar de Kaggle y guardar en `data/`:
- *Superstore* (`Sample - Superstore.csv`) **o**
- *E-Commerce Data / Online Retail* (`data.csv`)

## Cómo ejecutar
```bash
pip install -r requirements.txt
python retail_rfm_analysis.py --input "data/Sample - Superstore.csv" --output outputs
```

## Qué hace
1. **Limpieza:** duplicados, clientes sin ID, fechas inválidas, devoluciones y montos no positivos.
2. **Estadística descriptiva:** media, desviación, coeficiente de variación y asimetría; histogramas de ventas (original y log).
3. **Estacionalidad:** serie mensual, media móvil e índice estacional por mes.
4. **RFM:** puntajes 1–5 por quintiles, segmentos de clientes y principio de Pareto.

## Resultados (carpeta `outputs/`)
Gráficos `fig_*.png`, `rfm_clientes.csv`, `rfm_resumen_segmentos.csv`, `ventas_mensuales.csv`.

## Principales hallazgos
*(Completar con tus números reales tras ejecutar: % de ventas del top 20 % de clientes, meses de mayor estacionalidad, segmento que más aporta.)*

## Herramientas
Python · pandas · NumPy · Matplotlib
