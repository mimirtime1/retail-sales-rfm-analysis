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
- **Ventas muy asimétricas:** sobre 9,994 líneas de pedido, la venta media es 229.86 pero la mediana es solo 54.49 (coeficiente de variación 2.71, asimetría 12.97). Pocas compras grandes elevan el promedio, por eso también se analizó en escala logarítmica.
- **Estacionalidad marcada:** septiembre (1.61), noviembre (1.84) y diciembre (1.70) superan ampliamente el promedio mensual, mientras que febrero (0.31) y enero (0.50) son los meses más bajos.
- **Segmentación RFM de 793 clientes:** los *Campeones* son el 21.1 % de los clientes y generan el 29.4 % de las ventas. Junto con los *Clientes leales*, 4 de cada 10 clientes aportan más de la mitad de las ventas (52.7 %).
- **Oportunidad de reactivación:** el segmento *En riesgo* es solo el 10.2 % de los clientes, pero aporta el 15.3 % de las ventas y tiene el mayor gasto medio por cliente (4,351.7), con 204 días promedio sin comprar.
- **Concentración (Pareto):** el 20 % de clientes top concentra el 48.1 % de las ventas, una concentración menor a la regla clásica 80/20.

## Recomendaciones
- Lanzar campañas de retención dirigidas al segmento *En riesgo*.
- Reforzar inventario y promociones antes del pico de septiembre a diciembre.

## Herramientas
Python · pandas · NumPy · Matplotlib
