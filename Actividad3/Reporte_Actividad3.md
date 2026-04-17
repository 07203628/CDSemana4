# Actividad 3 - Modelo de Regresion Lineal Simple

## Contexto
Se desarrollo un modelo de regresion lineal simple utilizando los datos del simulador MLB de este proyecto.

## Adaptacion de variables
La actividad original sugiere predecir carreras a partir de bateos.
Para adaptar el ejercicio al algoritmo del programa, se uso:
- Variable independiente (X): `ops`
- Variable dependiente (y): `WAR`

Esta seleccion se alinea con el simulador, ya que `WAR` es una metrica central para evaluar el valor de los jugadores en decisiones de roster.

## 1) Obtencion de datos
Los datos se cargaron desde el flujo del simulador en [Actividad3/simulacionMLB.py](simulacionMLB.py), reutilizando:
- `cargar_stats_2025`
- `cargar_precios_spotrac`
- `enriquecer_con_precios`
- `completar_stats_avanzadas_aprox`

## 2) Limpieza y preparacion
Se aplico:
- Conversion numerica de `ops` y `WAR`.
- Eliminacion de filas con valores faltantes en variables de modelado.
- Filtro de valores no validos (`ops > 0` y `WAR >= -5`).
- Estandarizacion de `X` con `StandardScaler` (ajuste en train y transform en test).

## 3) Analisis exploratorio
Se calculo correlacion de Pearson entre `ops` y `WAR`.

Resultado:
- Pearson = **0.8504**

Interpretacion:
- Relacion lineal **fuerte y positiva**.

## 4) Construccion del modelo
- Modelo: `LinearRegression` de scikit-learn.
- Split: 80% entrenamiento, 20% prueba.
- `random_state=42` para reproducibilidad.

## 5) Entrenamiento y prediccion
El modelo se entreno con el conjunto de entrenamiento y se realizaron predicciones sobre el conjunto de prueba.

## 6) Evaluacion
Metricas obtenidas:
- MAE: **0.6332**
- RMSE: **0.7696**
- R2: **0.5508**

Interpretacion:
- El modelo explica alrededor del 55% de la variabilidad de `WAR` con una sola variable (`ops`).
- Es util como linea base para decisiones iniciales, pero puede mejorar con mas variables.

## 7) Conclusiones
- `ops` es un predictor relevante para estimar `WAR` en este contexto.
- Un modelo lineal simple ofrece una aproximacion interpretable y util para apoyo en scouting.
- Para aumentar precision, se recomienda incluir variables adicionales como defensa, posicion y volumen de juego.

## Evidencias generadas
### Codigo
- [Actividad3/Analisis.py](Analisis.py)

### Datos
- [Actividad3/Datos/dataset_modelado.csv](Datos/dataset_modelado.csv)
- [Actividad3/Datos/predicciones_test.csv](Datos/predicciones_test.csv)
- [Actividad3/Datos/metricas_modelo.json](Datos/metricas_modelo.json)

### Visualizaciones
- [Actividad3/Visualizaciones/scatter_ops_war.png](Visualizaciones/scatter_ops_war.png)
- [Actividad3/Visualizaciones/real_vs_pred_test.png](Visualizaciones/real_vs_pred_test.png)
