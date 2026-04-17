# Semana 4: Preparacion y Procesamiento de Datos

## 1. Ejercicios Complementarios

### Normalizacion y estandarizacion
- Ejercicio 1: Normalizacion Min-Max manual y en Python.
  Archivo: [s4ej1.py](../EjerciciosComp/s4ej1.py)
- Ejercicio 2: Estandarizacion Z-score (media, desviacion, verificacion de Z).
  Archivo: [s4ej2.py](../EjerciciosComp/s4ej2.py)
- Ejercicio 3: Comparacion MinMaxScaler vs StandardScaler.
  Archivo: [s4ej3.py](../EjerciciosComp/s4ej3.py)

### Valores faltantes e imputacion
- Ejercicio 4: Identificacion de faltantes, conteo y porcentaje por columna.
  Archivo: [s4ej4.py](../EjerciciosComp/s4ej4.py)
- Ejercicio 5: Estrategias de imputacion (drop, media, mediana, ffill, bfill).
  Archivo: [s4ej5.py](../EjerciciosComp/s4ej5.py)
- Ejercicio 6: Imputacion avanzada con SimpleImputer.
  Archivo: [s4ej6.py](../EjerciciosComp/s4ej6.py)

### Outliers
- Ejercicio 7: Deteccion de outliers con metodo IQR.
  Archivo: [s4ej7.py](../EjerciciosComp/s4ej7.py)
- Ejercicio 8: Deteccion de outliers con metodo Z-score.
  Archivo: [s4ej8.py](../EjerciciosComp/s4ej8.py)
- Ejercicio 9: Manejo de outliers (eliminacion, capping, log, Box-Cox).
  Archivo: [s4ej9.py](../EjerciciosComp/s4ej9.py)

### Transformaciones y feature engineering
- Ejercicio 10: Codificacion de variables categoricas (Label + One-Hot).
  Archivo: [s4ej10.py](../EjerciciosComp/s4ej10.py)
- Ejercicio 11: Transformaciones numericas (log, raiz, Box-Cox, binning).
  Archivo: [s4ej11.py](../EjerciciosComp/s4ej11.py)
- Ejercicio 12: Feature engineering (ratio, diferencia, binarios, polynomial, datetime).
  Archivo: [s4ej12.py](../EjerciciosComp/s4ej12.py)

### Escalamiento y pipeline
- Ejercicio 13: Comparacion de escaladores (MinMax, Standard, Robust, MaxAbs).
  Archivo: [s4ej13.py](../EjerciciosComp/s4ej13.py)
- Ejercicio 14: Pipeline de preprocesamiento con ColumnTransformer.
  Archivo: [s4ej14.py](../EjerciciosComp/s4ej14.py)

### Investigacion
- Ejercicio 15: Mejores practicas (preparacion de datos, data leakage, train/test).
  Archivo: [s4ej15.md](../EjerciciosComp/s4ej15.md)
- Ejercicio 16: Tecnicas avanzadas (SMOTE, KNN Imputer, Target Encoding).
  Archivo: [s4ej16.md](../EjerciciosComp/s4ej16.md)

## 2. Actividades Practicas

### Actividad 4.1: Identificacion de valores faltantes
- Entregable implementado: deteccion de nulos, conteo, completitud y manejo de faltantes con multiples tecnicas.
- Archivos:
  - [README.md](../Actividades/Ejercicio4.1/README.md)
  - [ejercicio4_1.py](../Actividades/Ejercicio4.1/ejercicio4_1.py)

### Actividad 4.2: Imputacion de datos
- Entregable implementado: comparacion de media, mediana, moda, forward fill y backward fill.
- Archivos:
  - [README.md](../Actividades/Ejercicio4.2/README.md)
  - [ejercicio4_2.py](../Actividades/Ejercicio4.2/ejercicio4_2.py)

### Actividad 4.3: Transformacion de datos
- Entregable implementado: Min-Max, Z-score, One-Hot y variable derivada.
- Archivos:
  - [README.md](../Actividades/Ejercicio4.3/README.md)
  - [ejercicio4_3.py](../Actividades/Ejercicio4.3/ejercicio4_3.py)

### Actividad 4.4: Pipeline de procesamiento
- Entregable implementado: pipeline con imputacion, escalamiento y transformacion categorica usando sklearn.
- Archivos:
  - [README.md](../Actividades/Ejercicio4.4/README.md)
  - [ejercicio4_4.py](../Actividades/Ejercicio4.4/ejercicio4_4.py)

## 3. Actividad Evaluable (Actividad 3)

Se desarrollo un modelo de regresion lineal simple adaptado al simulador MLB del proyecto.

### Adaptacion de variables
- Variable independiente (X): `ops`
- Variable dependiente (y): `WAR`

### Flujo realizado
- Carga y preparacion de datos desde el algoritmo del simulador.
- Limpieza y transformacion de variables.
- Correlacion de Pearson.
- Division train/test.
- Entrenamiento de `LinearRegression`.
- Prediccion y evaluacion con metricas.

### Resultados principales
- Pearson: 0.8504
- MAE: 0.6332
- RMSE: 0.7696
- R2: 0.5508

### Evidencias
- Codigo: [Analisis.py](../Actividad3/Analisis.py)
- Reporte: [Reporte_Actividad3.md](../Actividad3/Reporte_Actividad3.md)
- Datos:
  - [dataset_modelado.csv](../Actividad3/Datos/dataset_modelado.csv)
  - [predicciones_test.csv](../Actividad3/Datos/predicciones_test.csv)
  - [metricas_modelo.json](../Actividad3/Datos/metricas_modelo.json)
- Visualizaciones:
  - [scatter_ops_war.png](../Actividad3/Visualizaciones/scatter_ops_war.png)
  - [real_vs_pred_test.png](../Actividad3/Visualizaciones/real_vs_pred_test.png)

## 4. Resumen de Aprendizaje
- La calidad de datos impacta directamente la calidad del modelo.
- La imputacion y estandarizacion deben realizarse de forma controlada para evitar sesgos.
- Una regresion lineal simple es interpretable y util como linea base.
- Pearson, MAE, RMSE y R2 permiten evaluar desde distintos angulos el desempeno del modelo.

## 5. Dudas o Preguntas
- Como varia el desempeno del modelo si se agregan mas variables explicativas (ej. hits, homeRuns, atBats)?
- Conviene probar regularizacion (Ridge/Lasso) para mejorar generalizacion?

## 6. Referencias
- Scikit-learn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Pandas Missing Data: https://pandas.pydata.org/docs/user_guide/missing_data.html
- Scikit-learn Linear Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
