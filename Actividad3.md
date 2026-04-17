# Actividad 3 — Ciencia de Datos

**Curso:** QR.LSTI2309TEO — Universidad Tecmilenio
**Ponderación:** 6%
**Temas relacionados:** T9, T10 (Preparación de datos en Python, Procesamiento de datos en Python)

---

## Descripción

Esta actividad se centra en aplicar y profundizar los conocimientos y habilidades que adquiriste en los temas 9 al 12. A través de un conjunto de datos reales, relacionados con el rendimiento de equipos de béisbol, tendrás la oportunidad de practicar técnicas esenciales en el proceso de ciencia de datos, desde la adquisición y preparación de datos hasta la modelación, predicción y evaluación de resultados.

---

## Objetivo

Reforzar los conceptos de regresión lineal simple y limpieza de datos, utilizando datos reales de equipos de béisbol, para predecir el número de carreras (runs) basado en el número de bateos.

---

## Instrucciones

En esta actividad, crearás y evaluarás un modelo de regresión lineal simple con el objetivo de predecir el número de runs de los equipos de béisbol de acuerdo con el número de bateos que tienen.

### Parte 1: Preparación de los datos

1. **Obtención de los datos:** Guarda la base de datos en una variable. Los datos los encontrarás en la siguiente página: https://www.espn.com.mx/beisbol/mlb/estadisticas/jugador

2. **Limpieza y preparación de los datos:** Evalúa los datos recopilados en busca de valores faltantes o erróneos. Además, realiza la limpieza necesaria, la imputación de datos faltantes y la estandarización de los datos para asegurar su calidad y uniformidad.

### Parte 2: Modelado y evaluación

Continúa con el desarrollo del modelo de regresión lineal simple. Para esto, realiza lo siguiente:

1. **Análisis exploratorio:** Calcula la correlación de Pearson para determinar la relación entre el número de bateos y carreras. Interpreta este coeficiente para entender la fuerza y dirección de la relación.

2. **Construcción del modelo:** Identifica tu variable dependiente y tu variable independiente. Divide el conjunto de datos en dos grupos: uno para entrenamiento y otro para prueba.

3. **Entrenamiento y predicción:** Entrena tu modelo de regresión lineal simple con el conjunto de entrenamiento. Luego, utiliza este modelo para realizar predicciones sobre el conjunto de prueba.

4. **Evaluación:** Calcula el error de tus predicciones utilizando métricas adecuadas. Analiza estos errores para evaluar la precisión de tu modelo.

5. **Conclusión:** Reflexiona sobre los resultados obtenidos, discute la efectividad del modelo y su aplicabilidad en la toma de decisiones estratégicas basadas en el análisis de datos.

---

## Entregable(s)

> **⚠️ IMPORTANTE: La entrega se realiza a través del repositorio de GitHub**
>
> No se aceptan documentos en Word o PDF. Todo el trabajo debe estar subido a tu repositorio de GitHub.

### Estructura de entrega en GitHub:

```
Semana4/
├── Consolidado/
│   └── Semana4_Consolidado.md    # Documento consolidado semanal
├── Actividad3/
│   ├── Analisis.py                # Código Python del análisis
│   ├── Datos/                     # Dataset utilizado
│   └── Visualizaciones/           # Gráficas generadas
└── commits documentados
```

### Componentes de evaluación:

| Componente | Ponderación | Descripción |
|------------|-------------|-------------|
| **Actividad Evaluable** | **70% (4.2%)** | Modelo de regresión lineal simple completo |
| **Ejercicios Complementarios** | **10% (0.6%)** | Ejercicios de normalización, valores faltantes, outliers |
| **Actividades Prácticas Extra** | **10% (0.6%)** | Actividades 4.1 - 4.4 completadas |
| **Documentación Elaborada** | **10% (0.6%)** | Consolidado semanal organizado |

---

## Rúbrica de Evaluación

### 1. Actividad Evaluable (70% - 4.2%)

| Criterio | Descripción | Puntuación |
|----------|-------------|------------|
| **Obtención de datos** | Dataset cargado correctamente | 0-0.6% |
| **Limpieza de datos** | Manejo de valores faltantes y erróneos | 0-0.6% |
| **Análisis de correlación** | Cálculo e interpretación de Pearson | 0-0.6% |
| **División de datos** | Train/test split apropiado | 0-0.6% |
| **Modelo de regresión** | Implementación correcta | 0-0.6% |
| **Evaluación del modelo** | Métricas calculadas e interpretadas | 0-0.6% |
| **Conclusión** | Reflexión sobre resultados | 0-0.6% |

### 2. Ejercicios Complementarios (10% - 0.6%)

| Criterio | Descripción | Puntuación |
|----------|-------------|------------|
| Normalización y Estandarización | Resueltos correctamente | 0-0.2% |
| Valores Faltantes | Resueltos correctamente | 0-0.2% |
| Transformaciones | Resueltos correctamente | 0-0.2% |

### 3. Actividades Prácticas Extra (10% - 0.6%)

| Criterio | Descripción | Puntuación |
|----------|-------------|------------|
| Valores Faltantes | Completado | 0-0.15% |
| Imputación | Completado | 0-0.15% |
| Transformación de Datos | Completado | 0-0.15% |
| Pipeline | Completado | 0-0.15% |

### 4. Documentación Elaborada (10% - 0.6%)

| Criterio | Descripción | Puntuación |
|----------|-------------|------------|
| Estructura | Sigue la estructura propuesta | 0-0.15% |
| Completitud | Incluye todas las secciones | 0-0.15% |
| Calidad de contenido | Ejercicios bien resueltos | 0-0.15% |
| Reflexión | Reflexiones profundas | 0-0.15% |

---

## Calendario

- **Semana:** 4
- **Fecha de entrega:** Según calendario del curso
- **Fecha límite:** Domingo de la semana indicada

---

## Recursos

- [Tema 9: Preparación de datos](Tema9.md)
- [Tema 10: Procesamiento de datos](Tema10.md)
- [Ejercicios Complementarios Semana 4](Actividad%20Semana%204/Ejercicios%20Complementarios.md)
- [Actividades Prácticas Semana 4](Actividad%20Semana%204/Propuesta%20Actividades.md)
