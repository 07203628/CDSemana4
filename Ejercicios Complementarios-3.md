# Ejercicios Complementarios - Semana 4

## Temas Cubiertos
- **T9**: Preparación de los datos en Python
- **T10**: Procesamiento de datos en Python

## Prerrequisitos Recomendados
- **Matemáticas**: Normalización, estandarización, operaciones básicas
- **Estadística**: Valores atípicos, datos faltantes, distribuciones
- **Programación**: Manipulación de DataFrames, funciones lambda

---

## Ejercicios de Normalización y Estandarización

### Ejercicio 1: Normalización Min-Max
La fórmula de normalización Min-Max es:

```
X_normalized = (X - X_min) / (X_max - X_min)
```

Dados los datos: [10, 20, 30, 40, 50]

1. Aplicar Min-Max normalization manualmente
2. Verificar que el resultado esté entre 0 y 1
3. Implementar en Python

### Ejercicio 2: Estandarización (Z-Score)
La fórmula de estandarización es:

```
Z = (X - μ) / σ
```

Donde μ = media y σ = desviación estándar

Dados los datos: [2, 4, 4, 4, 5, 5, 7, 9]

1. Calcular la media
2. Calcular la desviación estándar
3. Estandarizar cada valor
4. Verificar que la media sea ~0 y std sea ~1

### Ejercicio 3: Comparación de Técnicas
```python
import numpy as np

datos = np.array([100, 200, 300, 400, 500]).reshape(-1, 1)

# Aplicar:
# 1. MinMaxScaler de sklearn
# 2. StandardScaler de sklearn
# Comparar resultados
```

---

## Ejercicios de Manejo de Valores Faltantes

### Ejercicio 4: Identificación de Valores Faltantes
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, np.nan],
    'C': [1, 2, 3, 4, 5]
})

# Ejercicios:
# 1. Identificar valores faltantes con isnull()
# 2. Contar valores faltantes por columna
# 3. Calcular porcentaje de valores faltantes
# 4. Mostrar solo filas con valores faltantes
```

### Ejercicio 5: Estrategias de Imputación
```python
# Para el mismo DataFrame, aplicar:
# 1. Eliminar filas con valores faltantes
# 2. Eliminar columnas con valores faltantes
# 3. Imputar con la media
# 4. Imputar con la mediana
# 5. Imputar con forward fill
# 6. Imputar con backward fill
```

### Ejercicio 6: Imputación Avanzada
```python
# Usar sklearn.impute.SimpleImputer
from sklearn.impute import SimpleImputer

# Probar diferentes estrategias:
# - mean
# - median
# - most_frequent
# - constant
```

---

## Ejercicios de Detección y Manejo de Outliers

### Ejercicio 7: Método IQR (Rango Intercuartil)
```python
datos = [10, 12, 14, 15, 16, 18, 20, 22, 25, 100]

# Calcular:
# 1. Q1 (percentil 25)
# 2. Q3 (percentil 75)
# 3. IQR = Q3 - Q1
# 4. Límite inferior = Q1 - 1.5 * IQR
# 5. Límite superior = Q3 + 1.5 * IQR
# 6. Identificar outliers
```

### Ejercicio 8: Método Z-Score
```python
from scipy import stats
import numpy as np

datos = np.array([10, 12, 14, 15, 16, 18, 20, 22, 25, 100])

# Calcular Z-scores y encontrar valores donde |Z| > 3
z_scores = stats.zscore(datos)
outliers = np.where(np.abs(z_scores) > 3)
```

### Ejercicio 9: Manejo de Outliers
```python
# Opciones para manejar outliers:
# 1. Eliminar outliers
# 2.替换为边界值 (capping)
# 3. Transformación logarítmica
# 4. Transformación Box-Cox
# Aplicar cada método
```

---

## Ejercicios de Transformación de Variables

### Ejercicio 10: Codificación de Variables Categóricas
```python
import pandas as pd

df = pd.DataFrame({
    'color': ['rojo', 'azul', 'verde', 'rojo', 'verde'],
    'talla': ['S', 'M', 'L', 'S', 'M']
})

# Aplicar:
# 1. Label Encoding
# 2. One-Hot Encoding con get_dummies
# 3. One-Hot Encoding con sklearn
```

### Ejercicio 11: Transformaciones Numéricas
```python
import numpy as np

datos = [1, 2, 3, 4, 5, 10, 20, 30]

# Aplicar:
# 1. Logaritmo natural
# 2. Raíz cuadrada
# 3. Transformación Box-Cox
# 4. Discretización (binned)
```

### Ejercicio 12: Feature Engineering
```python
# Crear nuevas features:
# 1. Ratio entre dos columnas
# 2. Diferencia entre columnas
# 3. Agregar indicadores binarios
# 4. Polynomial features
# 5. DateTime features
```

---

## Ejercicios de Escalamiento de Datos

### Ejercicio 13: Comparar Escaladores
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Aplicar cada escalador y comparar resultados
# ¿Cuándo usar cada uno?
```

### Ejercicio 14: Pipeline de Preprocesamiento
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

# Crear un pipeline completo:
# 1. Seleccionar columnas numéricas y categóricas
# 2. Aplicar transformaciones apropiadas
# 3. Combinar en un pipeline
```

---

## Ejercicios de Investigación

### Ejercicio 15: Mejores Prácticas
Investigar:
1. ¿Por qué es importante la preparación de datos?
2. ¿Qué es data leakage y cómo evitarlo?
3. ¿Cuál es la diferencia entre datos de entrenamiento y prueba?

### Ejercicio 16: Técnicas Avanzadas
Investigar:
1. ¿Qué es SMOTE para datos desbalanceados?
2. ¿Qué es la imputación por K-Nearest Neighbors?
3. ¿Qué es Target Encoding?

---

## Recursos Adicionales

### Documentación
- Scikit-learn - Preprocessing
- Pandas - Working with missing data
- Seaborn - Distribution plots

### Práctica
- Kaggle - Feature Engineering course
- DataCamp - Preprocessing in Python

---

## Próxima Semana
En la Semana 5 cubriremos:
- **T11**: Análisis preliminar de los datos en Python
- **T12**: Regresión lineal simple en Python

**Prerrequisitos para próxima semana:**
- Correlación y covarianza
- Coeficiente de Pearson
- p-valores e intervalos de confianza
- Supuestos de regresión, R², residuos
