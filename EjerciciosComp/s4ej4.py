import numpy as np
import pandas as pd


df = pd.DataFrame(
    {
        "A": [1, 2, np.nan, 4, 5],
        "B": [np.nan, 2, 3, 4, np.nan],
        "C": [1, 2, 3, 4, 5],
    }
)

faltantes = df.isnull()
conteo_faltantes = df.isnull().sum()
porcentaje_faltantes = (df.isnull().mean() * 100).round(2)
filas_con_faltantes = df[df.isnull().any(axis=1)]

print("DataFrame original:\n", df)
print("\nMatriz de faltantes (isnull):\n", faltantes)
print("\nConteo de faltantes por columna:\n", conteo_faltantes)
print("\nPorcentaje de faltantes por columna (%):\n", porcentaje_faltantes)
print("\nFilas con al menos un faltante:\n", filas_con_faltantes)
