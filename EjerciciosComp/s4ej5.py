import numpy as np
import pandas as pd


df = pd.DataFrame(
    {
        "A": [1, 2, np.nan, 4, 5],
        "B": [np.nan, 2, 3, 4, np.nan],
        "C": [1, 2, 3, 4, 5],
    }
)

df_sin_filas_faltantes = df.dropna()
df_sin_columnas_faltantes = df.dropna(axis=1)
df_imputado_media = df.fillna(df.mean(numeric_only=True))
df_imputado_mediana = df.fillna(df.median(numeric_only=True))
df_imputado_ffill = df.ffill()
df_imputado_bfill = df.bfill()

print("DataFrame original:\n", df)
print("\nEliminar filas con faltantes:\n", df_sin_filas_faltantes)
print("\nEliminar columnas con faltantes:\n", df_sin_columnas_faltantes)
print("\nImputar con media:\n", df_imputado_media)
print("\nImputar con mediana:\n", df_imputado_mediana)
print("\nImputar con forward fill:\n", df_imputado_ffill)
print("\nImputar con backward fill:\n", df_imputado_bfill)
