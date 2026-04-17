import numpy as np
import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "edad": [23, 35, np.nan, 29, 41],
            "ingreso": [1200, np.nan, 1800, 1600, np.nan],
            "ciudad": ["Lima", "Cusco", "Lima", None, "Arequipa"],
        }
    )

    print("DataFrame original:\n", df, "\n")

    # 1) Deteccion de faltantes
    print("isnull():\n", df.isnull(), "\n")
    print("notnull():\n", df.notnull(), "\n")

    # 2) Conteo de faltantes por columna
    print("Nulos por columna:\n", df.isnull().sum(), "\n")

    # 3) Completitud
    print("Info del DataFrame:")
    df.info()
    print()

    # 4) Mostrar filas con al menos un faltante
    filas_con_nulos = df[df.isnull().any(axis=1)]
    print("Filas con valores faltantes:\n", filas_con_nulos, "\n")

    # 5) Tecnicas de manejo
    print("Eliminar filas con nulos:\n", df.dropna(), "\n")
    print("Eliminar columnas con nulos:\n", df.dropna(axis=1), "\n")

    # Imputaciones numericas
    df_media = df.copy()
    df_media[["edad", "ingreso"]] = df_media[["edad", "ingreso"]].fillna(
        df_media[["edad", "ingreso"]].mean(numeric_only=True)
    )
    print("Imputacion por media (numericas):\n", df_media, "\n")

    df_mediana = df.copy()
    df_mediana[["edad", "ingreso"]] = df_mediana[["edad", "ingreso"]].fillna(
        df_mediana[["edad", "ingreso"]].median(numeric_only=True)
    )
    print("Imputacion por mediana (numericas):\n", df_mediana, "\n")

    df_moda = df.copy()
    for col in df_moda.columns:
        df_moda[col] = df_moda[col].fillna(df_moda[col].mode(dropna=True).iloc[0])
    print("Imputacion por moda (todas las columnas):\n", df_moda)


if __name__ == "__main__":
    main()
