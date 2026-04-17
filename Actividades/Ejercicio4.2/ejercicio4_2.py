import numpy as np
import pandas as pd


def main() -> None:
    # Dataset de ejemplo con faltantes
    df = pd.DataFrame(
        {
            "fecha": pd.date_range("2026-04-01", periods=8, freq="D"),
            "temperatura": [22.5, np.nan, 23.1, 24.0, np.nan, 25.2, np.nan, 24.8],
            "humedad": [60, 62, np.nan, 58, 57, np.nan, 55, 54],
            "ciudad": ["A", "A", None, "A", "B", "B", None, "B"],
        }
    )

    print("Dataset original:\n", df, "\n")

    # 1) Media (solo numericas)
    df_media = df.copy()
    cols_num = ["temperatura", "humedad"]
    df_media[cols_num] = df_media[cols_num].fillna(df_media[cols_num].mean())
    print("Imputacion por media:\n", df_media, "\n")

    # 2) Mediana (solo numericas)
    df_mediana = df.copy()
    df_mediana[cols_num] = df_mediana[cols_num].fillna(df_mediana[cols_num].median())
    print("Imputacion por mediana:\n", df_mediana, "\n")

    # 3) Moda (incluye categoricas)
    df_moda = df.copy()
    for col in df_moda.columns:
        if df_moda[col].isnull().any():
            df_moda[col] = df_moda[col].fillna(df_moda[col].mode(dropna=True).iloc[0])
    print("Imputacion por moda:\n", df_moda, "\n")

    # 4) Forward fill
    df_ffill = df.copy().ffill()
    print("Imputacion forward fill:\n", df_ffill, "\n")

    # 5) Backward fill
    df_bfill = df.copy().bfill()
    print("Imputacion backward fill:\n", df_bfill)


if __name__ == "__main__":
    main()
