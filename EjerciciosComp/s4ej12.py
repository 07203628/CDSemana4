import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


df = pd.DataFrame(
    {
        "ventas": [100, 120, 140, 160],
        "costos": [60, 80, 90, 100],
        "visitas": [1000, 1500, 1800, 2200],
        "fecha": pd.to_datetime(["2026-01-05", "2026-02-10", "2026-03-15", "2026-04-20"]),
    }
)

df["ratio_ventas_costos"] = df["ventas"] / df["costos"]
df["diferencia_ventas_costos"] = df["ventas"] - df["costos"]
df["alto_trafico"] = (df["visitas"] > 1600).astype(int)

df["anio"] = df["fecha"].dt.year
df["mes"] = df["fecha"].dt.month
df["dia_semana"] = df["fecha"].dt.dayofweek

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_data = poly.fit_transform(df[["ventas", "costos"]])
poly_cols = poly.get_feature_names_out(["ventas", "costos"])
df_poly = pd.DataFrame(poly_data, columns=poly_cols)

print("DataFrame con nuevas features:\n", df)
print("\nPolynomial features:\n", df_poly)
