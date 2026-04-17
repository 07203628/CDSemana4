import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from simulacionMLB import (
    cargar_precios_spotrac,
    cargar_stats_2025,
    completar_stats_avanzadas_aprox,
    enriquecer_con_precios,
)

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "Datos"
VIZ_DIR = BASE_DIR / "Visualizaciones"


def cargar_y_preparar_dataset() -> pd.DataFrame:
    stats, fuente = cargar_stats_2025()
    precios = cargar_precios_spotrac(2025)
    stats = enriquecer_con_precios(stats, precios)
    stats, _ = completar_stats_avanzadas_aprox(stats, fuente)

    # Variable independiente y dependiente adaptadas al simulador.
    # X: OPS (produccion ofensiva) | y: WAR (valor total del jugador)
    columnas_modelo = ["Name", "Team", "ops", "WAR"]
    df = stats[columnas_modelo].copy()
    df["ops"] = pd.to_numeric(df["ops"], errors="coerce")
    df["WAR"] = pd.to_numeric(df["WAR"], errors="coerce")

    # Limpieza basica: eliminar faltantes y valores no validos.
    df = df.dropna(subset=["ops", "WAR"]).copy()
    df = df[(df["ops"] > 0) & (df["WAR"] >= -5)].copy()
    df = df.reset_index(drop=True)
    return df


def entrenar_modelo(df: pd.DataFrame) -> tuple[LinearRegression, StandardScaler, dict, pd.DataFrame]:
    x = df[["ops"]].values
    y = df["WAR"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    modelo = LinearRegression()
    modelo.fit(x_train_scaled, y_train)
    y_pred = modelo.predict(x_test_scaled)

    pearson = float(df["ops"].corr(df["WAR"], method="pearson"))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metricas = {
        "variable_independiente": "ops",
        "variable_dependiente": "WAR",
        "pearson": round(pearson, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "n_total": int(len(df)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "coeficiente": round(float(modelo.coef_[0]), 4),
        "intercepto": round(float(modelo.intercept_), 4),
    }

    resultados_test = pd.DataFrame(
        {
            "ops": x_test.flatten(),
            "war_real": y_test,
            "war_pred": y_pred,
            "error_abs": np.abs(y_test - y_pred),
        }
    ).sort_values("error_abs", ascending=False)

    return modelo, scaler, metricas, resultados_test


def guardar_resultados(df_modelo: pd.DataFrame, metricas: dict, resultados_test: pd.DataFrame) -> None:
    DATOS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    df_modelo.to_csv(DATOS_DIR / "dataset_modelado.csv", index=False)
    resultados_test.to_csv(DATOS_DIR / "predicciones_test.csv", index=False)

    with open(DATOS_DIR / "metricas_modelo.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    # Visualizacion 1: relacion OPS vs WAR
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df_modelo["ops"], df_modelo["WAR"], alpha=0.75)
    ax.set_title("Relación entre OPS y WAR")
    ax.set_xlabel("OPS")
    ax.set_ylabel("WAR")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "scatter_ops_war.png", dpi=140)
    plt.close(fig)

    # Visualizacion 2: real vs predicho en test
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(resultados_test["war_real"], resultados_test["war_pred"], alpha=0.8)
    min_v = min(resultados_test["war_real"].min(), resultados_test["war_pred"].min())
    max_v = max(resultados_test["war_real"].max(), resultados_test["war_pred"].max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax.set_title("WAR real vs WAR predicho (test)")
    ax.set_xlabel("WAR real")
    ax.set_ylabel("WAR predicho")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "real_vs_pred_test.png", dpi=140)
    plt.close(fig)


def imprimir_conclusion(metricas: dict) -> None:
    pearson = metricas["pearson"]
    r2 = metricas["r2"]

    if pearson >= 0.7:
        fuerza = "fuerte"
    elif pearson >= 0.4:
        fuerza = "moderada"
    else:
        fuerza = "débil"

    print("\n=== CONCLUSION DEL ANALISIS ===")
    print(
        f"La correlacion de Pearson entre OPS y WAR es {pearson}, "
        f"lo que indica una relacion lineal {fuerza} y positiva."
    )
    print(
        f"El modelo obtuvo R2={r2}, MAE={metricas['mae']} y RMSE={metricas['rmse']}. "
        "Esto permite estimar WAR de forma razonable para apoyo en decisiones de scouting, "
        "aunque no reemplaza evaluaciones completas con variables de defensa y contexto."
    )


def main() -> None:
    print("Cargando y preparando dataset desde el simulador...")
    df = cargar_y_preparar_dataset()

    print("Entrenando regresion lineal simple...")
    _, _, metricas, resultados_test = entrenar_modelo(df)

    print("Guardando dataset, metricas y visualizaciones...")
    guardar_resultados(df, metricas, resultados_test)

    print("\n=== METRICAS ===")
    for k, v in metricas.items():
        print(f"{k}: {v}")

    imprimir_conclusion(metricas)


if __name__ == "__main__":
    main()
