import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def main() -> None:
    # Dataset de ejemplo
    df = pd.DataFrame(
        {
            "edad": [18, 22, 35, 40, 28],
            "ingreso": [450, 520, 900, 1100, 650],
            "ciudad": ["A", "B", "A", "C", "B"],
        }
    )

    print("DataFrame original:\n", df, "\n")

    # Min-Max
    minmax = MinMaxScaler()
    df[["edad_minmax", "ingreso_minmax"]] = minmax.fit_transform(df[["edad", "ingreso"]])

    # Z-score
    zscore = StandardScaler()
    df[["edad_z", "ingreso_z"]] = zscore.fit_transform(df[["edad", "ingreso"]])

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=["ciudad"], prefix="ciudad")

    # Variable derivada
    df["ingreso_por_edad"] = (df["ingreso"] / df["edad"]).round(2)

    print("DataFrame transformado:\n", df)


if __name__ == "__main__":
    main()
