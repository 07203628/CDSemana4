import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline() -> Pipeline:
    numeric_features = ["edad", "ingreso"]
    categorical_features = ["ciudad"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])


def main() -> None:
    df = pd.DataFrame(
        {
            "edad": [18, None, 35, 40, 28],
            "ingreso": [450, 520, None, 1100, 650],
            "ciudad": ["A", "B", None, "C", "B"],
        }
    )

    print("Datos originales:\n", df, "\n")

    pipeline = build_pipeline()
    transformed = pipeline.fit_transform(df)

    print("Pipeline ejecutado correctamente.")
    print("Forma de la salida:", transformed.shape)


if __name__ == "__main__":
    main()
