import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


df = pd.DataFrame(
    {
        "edad": [21, 35, 42, 28, 50],
        "ingreso": [1200, 2500, 3200, 1800, 4000],
        "ciudad": ["Lima", "Cusco", "Lima", "Arequipa", "Cusco"],
        "segmento": ["A", "B", "A", "C", "B"],
    }
)

numeric_features = ["edad", "ingreso"]
categorical_features = ["ciudad", "segmento"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
datos_transformados = pipeline.fit_transform(df)

print("DataFrame original:\n", df)
print("\nShape transformado:", datos_transformados.shape)
print("Primeras filas transformadas:\n", np.round(datos_transformados[:3], 4))
