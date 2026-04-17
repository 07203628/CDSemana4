import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


df = pd.DataFrame(
    {
        "A": [1, 2, np.nan, 4, 5],
        "B": [np.nan, 2, 3, 4, np.nan],
        "C": [1, 2, 3, 4, 5],
    }
)

imputadores = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "most_frequent": SimpleImputer(strategy="most_frequent"),
    "constant": SimpleImputer(strategy="constant", fill_value=-1),
}

print("DataFrame original:\n", df)
for nombre, imputador in imputadores.items():
    resultado = imputador.fit_transform(df)
    print(f"\nEstrategia {nombre}:\n", pd.DataFrame(resultado, columns=df.columns))
