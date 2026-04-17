import numpy as np
import pandas as pd
from scipy import stats


datos = np.array([1, 2, 3, 4, 5, 10, 20, 30], dtype=float)

log_natural = np.log(datos)
raiz = np.sqrt(datos)
boxcox, lambda_boxcox = stats.boxcox(datos)
bins = pd.cut(datos, bins=4, labels=["Bajo", "Medio", "Alto", "Muy Alto"])

print("Datos originales:", datos.tolist())
print("Logaritmo natural:", np.round(log_natural, 4).tolist())
print("Raiz cuadrada:", np.round(raiz, 4).tolist())
print("Box-Cox:", np.round(boxcox, 4).tolist())
print("Lambda Box-Cox:", round(lambda_boxcox, 4))
print("Discretizacion:", bins.astype(str).tolist())
