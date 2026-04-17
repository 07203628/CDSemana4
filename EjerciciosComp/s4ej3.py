import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


datos = np.array([100, 200, 300, 400, 500]).reshape(-1, 1)

# 1) MinMaxScaler
minmax_scaler = MinMaxScaler()
datos_minmax = minmax_scaler.fit_transform(datos)

# 2) StandardScaler
standard_scaler = StandardScaler()
datos_standard = standard_scaler.fit_transform(datos)

print("Datos originales:", datos.ravel().tolist())
print("MinMaxScaler:", datos_minmax.ravel().tolist())
print("Rango MinMax -> min:", float(datos_minmax.min()), "max:", float(datos_minmax.max()))
print("StandardScaler:", np.round(datos_standard.ravel(), 4).tolist())
print("Media StandardScaler:", float(np.round(datos_standard.mean(), 6)))
print("Std StandardScaler:", float(np.round(datos_standard.std(ddof=0), 6)))
