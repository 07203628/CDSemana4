import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler


data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)

scalers = {
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler(),
}

print("Data original:\n", data)
for nombre, scaler in scalers.items():
    escalado = scaler.fit_transform(data)
    print(f"\n{nombre}:\n", np.round(escalado, 4))

print("\nCuando usar cada uno:")
print("- MinMaxScaler: acota valores a un rango fijo.")
print("- StandardScaler: centra en media 0 y varianza 1.")
print("- RobustScaler: mejor con outliers.")
print("- MaxAbsScaler: util en datos dispersos.")
