import numpy as np


datos = np.array([10, 12, 14, 15, 16, 18, 20, 22, 25, 100], dtype=float)

q1 = np.percentile(datos, 25)
q3 = np.percentile(datos, 75)
iqr = q3 - q1
lim_inf = q1 - 1.5 * iqr
lim_sup = q3 + 1.5 * iqr
outliers = datos[(datos < lim_inf) | (datos > lim_sup)]

print("Datos:", datos.tolist())
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Limite inferior:", lim_inf)
print("Limite superior:", lim_sup)
print("Outliers:", outliers.tolist())
