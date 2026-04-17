import numpy as np
from scipy import stats


datos = np.array([10, 12, 14, 15, 16, 18, 20, 22, 25, 100], dtype=float)

z_scores = stats.zscore(datos)
outliers_idx = np.where(np.abs(z_scores) > 3)[0]
outliers_vals = datos[outliers_idx]

print("Datos:", datos.tolist())
print("Z-scores:", np.round(z_scores, 4).tolist())
print("Indices con |Z| > 3:", outliers_idx.tolist())
print("Valores outlier:", outliers_vals.tolist())
