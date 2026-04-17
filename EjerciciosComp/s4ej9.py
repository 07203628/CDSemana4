import numpy as np
from scipy import stats


datos = np.array([10, 12, 14, 15, 16, 18, 20, 22, 25, 100], dtype=float)

q1 = np.percentile(datos, 25)
q3 = np.percentile(datos, 75)
iqr = q3 - q1
lim_inf = q1 - 1.5 * iqr
lim_sup = q3 + 1.5 * iqr

datos_sin_outliers = datos[(datos >= lim_inf) & (datos <= lim_sup)]
datos_capping = np.clip(datos, lim_inf, lim_sup)
datos_log = np.log1p(datos)
datos_boxcox, lambda_boxcox = stats.boxcox(datos)

print("Datos originales:", datos.tolist())
print("Sin outliers:", datos_sin_outliers.tolist())
print("Capping:", np.round(datos_capping, 4).tolist())
print("Log1p:", np.round(datos_log, 4).tolist())
print("Box-Cox:", np.round(datos_boxcox, 4).tolist())
print("Lambda Box-Cox:", round(lambda_boxcox, 4))
