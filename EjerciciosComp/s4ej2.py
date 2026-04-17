from math import sqrt

datos = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

mu = sum(datos) / len(datos)
sigma = sqrt(sum((x - mu) ** 2 for x in datos) / len(datos))  # poblacional
z = [(x - mu) / sigma for x in datos]

media_z = sum(z) / len(z)
std_z = sqrt(sum((x - media_z) ** 2 for x in z) / len(z))

print("Media (mu):", mu)
print("Desviación estándar (sigma):", sigma)
print("Z-scores:", z)
print("Media de Z:", media_z)
print("Std de Z:", std_z)