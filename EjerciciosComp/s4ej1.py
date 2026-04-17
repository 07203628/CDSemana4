datos = [10, 20, 30, 40, 50]

x_min = min(datos)
x_max = max(datos)

normalizados = [(x - x_min) / (x_max - x_min) for x in datos]

print("Datos originales:", datos)
print("Datos normalizados:", normalizados)
print(all(0 <= x <= 1 for x in normalizados))