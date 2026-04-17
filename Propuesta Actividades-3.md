# Actividades Prácticas - Semana 4

## Preparación para: Actividad 3

**Temas cubiertos:** Preparación de los datos en Python, Procesamiento de datos en Python

---

## IMPORTANTE: Ejercicios Complementarios

Antes de comenzar con las actividades prácticas, es **fundamental** completar los ejercicios del archivo [`Ejercicios Complementarios-3.md`](Ejercicios%20Complementarios-3.md).

### ¿Por qué realizar estos ejercicios?

Los ejercicios complementarios te proporcionan la base matemática y de procesamiento necesaria para:

1. **Comprender normalización y estandarización** de datos
2. **Manejar valores faltantes y outliers** de manera efectiva
3. **Transformar variables** para análisis posteriores
4. **Diseñar pipelines** de procesamiento eficientes

> **Justificación:** Estos conocimientos son prerrequisitos básicos que facilitan la preparación y procesamiento de datos. Sin esta base, será más difícil completar las actividades prácticas de manera efectiva.

**Tiempo estimado:** 1-2 horas

---

## Objetivo

Estas actividades prácticas te prepararán para completar la Actividad 3, enfocándose en la preparación y procesamiento de datos.

---

## GIT: Recordatorio de Comandos

```bash
# Ver estado
git status

# Agregar cambios
git add .

# Crear commit
git commit -m "Mensaje descriptivo"

# Subir al repositorio
git push origin main

# Actualizar repositorio local
git pull origin main
```

---

## Actividades Propuestas

### Actividad 4.1: Identificación de Valores Faltantes

**Descripción:** Aprende a detectar y manejar valores faltantes.

**Instrucciones:**
1. Crea un DataFrame con valores nulos
2. Practica diferentes métodos de detección:
   - isnull() / notnull()
   - sum() para contar nulos por columna
   - info() para ver completitud
3. Investiga diferentes estrategias de manejo
4. Aplica al menos 3 técnicas diferentes

**Carpeta de entrega:** `Actividades/Ejercicio4.1/`

**Git:**
```bash
git add Actividades/Ejercicio4.1/
git commit -m "Semana4: Actividad 4.1 - Valores faltantes"
git push
```

---

### Actividad 4.2: Imputación de Datos

**Descripción:** Practica técnicas de imputación de datos.

**Instrucciones:**
1. Usa un dataset con valores nulos
2. Aplica las siguientes técnicas de imputación:
   - Imputación por media
   - Imputación por mediana
   - Imputación por moda
   - Imputación hacia adelante/atrás
3. Compara los resultados
4. Justifica cuándo usar cada método

**Carpeta de entrega:** `Actividades/Ejercicio4.2/`

**Git:**
```bash
git add Actividades/Ejercicio4.2/
git commit -m "Semana4: Actividad 4.2 - Imputación de datos"
git push
```

---

### Actividad 4.3: Transformación de Datos

**Descripción:** Practica transformaciones comunes de datos.

**Instrucciones:**
1. Con un dataset, realiza las siguientes transformaciones:
   - Normalización de datos (Min-Max)
   - Estandarización (Z-score)
   - Codificación de variables categóricas (One-Hot Encoding)
   - Creación de variables derivadas
2. Explica cada transformación y cuándo usarla

**Carpeta de entrega:** `Actividades/Ejercicio4.3/`

**Git:**
```bash
git add Actividades/Ejercicio4.3/
git commit -m "Semana4: Actividad 4.3 - Transformación de datos"
git push
```

---

### Actividad 4.4: Pipeline de Procesamiento

**Descripción:** Crea un pipeline completo de procesamiento.

**Instrucciones:**
1. Diseña un pipeline que incluya:
   - Carga de datos
   - Manejo de valores nulos
   - Transformación de variables
   - Selección de características
2. Usa sklearn Pipeline si es posible
3. Documenta cada paso

**Carpeta de entrega:** `Actividades/Ejercicio4.4/`

**Git:**
```bash
git add Actividades/Ejercicio4.4/
git commit -m "Semana4: Actividad 4.4 - Pipeline de procesamiento"
git push
```

---

## Actualizar Documentación Semanal

Crea/actualiza el archivo `Documentacion/Semana4.md`:

```markdown
# Semana 4: Preparación y Procesamiento de Datos

## Fecha: [DD/MM/AAAA]

## Actividades Completadas
- [ ] Actividad 4.1: Valores Faltantes
- [ ] Actividad 4.2: Imputación de Datos
- [ ] Actividad 4.3: Transformación de Datos
- [ ] Actividad 4.4: Pipeline de Procesamiento

## Aprendido
- [Puntos clave]

## Dudas o Bloqueos
- [Si tienes alguna duda]

## Commits Realizados
- Listar los commits de esta semana
```

**Git:**
```bash
git add Documentacion/Semana4.md
git commit -m "Semana4: Documentación actualizada"
git push
```

---

## Actualizar README

```bash
git add README.md
git commit -m "Update README: Semana 4 completada"
git push
```

---

## Recursos Adicionales

- Sklearn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Pandas Missing Data: https://pandas.pydata.org/docs/user_guide/missing_data.html

---

## Recordatorio

La Actividad 3 tiene una ponderación del 6% y cubre los Temas 9 y 10 del temario.

---

## 📋 Consolidado Semanal: Semana 4

Al finalizar todas las actividades de esta semana, debes crear un archivo consolidado llamado `Semana4_Consolidado.md` que contenga:

### Estructura del consolidado:

```markdown
# Semana 4: Preparación y Procesamiento de Datos

## 1. Ejercicios Complementarios

### Ejercicio 1: [Nombre]
**Solución:**
[Tu respuesta/solución]

### Ejercicio 2: [Nombre]
**Solución:**
[Tu respuesta/solución]

## 2. Actividades Prácticas

### Actividad 4.1: [Nombre]
**Entregable:** [Descripción]

### Actividad 4.2: [Nombre]
**Entregable:** [Descripción]

## 3. Resumen de Aprendizaje

- [Aprendizaje 1]
- [Aprendizaje 2]
- [Aprendizaje 3]

## 4. Dudas o Preguntas

- [Pregunta 1]
- [Pregunta 2]

## 5. Referencias

- [Referencia 1]
- [Referencia 2]
```

### Instrucciones de entrega:
1. Completa todos los ejercicios y actividades
2. Organiza todo en el archivo consolidado
3. Guarda en `Semana4_Consolidado.md`
4. Realiza el commit correspondiente:
   ```bash
   git add Semana4_Consolidado.md
   git commit -m "Semana4: Consolidado completo"
   git push
   ```

---

## ✅ Actividad Evaluable: Actividad 3

**Fecha de entrega:** [Según calendario del curso]

**Ponderación:** 6%

**Temas evaluados:** T9 y T10

**Instrucciones:**
1. Completa todos los ejercicios complementarios
2. Completa todas las actividades prácticas
3. Verifica tu consolidado semanal
4. Entrega según las indicaciones del profesor

**Criterios de evaluación:**
- [ ] Completó los ejercicios complementarios
- [ ] Completó todas las actividades prácticas
- [ ] El consolidado está organizado y completo
- [ ] Commits realizados correctamente en Git
