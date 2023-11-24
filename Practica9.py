import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Cargar el DataFrame desde el archivo CSV
df = pd.read_csv('archivo_modificado.csv')

# Supongamos que tienes dos columnas, 'Goles_de_casa' y 'Goles_de_visita'
X = df['Goles_de_casa']
y = df['Goles_de_visita']

# Añadir una constante a la variable independiente (intercepto)
X = sm.add_constant(X)

# Ajustar el modelo de regresión lineal
modelo_regresion = sm.OLS(y, X).fit()

# Imprimir un resumen del modelo
print(modelo_regresion.summary())

# Hacer predicciones para nuevos datos
nuevos_datos = pd.DataFrame({'Goles_de_casa': [6, 7, 8]})
nuevos_datos = sm.add_constant(nuevos_datos)  # Añadir una constante

# Hacer predicciones
predicciones = modelo_regresion.predict(nuevos_datos)

# Imprimir las predicciones
print("Predicciones:")
print(predicciones)

# Graficar los resultados
plt.scatter(df['Goles_de_casa'], df['Goles_de_visita'], color='black', label='Datos originales')
plt.plot(nuevos_datos['Goles_de_casa'], predicciones, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Predicciones')
plt.xlabel('Goles de Casa')
plt.ylabel('Goles de Visita')
plt.title('Regresión Lineal y Predicciones')
plt.legend()
plt.show()