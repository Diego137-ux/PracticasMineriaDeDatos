import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pandas as pd


# Cargar el DataFrame desde el archivo CSV
df = pd.read_csv('archivo_modificado.csv')

# Supongamos que tienes dos columnas, 'variable_independiente' y 'variable_dependiente'
X = df['Goles_de_casa']
y = df['Goles_de_visita']

# Añadir una constante a la variable independiente (intercepto)
X = sm.add_constant(X)

# Ajustar el modelo de regresión lineal
modelo_regresion = sm.OLS(y, X).fit()

# Imprimir un resumen del modelo
print(modelo_regresion.summary())

# Hacer predicciones
predicciones = modelo_regresion.predict(X)

# Graficar los resultados
plt.scatter(df['Goles_de_casa'], df['Goles_de_visita'], color='black')
plt.plot(df['Goles_de_casa'], predicciones, color='blue', linewidth=3)
plt.xlabel('Goles_de_casa')
plt.ylabel('Goles_de_visita')
plt.title('Regresión Lineal')
plt.show()






