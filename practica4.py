import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el archivo CSV

df = pd.read_csv('archivo_modificado.csv')

# Configuración de estilo de seaborn
sns.set(style="whitegrid")

# Convierte las columnas en valores numéricos y maneja valores faltantes
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)



# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_casa'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_casa'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_casa'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_casa'] - df['Goles_de_casa'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_casa'] - df['Goles_de_casa'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_casa'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_casa'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')

# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_visita'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_visita'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_visita'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_visita'] - df['Goles_de_visita'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_visita'] - df['Goles_de_visita'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_visita'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_visita'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')

# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_casa_medio_tiempo'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_casa_medio_tiempo'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_casa_medio_tiempo'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa_medio_tiempo']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa_medio_tiempo']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_casa_medio_tiempo'] - df['Goles_de_casa_medio_tiempo'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_casa_medio_tiempo'] - df['Goles_de_casa_medio_tiempo'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_casa_medio_tiempo'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_casa_medio_tiempo'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')

# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_visita_medio_tiempo'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_visita_medio_tiempo'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_visita_medio_tiempo'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita_medio_tiempo']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita_medio_tiempo']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_visita_medio_tiempo'] - df['Goles_de_visita_medio_tiempo'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_visita_medio_tiempo'] - df['Goles_de_visita_medio_tiempo'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_visita_medio_tiempo'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_visita_medio_tiempo'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')

# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_casa_tiempo_completo'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_casa_tiempo_completo'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_casa_tiempo_completo'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa_tiempo_completo']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa_tiempo_completo']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_casa_tiempo_completo'] - df['Goles_de_casa_tiempo_completo'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_casa_tiempo_completo'] - df['Goles_de_casa_tiempo_completo'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_casa_tiempo_completo'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_casa_tiempo_completo'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')

# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_visita_tiempo_completo'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_visita_tiempo_completo'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_visita_tiempo_completo'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita_tiempo_completo']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita_tiempo_completo']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_visita_tiempo_completo'] - df['Goles_de_visita_tiempo_completo'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_visita_tiempo_completo'] - df['Goles_de_visita_tiempo_completo'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_visita_tiempo_completo'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_visita_tiempo_completo'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')

# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_casa_tiempo_extra'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_casa_tiempo_extra'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_casa_tiempo_extra'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa_tiempo_extra']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa_tiempo_extra']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_casa_tiempo_extra'] - df['Goles_de_casa_tiempo_extra'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_casa_tiempo_extra'] - df['Goles_de_casa_tiempo_extra'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_casa_tiempo_extra'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_casa_tiempo_extra'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')

# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_visita_tiempo_extra'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_visita_tiempo_extra'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_visita_tiempo_extra'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita_tiempo_extra']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita_tiempo_extra']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_visita_tiempo_extra'] - df['Goles_de_visita_tiempo_extra'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_visita_tiempo_extra'] - df['Goles_de_visita_tiempo_extra'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_visita_tiempo_extra'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_visita_tiempo_extra'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')

# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_casa_penalizacion'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_casa_penalizacion'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_casa_penalizacion'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa_penalizacion']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_casa_penalizacion']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_casa_penalizacion'] - df['Goles_de_casa_penalizacion'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_casa_penalizacion'] - df['Goles_de_casa_penalizacion'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_casa_penalizacion'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_casa_penalizacion'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')

# Gráficos
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))

# Mínimo
sns.boxplot(x=df['Goles_de_visita_penalizacion'], ax=axs[0, 0])
axs[0, 0].set_title('Mínimo')

# Máximo
sns.boxplot(x=df['Goles_de_visita_penalizacion'], ax=axs[0, 1])
axs[0, 1].set_title('Máximo')

# Conteo
sns.histplot(df['Goles_de_visita_penalizacion'], ax=axs[0, 2], kde=False)
axs[0, 2].set_title('Conteo')

# Sumatoria
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita_penalizacion']), ax=axs[1, 0])
axs[1, 0].set_title('Sumatoria')

# Media
sns.lineplot(x=df.index, y=np.cumsum(df['Goles_de_visita_penalizacion']) / np.arange(1, len(df) + 1), ax=axs[1, 1])
axs[1, 1].set_title('Media')

# Varianza
sns.lineplot(x=df.index, y=np.cumsum((df['Goles_de_visita_penalizacion'] - df['Goles_de_visita_penalizacion'].mean())**2) / np.arange(1, len(df) + 1), ax=axs[1, 2])
axs[1, 2].set_title('Varianza')

# Desviación estándar
sns.lineplot(x=df.index, y=np.sqrt(np.cumsum((df['Goles_de_visita_penalizacion'] - df['Goles_de_visita_penalizacion'].mean())**2) / np.arange(1, len(df) + 1)), ax=axs[2, 0])
axs[2, 0].set_title('Desviación Estándar')

# Asimetría
sns.histplot(df['Goles_de_visita_penalizacion'], ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Asimetría')

# Kurtosis
sns.histplot(df['Goles_de_visita_penalizacion'], ax=axs[2, 2], kde=True)
axs[2, 2].set_title('Kurtosis')






plt.tight_layout()
plt.show()
