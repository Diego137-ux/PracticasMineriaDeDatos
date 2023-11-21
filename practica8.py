import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Ejemplo de carga de datos (reemplázalo con tus propios datos)
# Asegúrate de tener columnas relevantes para la agrupación.
df = pd.read_csv('archivo_modificado.csv')
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
# Seleccionar características relevantes
features = df[[
    'Goles_de_casa',
    'Goles_de_visita',
    'Goles_de_casa_medio_tiempo',
    'Goles_de_visita_medio_tiempo',
    'Goles_de_casa_tiempo_completo',
    'Goles_de_visita_tiempo_completo',
    'Goles_de_casa_tiempo_extra',
    'Goles_de_visita_tiempo_extra',
    'Goles_de_casa_penalizacion',
    'Goles_de_visita_penalizacion',
]]

# Escalar características para que tengan una escala similar
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Determinar el número de clusters (ajusta según tus necesidades)
k = 3

# Aplicar k-Means
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualizar los resultados (puedes ajustar según el número de características)
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=df['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, marker='X', c='red')
plt.title('Agrupación de Datos con k-Means')
plt.xlabel('Caracteristica1 (escalada)')
plt.ylabel('Caracteristica2 (escalada)')
plt.show()
