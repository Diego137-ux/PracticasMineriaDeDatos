import pandas as pd

# Lee el archivo CSV
df = pd.read_csv('archivo.csv')

# Lista de nombres de las columnas a eliminar
columnas_a_eliminar = ['id', 'timezone', 'venue_id']

# Elimina las columnas
df = df.drop(columnas_a_eliminar, axis=1)

# Guarda el DataFrame modificado en un nuevo archivo CSV
df.to_csv('archivo_modificado.csv', index=False)
 
 #--------------------------------------------------------


# Lee el archivo CSV
df = pd.read_csv('archivo_modificado.csv')

# Diccionario de mapeo para cambiar los nombres de las columnas
nombres_nuevos = {
    'referee': 'Arbitro',
    'date': 'Fecha',
    'venue_name': 'Estadio',
    'venue_city': 'Ciudad',
    'season': 'Temporada',
    'round': 'Ronda',
    'home_team': 'Equipo_de_casa',
    'away_team': 'Equipo_visita',
    'home_win':'Victoria_de_casa',
    'away_win':'Victoria_de_visita',
    'home_goals':'Goles_de_casa',
    'away_goals':'Goles_de_visita',
    'home_goals_half_time':'Goles_de_casa_medio_tiempo',
    'away_goals_half_time':'Goles_de_visita_medio_tiempo',
    'home_goals_fulltime':'Goles_de_casa_tiempo_completo',
    'away_goals_fulltime':'Goles_de_visita_tiempo_completo',
    'home_goals_extra_time':'Goles_de_casa_tiempo_extra',
    'away_goals_extratime':'Goles_de_visita_tiempo_extra',
    'home_goals_penalty':'Goles_de_casa_penalizacion',
    'away_goals_penalty':'Goles_de_visita_penalizacion',

}

# Cambia los nombres de las columnas
df = df.rename(columns=nombres_nuevos)

# Guarda el DataFrame modificado en un nuevo archivo CSV
df.to_csv('archivo_modificado.csv', index=False)