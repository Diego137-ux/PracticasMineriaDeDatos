import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd

df = pd.read_csv('archivo_modificado.csv')

modelo = ols('Goles_de_casa ~ Equipo_de_casa', data=df).fit()

# Realizar el análisis de varianza (ANOVA)
anova_resultado = sm.stats.anova_lm(modelo)

# Imprimir la tabla ANOVA
print("Tabla ANOVA:")
print(anova_resultado)