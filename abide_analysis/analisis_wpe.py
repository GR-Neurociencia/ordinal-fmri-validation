#!/usr/bin/env python3
"""
Análisis estadístico de wPE/PE sobre ABIDE:
- Asigna redes de Yeo a cada ROI AAL.
- Modelos mixtos (wPE ~ Network + covariables).
- Comparación wPE vs PE (tamaño del efecto).
- Correlación con ruido (std_raw).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression  # para correlación parcial

# -------------------------------
# 1. Cargar datos
# -------------------------------
import sys
if len(sys.argv) < 2:
    print("Uso: python analisis_wpe.py <archivo_resultados.csv>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])

# Verificar columnas necesarias
cols_needed = ['FILE_ID', 'SITE_ID', 'ROI', 'PE', 'wPE', 'std_raw', 'length',
               'DX_GROUP', 'AGE_AT_SCAN', 'SEX']
for c in cols_needed:
    if c not in df.columns:
        raise ValueError(f"Falta columna {c} en el CSV")

# -------------------------------
# 2. Mapeo AAL cortical -> redes de Yeo (7 redes + sin asignar)
# Basado en Yeo et al. 2011 y asignaciones estándar
# -------------------------------
# Diccionario: número de ROI AAL -> nombre de red
aal_to_yeo = {
    1: "Visual", 2: "Visual", 3: "Visual",
    4: "Visual", 5: "Limbic", 6: "Limbic",
    7: "Limbic", 8: "Limbic", 9: "Limbic",
    10: "Limbic", 11: "Limbic", 12: "Limbic",
    13: "Limbic", 14: "Limbic", 15: "Limbic",
    16: "Limbic", 17: "Visual", 18: "Visual",
    19: "Visual", 20: "Visual", 21: "Limbic",
    22: "Limbic", 23: "Limbic", 24: "Limbic",
    25: "Limbic", 26: "Limbic", 27: "Limbic",
    28: "Limbic", 29: "Limbic", 30: "Limbic",
    31: "Default", 32: "Default", 33: "Default",
    34: "Default", 35: "Default", 36: "Default",
    37: "Default", 38: "Default", 39: "Default",
    40: "Default", 41: "Default", 42: "Default",
    43: "Default", 44: "Default", 45: "Default",
    46: "Default", 47: "Limbic", 48: "Limbic",
    49: "Limbic", 50: "Limbic", 51: "Limbic",
    52: "Limbic", 53: "Limbic", 54: "Limbic",
    55: "Limbic", 56: "Limbic", 57: "Visual",
    58: "Visual", 59: "Visual", 60: "Visual",
    61: "Somatomotor", 62: "Somatomotor", 63: "Somatomotor",
    64: "Somatomotor", 65: "Somatomotor", 66: "Somatomotor",
    67: "Dorsal Attention", 68: "Dorsal Attention",
    69: "Dorsal Attention", 70: "Dorsal Attention",
    71: "Dorsal Attention", 72: "Dorsal Attention",
    73: "Dorsal Attention", 74: "Dorsal Attention",
    75: "Ventral Attention", 76: "Ventral Attention",
    77: "Ventral Attention", 78: "Ventral Attention",
    79: "Ventral Attention", 80: "Ventral Attention",
    81: "Fronto-parietal", 82: "Fronto-parietal",
    83: "Fronto-parietal", 84: "Fronto-parietal",
    85: "Fronto-parietal", 86: "Fronto-parietal",
    87: "Fronto-parietal", 88: "Fronto-parietal",
    89: "Default", 90: "Default"
}

df['Network'] = df['ROI'].map(aal_to_yeo).fillna('Unassigned')
# Conservar solo redes principales
df = df[df['Network'] != 'Unassigned']

# -------------------------------
# 3. Análisis descriptivo y gráficos básicos
# -------------------------------
# Orden de redes según jerarquía esperada
network_order = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention',
                 'Limbic', 'Fronto-parietal', 'Default']
df['Network'] = pd.Categorical(df['Network'], categories=network_order, ordered=True)

# Boxplot de wPE por red
plt.figure(figsize=(10, 6))
sns.boxplot(x='Network', y='wPE', data=df, order=network_order, palette='Set2')
plt.title('wPE across Yeo functional networks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('wpe_by_network.png')
plt.show()

# Boxplot similar para PE
plt.figure(figsize=(10, 6))
sns.boxplot(x='Network', y='PE', data=df, order=network_order, palette='Set2')
plt.title('PE across Yeo functional networks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pe_by_network.png')
plt.show()

# -------------------------------
# 4. Modelo lineal mixto (wPE ~ Network + covariables)
# -------------------------------
# Preparar variables
df['AGE'] = df['AGE_AT_SCAN']
df['SEX_f'] = df['SEX'].astype('category')
df['DX_GROUP_f'] = df['DX_GROUP'].astype('category')
df['SITE_ID_f'] = df['SITE_ID'].astype('category')
df['length_z'] = (df['length'] - df['length'].mean()) / df['length'].std()

# Modelo mixto: efecto fijo de red + covariables, intercepción aleatoria por sitio y sujeto anidado
print("\nAjustando modelo mixto (wPE ~ Network + AGE + SEX + DX + MEAN_FD + length)...")
model = mixedlm("wPE ~ Network + AGE + SEX_f + DX_GROUP_f + MEAN_FD + length_z",
                data=df, groups=df["SITE_ID"],
                re_formula="1", vc_formula={"FILE_ID": "0 + C(FILE_ID)"})
result = model.fit(reml=True, method='lbfgs', maxiter=200)
print(result.summary())

# Extraer coeficientes de red (comparación contra Visual, que es la categoría base)
# Por defecto, statsmodels toma la primera categoría como referencia.
# Aseguremos que 'Visual' sea la referencia
df['Network'] = pd.Categorical(df['Network'], categories=network_order, ordered=True)  # ya lo está
# Reajustar para que la referencia sea Visual (primera categoría)
# El modelo usará la primera categoría automáticamente.

# -------------------------------
# 5. Contraste wPE vs PE: tamaño del efecto entre redes extremas
# -------------------------------
# Seleccionar solo DMN y Visual
dmn_visual = df[df['Network'].isin(['Default', 'Visual'])]
# Calcular d de Cohen para wPE
from numpy import std, mean
def cohen_d(x, y):
    return (mean(x) - mean(y)) / np.sqrt((std(x, ddof=1)**2 + std(y, ddof=1)**2) / 2)

wpe_dmn = dmn_visual[dmn_visual['Network'] == 'Default']['wPE']
wpe_vis = dmn_visual[dmn_visual['Network'] == 'Visual']['wPE']
d_wpe = cohen_d(wpe_dmn, wpe_vis)

pe_dmn = dmn_visual[dmn_visual['Network'] == 'Default']['PE']
pe_vis = dmn_visual[dmn_visual['Network'] == 'Visual']['PE']
d_pe = cohen_d(pe_dmn, pe_vis)

print(f"\nTamaño del efecto (d de Cohen) entre DMN y Visual:")
print(f"  wPE: {d_wpe:.3f}")
print(f"  PE : {d_pe:.3f}")
print("Hipótesis H2: wPE debería mostrar un d mayor que PE clásica.")

# -------------------------------
# 6. Relación con proxy de ruido (std_raw)
# -------------------------------
# Correlación parcial (controlando SITE_ID y AGE)
# Usaremos residuos de regresiones lineales para controlar
def partial_corr(x, y, z):
    """Correlación parcial entre x e y después de regresar las variables en z (columnas)."""
    if isinstance(z, pd.Series):
        z = z.to_frame()
    # Regresión de x sobre z
    reg_x = LinearRegression().fit(z, x)
    res_x = x - reg_x.predict(z)
    # Regresión de y sobre z
    reg_y = LinearRegression().fit(z, y)
    res_y = y - reg_y.predict(z)
    return stats.pearsonr(res_x, res_y)[0]

# Preparar variables de control
control = df[['AGE', 'SITE_ID']]
control = pd.get_dummies(control, columns=['SITE_ID'], drop_first=True)

r_wpe = partial_corr(df['wPE'], df['std_raw'], control)
r_pe = partial_corr(df['PE'], df['std_raw'], control)
print(f"\nCorrelación parcial (controlando edad y sitio) con std_raw:")
print(f"  wPE vs std_raw: r = {r_wpe:.3f}")
print(f"  PE  vs std_raw: r = {r_pe:.3f}")
print("Hipótesis H3: wPE debe mostrar menor correlación que PE.")

# Guardar resultados complementarios en un CSV
resumen = pd.DataFrame({
    'Metrica': ['wPE', 'PE'],
    'd_DMN_Visual': [d_wpe, d_pe],
    'r_std_raw': [r_wpe, r_pe]
})
resumen.to_csv('resumen_hipotesis.csv', index=False)

print("\n✅ Análisis terminado. Revisa los gráficos y 'resumen_hipotesis.csv'.")