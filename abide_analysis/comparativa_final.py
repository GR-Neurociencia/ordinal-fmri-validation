#!/usr/bin/env python3
"""
Comparativa final de los tres pipelines ABIDE:
filt_global, filt_noglobal, nofilt_noglobal.
Genera figuras y tabla resumen para el artículo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# -------------------------------
# 1. Cargar los tres CSV
# -------------------------------
archivos = {
    'filt_global': 'resultados_wpe.csv',
    'filt_noglobal': 'resultados_wpe_filt_noglobal.csv',
    'nofilt_noglobal': 'resultados_wpe_nofilt_noglobal.csv'
}

dfs = {}
for nombre, archivo in archivos.items():
    if not Path(archivo).exists():
        print(f"⚠️  No se encuentra {archivo}, se omite.")
        continue
    df = pd.read_csv(archivo)
    # Mapeo AAL a redes (mismo que antes)
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
    df = df[df['Network'] != 'Unassigned']
    network_order = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention',
                     'Limbic', 'Fronto-parietal', 'Default']
    df['Network'] = pd.Categorical(df['Network'], categories=network_order, ordered=True)
    dfs[nombre] = df

if not dfs:
    print("No se encontraron CSV. Terminando.")
    exit()

# -------------------------------
# 2. Calcular d de Cohen para cada pipeline y métrica
# -------------------------------
resultados_d = []
for nombre, df in dfs.items():
    dmn = df[df['Network'] == 'Default']
    visual = df[df['Network'] == 'Visual']
    for metrica in ['wPE', 'PE']:
        x = dmn[metrica]
        y = visual[metrica]
        d = (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2) / 2)
        resultados_d.append({
            'Pipeline': nombre,
            'Métrica': metrica,
            'd_Cohen': d
        })

df_d = pd.DataFrame(resultados_d)

# Gráfico de barras comparativo
plt.figure(figsize=(10, 6))
bar_width = 0.35
pipelines = list(dfs.keys())
x = np.arange(len(pipelines))

for i, metrica in enumerate(['wPE', 'PE']):
    vals = df_d[df_d['Métrica'] == metrica].set_index('Pipeline').reindex(pipelines)['d_Cohen'].values
    plt.bar(x + i*bar_width, vals, bar_width, label=metrica, alpha=0.85)

plt.xlabel('Pipeline')
plt.ylabel('d de Cohen (DMN vs Visual)')
plt.title('Discriminación de complejidad cortical según pipeline y métrica')
plt.xticks(x + bar_width/2, pipelines, rotation=15)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparativa_d_Cohen.png', dpi=300)
plt.show()
print("✅ Figura 'comparativa_d_Cohen.png' guardada.")

# -------------------------------
# 3. Boxplots por red en tres paneles
# -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, (nombre, df) in zip(axes, dfs.items()):
    sns.boxplot(x='Network', y='wPE', data=df, order=network_order,
                hue='Network', palette='Set2', legend=False, ax=ax)
    ax.set_title(nombre)
    ax.tick_params(axis='x', rotation=45)
    if ax != axes[0]:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('wPE')

plt.suptitle('wPE across functional networks (three preprocessing pipelines)', fontsize=14)
plt.tight_layout()
plt.savefig('boxplots_wpe_triple.png', dpi=300)
plt.show()
print("✅ Figura 'boxplots_wpe_triple.png' guardada.")

# -------------------------------
# 4. Tabla resumen CSV
# -------------------------------
resumen = []
for nombre, df in dfs.items():
    dmn = df[df['Network'] == 'Default']
    visual = df[df['Network'] == 'Visual']
    d_wpe = (np.mean(dmn['wPE']) - np.mean(visual['wPE'])) / np.sqrt((np.std(dmn['wPE'], ddof=1)**2 + np.std(visual['wPE'], ddof=1)**2) / 2)
    d_pe = (np.mean(dmn['PE']) - np.mean(visual['PE'])) / np.sqrt((np.std(dmn['PE'], ddof=1)**2 + np.std(visual['PE'], ddof=1)**2) / 2)
    resumen.append({
        'Pipeline': nombre,
        'Sujetos': df['FILE_ID'].nunique(),
        'Total_ROIs': len(df),
        'wPE_DMN_media': dmn['wPE'].mean(),
        'wPE_Visual_media': visual['wPE'].mean(),
        'd_Cohen_wPE': d_wpe,
        'PE_DMN_media': dmn['PE'].mean(),
        'PE_Visual_media': visual['PE'].mean(),
        'd_Cohen_PE': d_pe,
        'Corr_wPE_std_raw': stats.pearsonr(df['wPE'], df['std_raw'])[0],
        'Corr_PE_std_raw': stats.pearsonr(df['PE'], df['std_raw'])[0]
    })

pd.DataFrame(resumen).to_csv('resumen_final.csv', index=False, float_format='%.5f')
print("✅ Tabla 'resumen_final.csv' guardada.")