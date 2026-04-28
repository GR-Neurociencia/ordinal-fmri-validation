#!/usr/bin/env python3
"""
Barrido de m=4,5,6 sobre un subconjunto de ABIDE para ver si wPE
recupera el gradiente entre redes (Visual vs Default).
Usa los mismos archivos .1D y la clase ComplexityMetrics del src/.
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ajustar path para importar src (como antes)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.metrics import ComplexityMetrics
from src import TAU_FIXED

# ----------------- Configuración -----------------
INPUT_DIR = "abide_timeseries~/Outputs/cpac/filt_global/rois_aal"
TAU = TAU_FIXED            # 15
MIN_VOLS = 200
CORTICAL_IDXS = range(0, 90)   # 0‑89

# Mapeo rápido para etiquetar Visual y Default (según AAL)
VISUAL_ROIS = [1,2,3,4,17,18,19,20,57,58,59,60]
DEFAULT_ROIS = list(range(31,47)) + [89,90]

# Tomar solo 20 sujetos al azar (para que sea rápido)
np.random.seed(42)
archivos_1d = sorted(Path(INPUT_DIR).glob('*_rois_aal.1D'))
if len(archivos_1d) > 20:
    archivos_1d = np.random.choice(archivos_1d, 20, replace=False)

print(f"Probando {len(archivos_1d)} sujetos con m=4,5,6, tau={TAU}...\n")

resultados = []
for m_val in [4, 5, 6]:
    fm = ComplexityMetrics(m=m_val, tau=TAU)
    wpe_visual = []
    wpe_default = []
    for arch in archivos_1d:
        try:
            data = np.loadtxt(arch, dtype=np.float64)
            n_vols = data.shape[0]
            if n_vols < MIN_VOLS:
                continue
            # Solo cortical
            cortical = data[:, CORTICAL_IDXS]
            for roi_idx in range(cortical.shape[1]):
                serie = cortical[:, roi_idx]
                serie_norm = (serie - np.mean(serie)) / (np.std(serie) + 1e-12)
                wpe = fm.weighted_permutation_entropy(serie_norm)
                roi_num = roi_idx + 1   # AAL 1‑90
                if roi_num in VISUAL_ROIS:
                    wpe_visual.append(wpe)
                elif roi_num in DEFAULT_ROIS:
                    wpe_default.append(wpe)
        except Exception as e:
            print(f"Error con {arch}: {e}")

    if wpe_visual and wpe_default:
        d = (np.mean(wpe_default) - np.mean(wpe_visual)) / np.sqrt(
            (np.std(wpe_default)**2 + np.std(wpe_visual)**2) / 2)
        resultados.append({
            'm': m_val,
            'wPE_Visual_mean': np.mean(wpe_visual),
            'wPE_Default_mean': np.mean(wpe_default),
            'd_Cohen': d
        })
        print(f"m={m_val}: Visual mean wPE={np.mean(wpe_visual):.3f}, "
              f"Default mean wPE={np.mean(wpe_default):.3f}, d={d:.3f}")
    else:
        print(f"m={m_val}: No se pudieron calcular suficientes ROIs.")

# Resumen final
df_res = pd.DataFrame(resultados)
print("\nResumen:")
print(df_res.to_string(index=False))