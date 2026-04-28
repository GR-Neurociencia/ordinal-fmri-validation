#!/usr/bin/env python3
"""
Batch cálculo de PE/wPE para ABIDE, con filtro opcional por lista de sujetos.
Uso:
  python3 batch_wpe_abide_v2.py --input_dir ... --phenotype ... --output ... [--subject_list sujetos.txt]
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.metrics import ComplexityMetrics
from src import M_FIXED, TAU_FIXED

M, TAU = M_FIXED, TAU_FIXED
MIN_VOLUMENES = 200
CORTICAL_START, CORTICAL_END = 0, 90

def extraer_file_id(nombre):
    return nombre.replace('_rois_aal.1D', '')

def cargar_serie_aal(archivo, min_vols=MIN_VOLUMENES):
    datos = np.loadtxt(archivo, dtype=np.float64)
    if datos.ndim != 2 or datos.shape[1] < CORTICAL_END:
        raise ValueError(f"Columnas insuficientes en {archivo}")
    n_vols = datos.shape[0]
    if n_vols < min_vols:
        raise ValueError(f"Serie corta ({n_vols} vols)")
    return datos[:, CORTICAL_START:CORTICAL_END], n_vols

def procesar_sujeto(archivo, fm):
    try:
        cortical_data, n_vols = cargar_serie_aal(archivo)
    except Exception as e:
        print(f"⚠️  Omitiendo {archivo}: {e}")
        return []
    file_id = extraer_file_id(os.path.basename(archivo))
    resultados = []
    for roi_idx in range(cortical_data.shape[1]):
        serie_raw = cortical_data[:, roi_idx]
        std_raw = float(np.std(serie_raw))
        serie_norm = (serie_raw - np.mean(serie_raw)) / (np.std(serie_raw) + 1e-12)
        pe = fm.permutation_entropy(serie_norm)
        wpe = fm.weighted_permutation_entropy(serie_norm)
        resultados.append({
            'FILE_ID': file_id,
            'ROI': roi_idx + 1,
            'PE': pe,
            'wPE': wpe,
            'std_raw': std_raw,
            'length': n_vols
        })
    return resultados

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--phenotype', required=True)
    parser.add_argument('--output', default='resultados_wpe.csv')
    parser.add_argument('--min_vols', type=int, default=MIN_VOLUMENES)
    parser.add_argument('--subject_list', help='Archivo con FILE_ID permitidos (uno por línea)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Error: {input_dir} no existe")

    # Cargar lista blanca si se proporciona
    whitelist = None
    if args.subject_list:
        with open(args.subject_list, 'r') as f:
            whitelist = set(line.strip() for line in f if line.strip())
        print(f"Usando lista blanca con {len(whitelist)} sujetos.")

    fm = ComplexityMetrics(m=M, tau=TAU)
    archivos = sorted(input_dir.glob('*_rois_aal.1D'))
    if whitelist:
        archivos = [a for a in archivos if extraer_file_id(a.name) in whitelist]
    print(f"Procesando {len(archivos)} archivos...")

    todos = []
    validos = 0
    for arch in archivos:
        res = procesar_sujeto(str(arch), fm)
        if res:
            todos.extend(res)
            validos += 1
        if validos % 50 == 0:
            print(f"  {validos} sujetos procesados...")
    if not todos:
        sys.exit("Ningún sujeto válido")
    df_metric = pd.DataFrame(todos)
    print(f"\nSujetos incluidos: {validos}, total filas: {len(df_metric)}")

    # Fenotipos
    df_feno = pd.read_csv(args.phenotype, encoding='latin1')
    col_feno = ['FILE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'MEAN_FD', 'SITE_ID']
    col_feno = [c for c in col_feno if c in df_feno.columns]
    df_feno = df_feno[col_feno]

    df_final = pd.merge(df_metric, df_feno, on='FILE_ID', how='left')
    col_orden = ['FILE_ID', 'SITE_ID', 'ROI', 'PE', 'wPE', 'std_raw', 'length',
                 'DX_GROUP', 'AGE_AT_SCAN', 'SEX']
    if 'MEAN_FD' in df_final.columns:
        col_orden.append('MEAN_FD')
    df_final = df_final[[c for c in col_orden if c in df_final.columns]]
    df_final.to_csv(args.output, index=False, float_format='%.6f')
    print(f"\nResultados guardados en '{args.output}'")

if __name__ == '__main__':
    main()