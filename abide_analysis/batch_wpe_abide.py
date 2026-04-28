#!/usr/bin/env python3
"""
Cálculo por lotes de PE y wPE para el dataset ABIDE preprocesado.

Estructura esperada:
- Un directorio con archivos *_rois_aal.1D (cada uno 116 columnas, sin cabecera).
- Un archivo CSV fenotípico (Phenotypic_V1_0b_preprocessed1.csv).

Salida: un CSV con columnas:
  SUB_ID, SITE, ROI, PE, wPE, std_raw, length, DX_GROUP, AGE_AT_SCAN, SEX, MEAN_FD
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ajustar ruta para importar la implementación del preprint (igual que en los scripts originales)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.metrics import ComplexityMetrics
from src import M_FIXED, TAU_FIXED, TR

# ------------------- Configuración -------------------
M = M_FIXED          # 4
TAU = TAU_FIXED      # 15
TR_VAL = TR          # solo referencia
MIN_VOLUMENES = 200   # longitud mínima de la serie

# Índices de las regiones corticales del AAL (0‑based)
# AAL tiene 116 regiones; las primeras 90 son corticales, el resto subcortical/cerebelo.
CORTICAL_START = 0
CORTICAL_END = 90

# Método de peso para wPE
WPE_METHOD = 'variance'   # igual que en calcular_metricas_fsl.py

# ------------------- Funciones -------------------
def extraer_file_id(nombre_archivo: str) -> str:
    """
    Extrae el FILE_ID a partir del nombre del archivo.
    Ejemplo: 'Pitt_0050003_rois_aal.1D' -> 'Pitt_0050003'
    """
    # Eliminar sufijo conocido
    base = nombre_archivo.replace('_rois_aal.1D', '')
    return base

def cargar_serie_aal(archivo: str, min_vols: int = MIN_VOLUMENES):
    """
    Carga el archivo .1D de AAL y devuelve:
    - datos: array (n_vols, 90) sólo regiones corticales
    - n_vols: número de volúmenes
    Lanza excepción si no se alcanza el mínimo de volúmenes o hay error de lectura.
    """
    # Leer el archivo (columnas separadas por espacios/tabulaciones)
    try:
        datos = np.loadtxt(archivo, dtype=np.float64)
    except Exception as e:
        raise IOError(f"No se pudo leer {archivo}: {e}")

    if datos.ndim != 2 or datos.shape[1] < CORTICAL_END:
        raise ValueError(f"El archivo {archivo} no tiene 116 columnas (encontradas {datos.shape[1]})")

    n_vols = datos.shape[0]
    if n_vols < min_vols:
        raise ValueError(f"Serie demasiado corta ({n_vols} volúmenes, mínimo {min_vols})")

    # Extraer solo las 90 regiones corticales (columnas 0 a 89)
    cortical_data = datos[:, CORTICAL_START:CORTICAL_END]
    return cortical_data, n_vols

def procesar_sujeto(archivo: str, fm: ComplexityMetrics):
    """
    Procesa las 90 ROIs de un sujeto.
    Retorna una lista de diccionarios con los resultados de cada ROI.
    """
    try:
        cortical_data, n_vols = cargar_serie_aal(archivo)
    except Exception as e:
        print(f"⚠️  Omitiendo {archivo}: {e}")
        return []

    file_id = extraer_file_id(os.path.basename(archivo))
    resultados = []

    for roi_idx in range(cortical_data.shape[1]):
        # Serie en crudo (sin estandarizar)
        serie_raw = cortical_data[:, roi_idx]
        # Proxy de ruido: desviación estándar de la señal original
        std_raw = float(np.std(serie_raw))
        # Estandarización (z-score) como en el preprint
        serie_norm = (serie_raw - np.mean(serie_raw)) / (np.std(serie_raw) + 1e-12)

        # Calcular entropías con los parámetros fijos
        pe = fm.permutation_entropy(serie_norm)
        wpe = fm.weighted_permutation_entropy(serie_norm)

        resultados.append({
            'FILE_ID': file_id,
            'ROI': roi_idx + 1,      # numeración AAL (1-90)
            'PE': pe,
            'wPE': wpe,
            'std_raw': std_raw,
            'length': n_vols
        })

    return resultados

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch cálculo de PE/wPE sobre series AAL de ABIDE")
    parser.add_argument('--input_dir', required=True, help="Directorio con archivos *_rois_aal.1D")
    parser.add_argument('--phenotype', required=True, help="Archivo CSV fenotípico de ABIDE")
    parser.add_argument('--output', default='resultados_wpe_abide.csv', help="Archivo CSV de salida")
    parser.add_argument('--min_vols', type=int, default=MIN_VOLUMENES, help="Volúmenes mínimos (por defecto 200)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} no es un directorio válido.")
        sys.exit(1)

    # Instanciar la clase de métricas una única vez
    fm = ComplexityMetrics(m=M, tau=TAU)

    # Listar todos los archivos .1D
    archivos_1d = sorted(input_dir.glob('*_rois_aal.1D'))
    if not archivos_1d:
        print(f"No se encontraron archivos *_rois_aal.1D en {input_dir}")
        sys.exit(1)

    print(f"\nProcesando {len(archivos_1d)} sujetos...")
    print(f"Parámetros: m={M}, τ={TAU}, mín. {args.min_vols} volúmenes\n")

    # Acumular resultados
    todos_resultados = []
    sujetos_validos = 0
    for archivo in archivos_1d:
        res = procesar_sujeto(str(archivo), fm)
        if res:
            todos_resultados.extend(res)
            sujetos_validos += 1
        # Mostrar progreso
        if sujetos_validos % 20 == 0:
            print(f"  {sujetos_validos} sujetos procesados...")

    if not todos_resultados:
        print("Ningún sujeto superó los criterios. Abortando.")
        sys.exit(1)

    # Convertir a DataFrame
    df_metric = pd.DataFrame(todos_resultados)
    print(f"\nSujetos incluidos: {sujetos_validos}")
    print(f"Total de series ROI: {len(df_metric)}")

    # Cargar fenotipos
    print("\nCargando archivo fenotípico...")
    try:
        df_feno = pd.read_csv(args.phenotype, encoding='latin1')
    except Exception as e:
        print(f"Error al leer CSV fenotípico: {e}")
        sys.exit(1)

    # La tabla ABIDE preprocessed suele tener columnas 'FILE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'MEAN_FD', 'SITE_ID', etc.
    # Ajusta los nombres si tu versión difiere.
    columnas_necesarias = ['FILE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'MEAN_FD', 'SITE_ID']
    columnas_presentes = [c for c in columnas_necesarias if c in df_feno.columns]
    if len(columnas_presentes) < 5:
        print("El archivo fenotípico no contiene las columnas esperadas. Revisa los nombres.")
        print(f"Columnas disponibles: {list(df_feno.columns)}")
        sys.exit(1)

    df_feno = df_feno[columnas_presentes]

    # Unir por FILE_ID (mantener solo los sujetos con métricas)
    df_final = pd.merge(df_metric, df_feno, on='FILE_ID', how='left')
    perdi = df_final['FILE_ID'].isna().sum()
    if perdi > 0:
        print(f"¡Atención! {perdi} registros sin fenotipo (FILE_ID no encontrado).")

    # Reordenar columnas para claridad
    col_orden = ['FILE_ID', 'SITE_ID', 'ROI', 'PE', 'wPE', 'std_raw', 'length',
                 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'MEAN_FD']
    col_final = [c for c in col_orden if c in df_final.columns]
    df_final = df_final[col_final]

    # Guardar
    df_final.to_csv(args.output, index=False, float_format='%.6f')
    print(f"\nResultados guardados en '{args.output}'")
    print(f"Muestra:\n{df_final.head()}")

if __name__ == "__main__":
    main()