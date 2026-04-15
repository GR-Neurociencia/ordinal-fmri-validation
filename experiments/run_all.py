#!/usr/bin/env python3
import sys, os, random, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

random.seed(42)
np.random.seed(42)

from config.params import M_FIXED, TAU_FIXED, TR
from src.validation import ValidationExperiments
from visualization.plots import (plot_reference_systems, plot_exp1_results,
                                 plot_exp2_results, plot_exp3_results,
                                 plot_exp4_results, plot_exp5_results,
                                 create_summary_report, plot_3d_embedding)
import json
import pickle
from datetime import datetime

def save_results(all_res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, data in all_res.items():
        with open(os.path.join(out_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(data, f)
        # Also save as JSON for human readability
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k,v in obj.items()}
            if isinstance(obj, list):
                return [to_serializable(i) for i in obj]
            return obj
        with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
            json.dump(to_serializable(data), f, indent=2)

def main():
    print("="*80)
    print("STAGE 01: VALIDATION OF PE AND wPE FOR fMRI-BOLD")
    print(f"Parameters: m={M_FIXED}, τ={TAU_FIXED}, TR={TR}s, seed=42")
    print("="*80)

    exp = ValidationExperiments(seed=42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results_{timestamp}"
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Experiment 1
    res1 = exp.exp1_reference_systems(n=500)
    plot_reference_systems(res1['signals'], fig_dir)
    plot_exp1_results(res1, fig_dir)

    # Experiment 2
    res2 = exp.exp2_noise_robustness(n_bootstrap=50)
    plot_exp2_results(res2, fig_dir)

    # Experiment 3
    res3 = exp.exp3_series_length()
    plot_exp3_results(res3, fig_dir)

    # Experiment 4
    res4 = exp.exp4_parameter_sensitivity()
    plot_exp4_results(res4, fig_dir)

    # Experiment 5
    res5 = exp.exp5_bold_simulation(n_trials=5)
    plot_exp5_results(res5, fig_dir)

    # 3D embeddings for key systems
    for name, sig in res1['signals'].items():
        if name in ['Lorenz', 'Rössler', 'Sinusoidal', 'White Noise']:
            plot_3d_embedding(sig, name, M_FIXED, TAU_FIXED, fig_dir)

    # Summary report
    all_res = {
        'reference_systems': res1,
        'noise_robustness': res2,
        'series_length': res3,
        'parameter_sensitivity': res4,
        'bold_simulation': res5
    }
    create_summary_report(all_res, fig_dir)
    save_results(all_res, out_dir)

    print(f"\n✅ All results saved in: {out_dir}")
    print("✅ Figures saved in:", fig_dir)

if __name__ == '__main__':
    main()