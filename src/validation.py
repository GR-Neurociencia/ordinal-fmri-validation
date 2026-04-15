import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from .reference_systems import ChaosModel
from .bold_pipeline import BoldModel
from .metrics import ComplexityMetrics
from config.params import M_FIXED, TAU_FIXED, TR

class ValidationExperiments:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.chaos = ChaosModel(seed=seed)
        self.bold = BoldModel(TR=TR, n_points=500, seed=seed)
        self.metrics = ComplexityMetrics(m=M_FIXED, tau=TAU_FIXED)

    def exp1_reference_systems(self, n=500):
        signals = self.chaos.generate_all(n)
        results = {}
        for name, sig in signals.items():
            results[name] = self.metrics.calculate_all_metrics(sig)
        return {'individual': results, 'signals': signals}

    def exp2_noise_robustness(self, n_bootstrap=50):
        base_signal, _ = self.chaos.integrate('lorenz', t_span=(0, 200), dt=0.05, discard_points=500)
        base_signal = base_signal[:300]   # nos aseguramos de tener exactamente 300 puntos
        base = (base_signal - np.mean(base_signal)) / (np.std(base_signal) + 1e-12)
        levels = np.logspace(-2, 0, 12)
        noise_types = ['white', 'pink', 'brown']
        res = {}
        for nt in noise_types:
            pe_all = np.zeros((len(levels), n_bootstrap))
            wpe_all = np.zeros((len(levels), n_bootstrap))
            for i, lev in enumerate(levels):
                for b in range(n_bootstrap):
                    noise = self.chaos.colored_noise(len(base), nt)
                    noise = (noise - np.mean(noise)) / (np.std(noise) + 1e-12)
                    noisy = base + lev * noise
                    noisy = (noisy - np.mean(noisy)) / (np.std(noisy) + 1e-12)
                    m = self.metrics.calculate_all_metrics(noisy)
                    pe_all[i, b] = m['PE']
                    wpe_all[i, b] = m['wPE']
            res[nt] = {
                'levels': levels,
                'PE_mean': np.mean(pe_all, axis=1),
                'PE_low': np.percentile(pe_all, 2.5, axis=1),
                'PE_high': np.percentile(pe_all, 97.5, axis=1),
                'wPE_mean': np.mean(wpe_all, axis=1),
                'wPE_low': np.percentile(wpe_all, 2.5, axis=1),
                'wPE_high': np.percentile(wpe_all, 97.5, axis=1)
            }
        return {'noise_results': res}

    def exp3_series_length(self):
        sig, _ = self.chaos.integrate('rossler', t_span=(0, 100), dt=0.1, discard_points=500)
        max_len = min(1000, len(sig))
        lengths = np.linspace(50, max_len, 20).astype(int)
        pe_vals, wpe_vals, pat_counts = [], [], []
        for L in lengths:
            seg = sig[:L]
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-12)
            m = self.metrics.calculate_all_metrics(seg)
            pe_vals.append(m['PE'])
            wpe_vals.append(m['wPE'])
            pat_counts.append(m['n_unique_patterns'])
        if len(pe_vals) > 10:
            smooth = gaussian_filter1d(pe_vals, sigma=2)
            deriv = np.abs(np.diff(smooth) / np.diff(lengths))
            conv_idx = np.where(deriv < 0.001)[0]
            conv_len = lengths[conv_idx[0]] if len(conv_idx) else lengths[-1]
        else:
            conv_len = lengths[-1]
        return {'lengths': lengths, 'PE': pe_vals, 'wPE': wpe_vals,
                'pattern_counts': pat_counts, 'convergence_length': conv_len}

    def exp4_parameter_sensitivity(self):
        sig, _ = self.chaos.integrate('lorenz', t_span=(0, 100), dt=0.05, discard_points=500)
        sig = sig[:500]
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-12)
        m_range = range(3, 8)
        m_res = {}
        for m in m_range:
            tmp = ComplexityMetrics(m=m, tau=TAU_FIXED)
            m_res[m] = tmp.calculate_all_metrics(sig)
        tau_range = range(1, 31, 2)
        tau_res = {}
        for tau in tau_range:
            tmp = ComplexityMetrics(m=M_FIXED, tau=tau)
            tau_res[tau] = tmp.calculate_all_metrics(sig)
        return {'m_sensitivity': m_res, 'tau_sensitivity': tau_res}

    def exp5_bold_simulation(self, n_trials=5):
        neural, _ = self.chaos.integrate('rossler', t_span=(0, 600), dt=0.1, discard_points=1000)
        # Reference PE from first 500 samples (clean neural)
        neural_metrics = self.metrics.calculate_all_metrics(neural[:500])
        pe_neural = neural_metrics['PE']
        conditions = {
            'clean':       {'hrf': 'canonical', 'noise': 0.0, 'physio': False, 'motion': False},
            'low_noise':   {'hrf': 'canonical', 'noise': 0.1, 'physio': False, 'motion': False},
            'high_noise':  {'hrf': 'canonical', 'noise': 0.3, 'physio': False, 'motion': False},
            'physiological': {'hrf': 'canonical', 'noise': 0.1, 'physio': True, 'motion': False},
            'complete':    {'hrf': 'variable', 'noise': 0.3, 'physio': True, 'motion': True}
        }
        results = {}
        delta_all = []
        for cond, params in conditions.items():
            pe_list, wpe_list, delta_list = [], [], []
            for _ in range(n_trials):
                out = self.bold.generate_pipeline(
                    neural,
                    hrf_type=params['hrf'],
                    noise_level=params['noise'],
                    add_physio=params['physio'],
                    add_motion=params['motion']
                )
                bold_metrics = self.metrics.calculate_all_metrics(out['final'])
                pe_bold = bold_metrics['PE']
                wpe_bold = bold_metrics['wPE']
                delta = (pe_neural - pe_bold) / pe_neural
                pe_list.append(pe_bold)
                wpe_list.append(wpe_bold)
                delta_list.append(delta)
                if cond == 'complete':
                    delta_all.append(delta)
            results[cond] = {
                'PE_mean': np.mean(pe_list), 'PE_std': np.std(pe_list),
                'wPE_mean': np.mean(wpe_list), 'wPE_std': np.std(wpe_list),
                'delta_mean': np.mean(delta_list), 'delta_std': np.std(delta_list)
            }
        results['neural_reference'] = {'PE': pe_neural}
        results['complete_delta_summary'] = {
            'mean': np.mean(delta_all),
            'ci_95': np.percentile(delta_all, [2.5, 97.5])
        }
        return {'condition_results': results}
