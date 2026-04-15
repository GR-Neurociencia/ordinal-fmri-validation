import numpy as np
from .reference_systems import ChaosModel

class BoldModel:
    def __init__(self, TR=2.0, n_points=300, sampling_rate=100.0, seed=42):
        np.random.seed(seed)
        self.TR = TR
        self.n_points = n_points
        self.sr = sampling_rate
        self.t = np.arange(n_points) * TR

    def hrf(self, t, a1=6.0, a2=16.0, d1=6.0, d2=16.0, c=6.0):
        t = np.maximum(t, 1e-6)
        peak = (t/d1)**a1 * np.exp(-t/d1)
        undershoot = (t/d2)**a2 * np.exp(-t/d2)
        h = peak - c * undershoot
        return h / np.max(h) if np.max(h)>0 else h

    def generate_hrf(self, hrf_type='canonical'):
        t_hrf = np.arange(0, 32, 1/self.sr)
        if hrf_type == 'canonical':
            h = self.hrf(t_hrf)
        elif hrf_type == 'fast':
            h = self.hrf(t_hrf, d1=4, d2=12, c=8)
        elif hrf_type == 'slow':
            h = self.hrf(t_hrf, d1=8, d2=20, c=4)
        else:  # variable
            d1 = np.random.uniform(5,7)
            d2 = np.random.uniform(14,18)
            c = np.random.uniform(5,7)
            h = self.hrf(t_hrf, d1=d1, d2=d2, c=c)
            h += np.random.randn(len(h))*0.05
            h = np.maximum(h, 0)
        return t_hrf, h / np.sum(np.abs(h))

    def neural_to_bold(self, neural, hrf_type='canonical', noise_level=0.2):
        neural = neural[:self.n_points]
        t_high = np.linspace(0, self.n_points*self.TR, int(self.n_points*self.TR*self.sr))
        neural_high = np.interp(t_high, self.t, neural)
        _, hrf = self.generate_hrf(hrf_type)
        bold_high = np.convolve(neural_high, hrf, mode='full')[:len(neural_high)]
        step = int(self.sr * self.TR)
        bold = bold_high[::step][:self.n_points]
        bold = (bold - np.mean(bold)) / (np.std(bold)+1e-12)
        if noise_level > 0:
            noise = ChaosModel().colored_noise(self.n_points, 'pink')
            bold = bold + noise_level * noise
            bold = (bold - np.mean(bold)) / (np.std(bold)+1e-12)
        return bold

    def add_physio(self, bold):
        resp = 0.15 * (1+0.3*np.sin(2*np.pi*0.01*self.t)) * np.sin(2*np.pi*0.3*self.t + 0.1*np.cumsum(np.random.randn(len(self.t)))*0.1)
        nyq = 1/(2*self.TR)
        card_aliased = 1.2 % (2*nyq)
        if card_aliased > nyq:
            card_aliased = 2*nyq - card_aliased
        card = 0.1 * np.sin(2*np.pi*card_aliased*self.t + 0.05*np.cumsum(np.random.randn(len(self.t)))*0.05)
        return bold + resp + card

    def generate_pipeline(self, neural, hrf_type='variable', noise_level=0.3,
                          add_physio=True, add_motion=True):
        bold = self.neural_to_bold(neural, hrf_type, noise_level/2)
        if add_physio:
            bold = self.add_physio(bold)
        if add_motion:
            n_spikes = np.random.randint(1,4)
            for _ in range(n_spikes):
                pos = np.random.randint(20, len(bold)-20)
                dur = np.random.randint(1,3)
                amp = np.random.uniform(0.5,1.5)
                bold[pos:pos+dur] += amp
        bold = (bold - np.mean(bold)) / (np.std(bold)+1e-12)
        return {'neural': neural[:self.n_points], 'final': bold}