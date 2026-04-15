import numpy as np
from scipy import integrate

class ChaosModel:
    def __init__(self, seed=42):
        np.random.seed(seed)

    def lorenz(self, t, state, sigma=10.0, rho=28.0, beta=8/3):
        x, y, z = state
        return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]

    def rossler(self, t, state, a=0.2, b=0.2, c=5.7):
        x, y, z = state
        return [-y-z, x + a*y, b + z*(x-c)]

    def integrate(self, system, t_span=(0, 200), dt=0.01, discard_points=1000):
        """
        Integrates Lorenz or Rössler system.
        discard_points: number of initial points to discard (independent of dt).
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)
        if system == 'lorenz':
            func = self.lorenz
            init = [1.0, 1.0, 1.0]
        else:
            func = self.rossler
            init = [0.0, 0.0, 0.0]
        sol = integrate.solve_ivp(func, t_span, init, t_eval=t_eval,
                                  method='RK45', rtol=1e-9, atol=1e-12)
        x = sol.y[0]
        # Ensure discard_points is not larger than half the signal length
        discard = min(discard_points, len(x) // 2)
        return x[discard:], t_eval[discard:]

    def ar1(self, n, phi=0.9):
        eps = np.random.randn(n)
        x = np.zeros(n)
        x[0] = eps[0]
        for t in range(1, n):
            x[t] = phi * x[t-1] + eps[t]
        return x

    def ar2(self, n, phi1=0.6, phi2=0.3):
        eps = np.random.randn(n)
        x = np.zeros(n)
        x[0], x[1] = eps[0], eps[1]
        for t in range(2, n):
            x[t] = phi1 * x[t-1] + phi2 * x[t-2] + eps[t]
        return x

    def colored_noise(self, n, color):
        if n <= 1:
            return np.random.randn(n) if n > 0 else np.array([0.0])
        if color == 'white':
            return np.random.randn(n)
        elif color == 'pink':
            white = np.random.randn(n)
            n_pad = 2 ** int(np.ceil(np.log2(n))) * 2
            white_pad = np.zeros(n_pad)
            white_pad[:n] = white
            fft = np.fft.rfft(white_pad)
            freqs = np.fft.rfftfreq(n_pad)
            with np.errstate(divide='ignore'):
                filt = np.where(freqs > 0, 1.0 / np.sqrt(freqs), 1.0)
            pink_pad = np.fft.irfft(fft * filt)
            return pink_pad[:n] / (np.std(pink_pad[:n]) + 1e-12)
        else:  # brown
            brown = np.cumsum(np.random.randn(n))
            return brown / (np.std(brown) + 1e-12)

    def generate_all(self, n_points=500):
        signals = {}
        # Chaotic
        lorenz, _ = self.integrate('lorenz', t_span=(0, 100), dt=0.05, discard_points=500)
        signals['Lorenz'] = lorenz[:n_points]
        rossler, _ = self.integrate('rossler', t_span=(0, 200), dt=0.1, discard_points=500)
        signals['Rössler'] = rossler[:n_points]
        # Stochastic
        signals['AR(1) φ=0.9'] = self.ar1(n_points, phi=0.9)
        signals['AR(2)'] = self.ar2(n_points)
        # Noise
        signals['White Noise'] = self.colored_noise(n_points, 'white')
        signals['Pink Noise (1/f)'] = self.colored_noise(n_points, 'pink')
        signals['Brown Noise (1/f²)'] = self.colored_noise(n_points, 'brown')
        # Periodic
        t = np.linspace(0, 10 * np.pi, n_points)
        signals['Sinusoidal'] = np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(n_points)
        # Normalize all
        for name in signals:
            s = signals[name]
            signals[name] = (s - np.mean(s)) / (np.std(s) + 1e-12)
        return signals
