import numpy as np

class EmbeddingTools:
    def __init__(self, m_fixed=4, tau_fixed=15):
        self.m_fixed = m_fixed
        self.tau_fixed = tau_fixed

    def reconstruct_embedding(self, signal, m=None, tau=None):
        if m is None:
            m = self.m_fixed
        if tau is None:
            tau = self.tau_fixed
        n = len(signal)
        n_vectors = n - (m - 1) * tau
        if n_vectors < 1:
            return np.array([]).reshape(0, m), m, tau
        embedding = np.zeros((n_vectors, m))
        for i in range(m):
            start = i * tau
            end = start + n_vectors
            embedding[:, i] = signal[start:end]
        return embedding, m, tau