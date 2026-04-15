import numpy as np
import itertools
from math import factorial
from .embedding import EmbeddingTools

class ComplexityMetrics:
    def __init__(self, m=4, tau=15):
        self.m = m
        self.tau = tau
        self.embedder = EmbeddingTools(m_fixed=m, tau_fixed=tau)
        self.n_permutations = factorial(m)

    def get_ordinal_patterns(self, signal):
        embedding, _, _ = self.embedder.reconstruct_embedding(signal)
        if len(embedding) == 0:
            return [], np.array([])
        patterns = [tuple(np.argsort(vec)) for vec in embedding]
        return patterns, embedding

    def permutation_entropy(self, signal, normalize=True):
        patterns, _ = self.get_ordinal_patterns(signal)
        if len(patterns) < 10:
            return 0.0
        _, counts = np.unique(patterns, axis=0, return_counts=True)
        probs = counts / len(patterns)
        pe = -np.sum(probs * np.log(probs + 1e-12))
        if normalize:
            max_pe = np.log(self.n_permutations)
            return min(pe / max_pe, 1.0)
        return pe

    def weighted_permutation_entropy(self, signal, normalize=True):
        """Weighted PE using variance of embedded vectors (Fadlallah et al. 2013)."""
        patterns, embedding = self.get_ordinal_patterns(signal)
        if len(patterns) < 10:
            return 0.0
        weight_sum = {}
        for pat, vec in zip(patterns, embedding):
            w = np.var(vec)
            weight_sum[pat] = weight_sum.get(pat, 0.0) + w
        total = sum(weight_sum.values())
        if total == 0:
            return 0.0
        probs = np.array(list(weight_sum.values())) / total
        wpe = -np.sum(probs * np.log(probs + 1e-12))
        if normalize:
            max_wpe = np.log(self.n_permutations)
            return min(wpe / max_wpe, 1.0)
        return wpe

    def calculate_all_metrics(self, signal):
        pe = self.permutation_entropy(signal)
        wpe = self.weighted_permutation_entropy(signal)
        patterns, _ = self.get_ordinal_patterns(signal)
        n_unique = len(set(patterns)) if patterns else 0
        return {
            'PE': pe,
            'wPE': wpe,
            'n_patterns': len(patterns),
            'n_unique_patterns': n_unique,
            'pattern_diversity': n_unique / self.n_permutations if patterns else 0
        }