"""Value normalization for stable PPO training.

The critic predicts values in a normalized space; targets are normalized with running
statistics of the observed returns. (The original fixed mean=0.5/std=0.3 was borrowed from
Catan and does not match Exploding Kittens, where most players lose: returns average well
below zero.)
"""

import torch


class ValueNormalizer:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.var = float(std) ** 2
        self.count = 1e-4

    @property
    def std(self) -> float:
        return max(self.var, 1e-8) ** 0.5

    def update(self, values: torch.Tensor):
        """Fold a batch of raw returns into the running mean/variance (parallel Welford)."""
        values = values.detach().float().reshape(-1)
        n = values.numel()
        if n == 0:
            return
        batch_mean = values.mean().item()
        batch_var = values.var(unbiased=False).item()
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m_a = self.var * self.count
        m_b = batch_var * n
        self.var = (m_a + m_b + delta ** 2 * self.count * n / total) / total
        self.count = total

    def normalize(self, values):
        return (values - self.mean) / (self.std + 1e-8)

    def denormalize(self, normalized_values):
        return normalized_values * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state):
        self.mean, self.var, self.count = state['mean'], state['var'], state['count']
