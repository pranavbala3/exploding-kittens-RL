"""Value normalization for stable PPO training."""

import torch


class ValueNormalizer:
    """
    Normalizes value predictions to improve training stability.
    
    Based on Catan RL - uses fixed mean/std estimated from game statistics.
    """
    
    def __init__(self, mean=0.5, std=0.3):
        self.mean = mean
        self.std = std
    
    def normalize(self, values):
        """Normalize values to ~N(0,1)."""
        return (values - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, normalized_values):
        """Denormalize back to original scale."""
        return normalized_values * self.std + self.mean
