import torch
import torch.nn as nn
from typing import Dict


class HandEncoder(nn.Module):
    """Encodes the player's hand (13 card counts)."""
    
    def __init__(self, hand_dim: int = 13, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hand_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, hand: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand: [batch, 13] - card counts
        Returns:
            [batch, embed_dim]
        """
        return self.encoder(hand)


class OpponentEncoder(nn.Module):
    """Encodes opponent information with self-attention."""
    
    def __init__(
        self,
        opponent_feature_dim: int = 29,
        embed_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        
        # Project opponent features to embedding space
        self.feature_proj = nn.Linear(opponent_feature_dim, embed_dim)
        
        # Positional embeddings for opponent slots
        self.pos_embed = nn.Parameter(torch.randn(4, embed_dim))
        
        # Self-attention over opponents
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, opponent_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            opponent_features: [batch, 4, 29] - features for 4 opponents
        Returns:
            [batch, 4, embed_dim] - encoded opponent representations
        """
        # Project features
        x = self.feature_proj(opponent_features)  # [B, 4, D]
        
        # Add positional encoding
        x = x + self.pos_embed.unsqueeze(0)
        
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class GameStateEncoder(nn.Module):
    """Encodes global game state information."""
    
    def __init__(
        self,
        deck_dim: int = 2,
        discard_dim: int = 13,
        game_state_dim: int = 2,
        see_future_dim: int = 40,
        embed_dim: int = 128,
    ):
        super().__init__()
        
        self.deck_proj = nn.Linear(deck_dim, embed_dim)
        self.discard_proj = nn.Linear(discard_dim, embed_dim)
        self.game_state_proj = nn.Linear(game_state_dim, embed_dim)
        self.see_future_proj = nn.Linear(see_future_dim, embed_dim)
        
        # Combine all components
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            state_dict: Dict with 'deck_info', 'discard', 'game_state', 'see_future'
        Returns:
            [batch, embed_dim]
        """
        deck_enc = self.deck_proj(state_dict['deck_info'])
        discard_enc = self.discard_proj(state_dict['discard'])
        game_enc = self.game_state_proj(state_dict['game_state'])
        future_enc = self.see_future_proj(state_dict['see_future'])
        
        combined = torch.cat([deck_enc, discard_enc, game_enc, future_enc], dim=-1)
        return self.combiner(combined)