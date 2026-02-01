import torch
import torch.nn as nn
from typing import Optional


class ActionHead(nn.Module):
    """
    General action head that can optionally use attention.
    Similar to Settlers of Catan's modular head design.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        use_attention: bool = False,
        attention_dim: Optional[int] = None,
        num_attention_heads: int = 4,
    ):
        super().__init__()
        
        self.use_attention = use_attention
        hidden_dim = hidden_dim or input_dim
        
        if use_attention:
            assert attention_dim is not None
            
            self.query_proj = nn.Linear(input_dim, attention_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=num_attention_heads,
                batch_first=True,
            )
            self.output_mlp = nn.Sequential(
                nn.Linear(attention_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.output_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
    
    def forward(
        self,
        state_encoding: torch.Tensor,
        attention_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_attention:
            query = self.query_proj(state_encoding).unsqueeze(1)
            attended, _ = self.attention(query, attention_tokens, attention_tokens)
            return self.output_mlp(attended.squeeze(1))
        else:
            return self.output_mlp(state_encoding)
