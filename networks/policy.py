import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Tuple
from networks.encoders import HandEncoder, OpponentEncoder, GameStateEncoder
from networks.action_heads import ActionHead
from utils.value_norm import ValueNormalizer


class ExplodingKittensPolicy(nn.Module):
    """
    Main policy network for Exploding Kittens.
    
    Architecture:
    1. Encode different state components separately
    2. Combine encodings into unified state representation
    3. Use specialized heads for each action dimension
    
    Design philosophy similar to Settlers of Catan RL:
    - Modular encoders for different state components
    - Reusable ActionHead class
    - Clear separation between encoding and action selection
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        use_value_norm: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_value_norm = use_value_norm
        
        # Value normalization (from Catan RL)
        if use_value_norm:
            self.value_normalizer = ValueNormalizer(mean=0.5, std=0.3)
        
        # Encoders for different state components
        self.hand_encoder = HandEncoder(hand_dim=13, embed_dim=embed_dim)
        self.opponent_encoder = OpponentEncoder(
            opponent_feature_dim=29,
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
        )
        self.game_state_encoder = GameStateEncoder(
            deck_dim=2,
            discard_dim=13,
            game_state_dim=2,
            see_future_dim=40,
            embed_dim=embed_dim,
        )
        
        # Combine all encodings
        self.state_combiner = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # hand + game_state
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Action heads (5 heads for the 5 action dimensions)
        self.action_type_head = ActionHead(
            input_dim=embed_dim,
            output_dim=10,
            hidden_dim=hidden_dim,
            use_attention=False,
        )
        
        self.target_player_head = ActionHead(
            input_dim=embed_dim,
            output_dim=4,
            hidden_dim=hidden_dim,
            use_attention=True,
            attention_dim=embed_dim,
            num_attention_heads=num_attention_heads,
        )
        
        self.combo_card_head = ActionHead(
            input_dim=embed_dim,
            output_dim=13,
            hidden_dim=hidden_dim,
            use_attention=False,
        )
        
        self.request_card_head = ActionHead(
            input_dim=embed_dim,
            output_dim=13,
            hidden_dim=hidden_dim,
            use_attention=False,
        )
        
        self.defuse_position_head = ActionHead(
            input_dim=embed_dim,
            output_dim=56,  # Must match max_deck_size from GameConfig
            hidden_dim=hidden_dim,
            use_attention=False,
        )
        
        # Value head (for actor-critic methods)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def encode_state(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation into state and opponent representations.
        
        Args:
            obs: Observation dictionary from environment
        Returns:
            (state_encoding, opponent_encodings)
        """
        # Encode hand
        hand_enc = self.hand_encoder(obs['hand'])
        
        # Encode opponents
        opponent_enc = self.opponent_encoder(obs['opponents'])
        
        # Encode global game state
        game_state_dict = {
            'deck_info': obs['deck_info'],
            'discard': obs['discard'],
            'game_state': obs['game_state'],
            'see_future': obs['see_future'],
        }
        game_enc = self.game_state_encoder(game_state_dict)
        
        # Combine hand and game state into unified representation
        state_enc = self.state_combiner(torch.cat([hand_enc, game_enc], dim=-1))
        
        return state_enc, opponent_enc
    
    def forward(self, obs: Dict, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            obs: Observation dictionary
            masks: Action masks for each head
        Returns:
            Dictionary with logits for each head and value estimate
        """
        # Encode state
        state_enc, opponent_enc = self.encode_state(obs)
        
        # Get logits from each action head
        logits_action_type = self.action_type_head(state_enc)
        logits_target_player = self.target_player_head(state_enc, attention_tokens=opponent_enc)
        logits_combo_card = self.combo_card_head(state_enc)
        logits_request_card = self.request_card_head(state_enc)
        logits_defuse_pos = self.defuse_position_head(state_enc)
        
        # Apply masks (set invalid actions to -inf)
        logits_action_type = logits_action_type.masked_fill(~masks['head1'], float('-inf'))
        logits_target_player = logits_target_player.masked_fill(~masks['head2'], float('-inf'))
        logits_combo_card = logits_combo_card.masked_fill(~masks['head3'], float('-inf'))
        logits_request_card = logits_request_card.masked_fill(~masks['head4'], float('-inf'))
        logits_defuse_pos = logits_defuse_pos.masked_fill(~masks['head5'], float('-inf'))
        
        # Get value estimate
        value = self.value_head(state_enc)
        
        return {
            'logits1': logits_action_type,
            'logits2': logits_target_player,
            'logits3': logits_combo_card,
            'logits4': logits_request_card,
            'logits5': logits_defuse_pos,
            'value': value,
        }
    
    def get_action(self, obs: Dict, masks: Dict[str, torch.Tensor], deterministic: bool = False):
        """
        Sample an action from the policy.
        
        Args:
            obs: Observation dictionary
            masks: Action masks
            deterministic: If True, take argmax instead of sampling
        Returns:
            Dictionary with sampled actions and log probabilities
        """
        with torch.no_grad():
            output = self.forward(obs, masks)
            
            if deterministic:
                action1 = output['logits1'].argmax(dim=-1)
                action2 = output['logits2'].argmax(dim=-1)
                action3 = output['logits3'].argmax(dim=-1)
                action4 = output['logits4'].argmax(dim=-1)
                action5 = output['logits5'].argmax(dim=-1)
            else:
                # Sample from categorical distributions
                from torch.distributions import Categorical
                
                dist1 = Categorical(logits=output['logits1'])
                dist2 = Categorical(logits=output['logits2'])
                dist3 = Categorical(logits=output['logits3'])
                dist4 = Categorical(logits=output['logits4'])
                dist5 = Categorical(logits=output['logits5'])
                
                action1 = dist1.sample()
                action2 = dist2.sample()
                action3 = dist3.sample()
                action4 = dist4.sample()
                action5 = dist5.sample()
            
            return {
                'head1': action1.item() if action1.dim() == 0 else action1,
                'head2': action2.item() if action2.dim() == 0 else action2,
                'head3': action3.item() if action3.dim() == 0 else action3,
                'head4': action4.item() if action4.dim() == 0 else action4,
                'head5': action5.item() if action5.dim() == 0 else action5,
            }
    
    def evaluate_actions(self, obs, recurrent_hidden_states, masks, actions, action_masks):
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: Observation dict
            recurrent_hidden_states: For LSTM (you don't use this)
            masks: Done masks
            actions: Dict with head1-head5 actions taken
            action_masks: Valid action masks
        
        Returns:
            values: State value estimates
            action_log_probs: Log probability of actions
            entropy: Policy entropy
            recurrent_hidden_states: Updated hidden states (None for you)
        """
        # Forward pass
        output = self.forward(obs, action_masks)
        
        # Helper to handle empty masks
        def safe_categorical(logits, mask):
            """Create categorical, replacing all-inf with zeros."""
            # Check if all values are -inf (all masked)
            all_masked = torch.all(torch.isinf(logits) & (logits < 0), dim=-1, keepdim=True)
            # Replace all-inf rows with zeros
            safe_logits = torch.where(all_masked.expand_as(logits), torch.zeros_like(logits), logits)
            return Categorical(logits=safe_logits)
        
        # Create distributions for each head
        dist1 = safe_categorical(output['logits1'], action_masks['head1'])
        dist2 = safe_categorical(output['logits2'], action_masks['head2'])
        dist3 = safe_categorical(output['logits3'], action_masks['head3'])
        dist4 = safe_categorical(output['logits4'], action_masks['head4'])
        dist5 = safe_categorical(output['logits5'], action_masks['head5'])
        
        # Get log probs of the actions that were taken
        log_prob1 = dist1.log_prob(actions['head1'])
        log_prob2 = dist2.log_prob(actions['head2'])
        log_prob3 = dist3.log_prob(actions['head3'])
        log_prob4 = dist4.log_prob(actions['head4'])
        log_prob5 = dist5.log_prob(actions['head5'])
        
        # Sum log probs (product of probabilities = sum of log probs)
        action_log_probs = log_prob1 + log_prob2 + log_prob3 + log_prob4 + log_prob5
        
        # Compute entropy (measure of randomness)
        entropy = dist1.entropy() + dist2.entropy() + dist3.entropy() + \
                  dist4.entropy() + dist5.entropy()
        
        return output['value'], action_log_probs, entropy.mean(), None
