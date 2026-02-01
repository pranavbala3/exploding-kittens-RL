import torch
import numpy as np
from typing import Dict


def process_observation(obs: Dict) -> Dict[str, torch.Tensor]:
    """
    Convert environment observation to tensor format for network input.
    
    Args:
        obs: Raw observation from environment
    Returns:
        Dictionary of tensors ready for network input
    """
    # Extract and convert to tensors
    hand = torch.FloatTensor(obs['your_hand']).unsqueeze(0)  # [1, 13]
    
    # Process opponents (stack their features)
    opponent_features = []
    for opp in obs['opponents']:
        features = np.concatenate([
            [opp['num_cards']],
            [float(opp['is_alive'])],
            [opp['turns_remaining']],
            opp['min_counts'],
            opp['max_counts'],
        ])
        opponent_features.append(features)
    opponents = torch.FloatTensor(np.stack(opponent_features)).unsqueeze(0)  # [1, 4, 29]
    
    # Deck info
    deck_info = torch.FloatTensor([
        obs['deck_size'],
        obs['exploding_kittens_in_deck'],
    ]).unsqueeze(0)  # [1, 2]
    
    # Discard pile
    discard = torch.FloatTensor(obs['discard_counts']).unsqueeze(0)  # [1, 13]
    
    # Game state
    game_state = torch.FloatTensor([
        float(obs['have_drawn_this_turn']),
        float(obs['holding_exploding_kitten']),
    ]).unsqueeze(0)  # [1, 2]
    
    # See the future (one-hot encode top 3 cards)
    see_future = np.zeros(40)  # 3 cards × 13 types + 1 for active flag
    if obs['see_future_active']:
        for i, card in enumerate(obs['top_3_cards']):
            if card is not None:
                see_future[i * 13 + card] = 1.0
        see_future[39] = 1.0  # Active flag
    see_future = torch.FloatTensor(see_future).unsqueeze(0)  # [1, 40]
    
    return {
        'hand': hand,
        'opponents': opponents,
        'deck_info': deck_info,
        'discard': discard,
        'game_state': game_state,
        'see_future': see_future,
    }


def process_masks(masks: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert action masks to tensors."""
    return {
        'head1': torch.BoolTensor(masks['head1']).unsqueeze(0),
        'head2': torch.BoolTensor(masks['head2']).unsqueeze(0),
        'head3': torch.BoolTensor(masks['head3']).unsqueeze(0),
        'head4': torch.BoolTensor(masks['head4']).unsqueeze(0),
        'head5': torch.BoolTensor(masks['head5']).unsqueeze(0),
    }


# ==================== FILE: main.py ====================
"""Example usage of the policy network."""

import torch
from networks.policy import ExplodingKittensPolicy
from utils.observation_utils import process_observation, process_masks
# from exploding_kittens.environment import ExplodingKittensEnv


def example_usage():
    """Demonstrate how to use the policy network."""
    
    # Create policy
    policy = ExplodingKittensPolicy(
        embed_dim=128,
        hidden_dim=128,
        num_attention_heads=4,
    )
    
    # Example: Get action from observation
    # env = ExplodingKittensEnv()
    # raw_obs = env.reset()
    # raw_masks = env.get_action_mask(0)
    
    # obs = process_observation(raw_obs)
    # masks = process_masks(raw_masks)
    
    # action = policy.get_action(obs, masks, deterministic=False)
    # print(f"Sampled action: {action}")
    
    print("Policy network created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")


if __name__ == "__main__":
    example_usage()