from dataclasses import dataclass, field
from typing import Dict
from core.card_types import CardType


@dataclass
class GameConfig:
    """Configuration for game rules and setup."""
    num_players: int = 4
    cards_per_player: int = 7
    max_deck_size: int = 56
    enable_special_combos: bool = True
    enable_nope: bool = True
    
    # Base game card counts
    card_counts: Dict[CardType, int] = field(default_factory=lambda: {
        CardType.EXPLODING_KITTEN: 4,
        CardType.DEFUSE: 6,
        CardType.ATTACK: 4,
        CardType.SKIP: 4,
        CardType.FAVOR: 4,
        CardType.SHUFFLE: 4,
        CardType.SEE_FUTURE: 5,
        CardType.NOPE: 5,
        CardType.TACOCAT: 4,
        CardType.RAINBOW_CAT: 4,
        CardType.BEARDED_CAT: 4,
        CardType.HAIRY_POTATO_CAT: 4,
        CardType.CATTERMELON: 4,
    })
    
    def validate(self):
        """Validate configuration."""
        assert 2 <= self.num_players <= 5, "Must have 2-5 players"
        assert self.cards_per_player >= 1, "Must deal at least 1 card"