from enum import IntEnum


class CardType(IntEnum):
    """Enumeration of all card types in the game."""
    EXPLODING_KITTEN = 0
    DEFUSE = 1
    ATTACK = 2
    SKIP = 3
    FAVOR = 4
    SHUFFLE = 5
    SEE_FUTURE = 6
    NOPE = 7
    TACOCAT = 8
    RAINBOW_CAT = 9
    BEARDED_CAT = 10
    HAIRY_POTATO_CAT = 11
    CATTERMELON = 12


class ActionType(IntEnum):
    """Enumeration of all possible action types."""
    DRAW = 0
    PLAY_SKIP = 1
    PLAY_ATTACK = 2
    PLAY_SHUFFLE = 3
    PLAY_SEE_FUTURE = 4
    PLAY_FAVOR = 5
    PLAY_DEFUSE = 6
    PLAY_NOPE = 7          # only legal while responding to a pending action
    PLAY_2_OF_A_KIND = 8
    PLAY_3_OF_A_KIND = 9
    PASS = 10              # decline to Nope a pending action


NUM_CARD_TYPES = len(CardType)
NUM_ACTION_TYPES = len(ActionType)

# Actions that open a Nope window before they take effect (Draw and Defuse cannot be Noped).
NOPEABLE_ACTIONS = frozenset({
    ActionType.PLAY_SKIP,
    ActionType.PLAY_ATTACK,
    ActionType.PLAY_SHUFFLE,
    ActionType.PLAY_SEE_FUTURE,
    ActionType.PLAY_FAVOR,
    ActionType.PLAY_2_OF_A_KIND,
    ActionType.PLAY_3_OF_A_KIND,
})
