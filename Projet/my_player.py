from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from seahorse.utils.custom_exceptions import MethodNotImplementedError

from game_state_hex import GameStateHex
from player_hex import PlayerHex


class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        return self.minimax_search(current_state)

    def minimax_search(self, current_state: GameState) -> Action:
        best_action, _ = self.max_value(current_state, None, None)
        return best_action

    def max_value(self, current_state: GameState, alpha: int = None, beta: int = None):
        current_score: int = 0
        best_action : LightAction = None

        if current_state.is_done():
            return (best_action, current_score)

        for action in current_state.get_possible_light_actions():
            potential_action = current_state.apply_action(action)
            _, potential_score = self.min_value(potential_action, alpha, beta)

            if potential_score > current_score:
                current_score = potential_score
                best_action = action
                alpha = max(alpha, current_score) if alpha is not None else current_score
        return (best_action, current_score)

    def min_value(self, current_state: GameState, alpha: int = None, beta: int = None):
        current_score: int = 0
        best_action : LightAction = None

        if current_state.is_done():
            return (best_action, current_score)

        for action in current_state.get_possible_light_actions():
            potential_action = current_state.apply_action(action)
            _, potential_score = self.max_value(potential_action, alpha, beta)

            if potential_score < current_score:
                current_score = potential_score
                best_action = action
                beta = min(beta, current_score) if beta is not None else current_score
        return (best_action, current_score)