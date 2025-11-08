from math import inf

from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.heavy_action import HeavyAction
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
        self._max_depth = 3

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        return self.minimax_search(current_state)

    def minimax_search(self, current_state: GameState, alpha: int = inf, beta: int = -inf) -> Action:
        _, best_heavy_action = self.max_value(current_state, self._max_depth, alpha, beta)
        return current_state.convert_heavy_action_to_light_action(best_heavy_action)

    def max_value(self, current_state: GameState, depth: int, alpha: int, beta: int):
        if current_state.is_done():
            return (current_state.get_player_score(self), None)
        if depth == 0:
            return (self.heuristic_evalutation(current_state), None)

        best_score: float = -inf
        best_heavy_action: HeavyAction | None = None

        for heavy_action in current_state.generate_possible_heavy_actions():
            next_state = heavy_action.get_next_game_state()
            score, _ = self.min_value(next_state, depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_heavy_action = heavy_action
                alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return (best_score, best_heavy_action)

    def min_value(self, current_state: GameState, depth: int, alpha: int, beta: int):
        if current_state.is_done():
            return (current_state.get_player_score(self), None)
        if depth == 0:
            return (self.heuristic_evalutation(current_state), None)

        best_score: float = inf
        best_heavy_action: HeavyAction | None = None

        for heavy_action in current_state.generate_possible_heavy_actions():
            next_state = heavy_action.get_next_game_state()
            score, _ = self.max_value(next_state, depth - 1, alpha, beta)
            if score < best_score:
                best_score = score
                best_heavy_action = heavy_action
                beta = min(beta, best_score)
            if beta <= alpha:
                break
        return (best_score, best_heavy_action)

    def heuristic_evalutation(self, state: GameStateHex) -> float:
        my_piece = self.piece_type
        opp_piece = "B" if my_piece == "R" else "R"
        env = state.rep.env
        my_count = sum(1 for p in env.values() if p.get_type() == my_piece)
        opp_count = sum(1 for p in env.values() if p.get_type() == opp_piece)
        return (my_count - opp_count)