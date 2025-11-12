from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.light_action import LightAction
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex

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
        self.max_depth = 2

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        if self.is_opening_move(current_state):
            return self.opening_strategy(current_state)

        return self.minimax_search(current_state)
    
    def is_opening_move(self, current_state: GameStateHex) -> bool:
        env = current_state.rep.env
        num_pieces = len(env)
        return num_pieces <= 1
    
    def opening_strategy(self, current_state: GameStateHex) -> LightAction:
        """Stratégie d'ouverture pour Hex"""
        env = current_state.rep.env
        board_size = current_state.rep.get_dimensions()[0]
        center = board_size // 2 - 1
        
        if len(env) == 0:
            # Premier joueur : centre
            row, col = center, center
            
        elif len(env) == 1:
            # Deuxième joueur
            opp_row, opp_col = list(env.keys())[0]
            
            # Vérifier si adversaire au centre (rayon 2)
            dist_center = abs(opp_row - center) + abs(opp_col - center)
            
            if dist_center <= 2:
                # Adversaire proche du centre → Jouer une réponse "shape"
                # "lean toward your connection direction"
                
                if self.piece_type == "R":  # Rouge connecte HAUT-BAS
                    # Jouer proche du centre, mais vers le HAUT ou BAS
                    row = center - 1  # Légèrement vers le haut
                    col = center
                else:  # Bleu connecte GAUCHE-DROITE
                    # Jouer proche du centre, mais vers la GAUCHE ou DROITE
                    row = center
                    col = center - 1  # Légèrement vers la gauche
            else:
                # Adversaire loin du centre → Prendre le centre
                row, col = center, center
        else:
            row, col = center, center
            
        return LightAction({"piece": self.piece_type, "position": (row, col)})

    def minimax_search(self, current_state: GameState) -> Action:
        _, best_action = self.max_value(current_state, self.max_depth, float('-inf'), float('inf'))
        return best_action

    def max_value(self, current_state: GameState, depth: int, alpha: float, beta: float):
        if current_state.is_done():
            return (current_state.get_player_score(self), None)
        
        if depth == 0:
            return (self.evaluate_state(current_state), None)
        
        current_score: float = float('-inf')
        best_action: LightAction | None = None

        for action in current_state.get_possible_light_actions():
            potential_action = current_state.apply_action(action)
            potential_score, _ = self.min_value(potential_action, depth - 1, alpha, beta)

            if potential_score > current_score:
                current_score = potential_score
                best_action = action
            
            alpha = max(alpha, current_score)
            if current_score >= beta:
                break 
                
        return (current_score, best_action)

    def min_value(self, current_state: GameState, depth: int, alpha: float, beta: float):
        if current_state.is_done():
            return (current_state.get_player_score(self), None)
        
        if depth == 0:
            return (self.evaluate_state(current_state), None)
        
        current_score: float = float('inf')
        best_action: LightAction | None = None

        for action in current_state.get_possible_light_actions():
            potential_action = current_state.apply_action(action)
            potential_score, _ = self.max_value(potential_action, depth - 1, alpha, beta)

            if potential_score < current_score:
                current_score = potential_score
                best_action = action
            
            beta = min(beta, current_score)
            if current_score <= alpha:
                break 
                
        return (current_score, best_action)
    
    def evaluate_state(self, current_state: GameStateHex) -> float:
        """Évalue un état non-terminal"""
        my_piece = self.piece_type
        opp_piece = "B" if my_piece == "R" else "R"
        env = current_state.rep.env
        
        my_count = sum(1 for p in env.values() if p.get_type() == my_piece)
        opp_count = sum(1 for p in env.values() if p.get_type() == opp_piece)
        
        return float(my_count - opp_count)