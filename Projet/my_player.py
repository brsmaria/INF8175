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
            score = current_state.get_player_score(self)
            if score == 1: # C'est une victoire pour nous
                return (float('inf'), None)
            else: # C'est une défaite ou une égalité
                return (float('-inf'), None)
        
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
            score = current_state.get_player_score(self)
            if score == 1: # C'est une victoire pour l'adversaire (du point de vue de max_value)
                return (float('inf'), None)
            else: # C'est une défaite pour l'adversaire
                return (float('-inf'), None)
        
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
        """
        Évalue un état non-terminal en utilisant l'algorithme de Dijkstra pour trouver
        le coût du plus court chemin pour chaque joueur.
        L'heuristique est la différence entre le coût du chemin de l'adversaire et celui du joueur.
        """
        my_path_cost = self.dijkstra_path_cost(current_state, self.piece_type)
        
        opponent_piece = "B" if self.piece_type == "R" else "R"
        opponent_path_cost = self.dijkstra_path_cost(current_state, opponent_piece)

        # Un coût plus faible est meilleur. Nous voulons maximiser (opp_cost - my_cost).
        return opponent_path_cost - my_path_cost

    def dijkstra_path_cost(self, current_state: GameStateHex, piece_type: str) -> float:
        """
        Calcule le coût du plus court chemin pour un joueur donné en utilisant Dijkstra.
        """
        import heapq

        board = current_state.get_rep().get_env()
        size = current_state.get_rep().get_dimensions()[0]
        
        # Nœuds de départ virtuels et d'arrivée virtuels pour gérer les bords
        start_node = (-1, -1)
        end_node = (-2, -2)

        distances = { (r, c): float('inf') for r in range(size) for c in range(size) }
        distances[start_node] = 0
        
        pq = [(0, start_node)]

        while pq:
            dist, current_pos = heapq.heappop(pq)

            if dist > distances.get(current_pos, float('inf')):
                continue
            
            if current_pos == end_node:
                return dist # Chemin trouvé

            # Gérer la connexion du nœud de départ aux bords appropriés
            if current_pos == start_node:
                if piece_type == "R": # Haut-Bas
                    neighbors = [(0, c) for c in range(size)]
                else: # Gauche-Droite
                    neighbors = [(r, 0) for r in range(size)]
            else:
                # On récupère le dictionnaire des voisins
                raw_neighbors = current_state.get_neighbours(current_pos[0], current_pos[1])
                # On extrait uniquement les coordonnées des voisins qui sont sur le plateau
                neighbors = []
                for neighbor_info in raw_neighbors.values():
                    neighbor_type, neighbor_pos = neighbor_info
                    if neighbor_type != "OUTSIDE":
                        neighbors.append(neighbor_pos)

            for neighbor_pos in neighbors:
                # Coût pour se déplacer vers le voisin
                cell_content = board.get(neighbor_pos)
                if cell_content is None: # Case vide
                    cost = 1
                elif cell_content.get_type() == piece_type: # Notre pièce
                    cost = 0
                else: # Pièce adverse
                    cost = float('inf')

                # Si le voisin est sur le bord d'arrivée, le connecter au nœud final
                is_end_border = False
                if piece_type == "R" and neighbor_pos[0] == size - 1:
                    is_end_border = True
                elif piece_type == "B" and neighbor_pos[1] == size - 1:
                    is_end_border = True

                if is_end_border:
                    if distances[start_node] + cost < distances.get(end_node, float('inf')):
                        distances[end_node] = distances[start_node] + cost if current_pos == start_node else dist + cost
                        heapq.heappush(pq, (distances[end_node], end_node))
                
                # Mise à jour de la distance normale pour les autres nœuds
                elif dist + cost < distances.get(neighbor_pos, float('inf')):
                    distances[neighbor_pos] = dist + cost
                    heapq.heappush(pq, (distances[neighbor_pos], neighbor_pos))

        return distances.get(end_node, float('inf'))