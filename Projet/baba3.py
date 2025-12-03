from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from collections import deque

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
        self.max_depth = 3
        self.opponent_piece = "B" if piece_type == "R" else "R"
        self.is_opening_move_called = False

        # Cache pour les sources/destinations
        self._sources_cache = {}
        self._destinations_cache = {}
        self._is_horizontal_cache = {}
        self._board_size = None

        # Cache for neighbors within distance 2
        self._neighbors_cache = {}

        # Cache for neighbors within distance 2
        self._neighbors_cache = {}

        # Killer heuristic: depth -> action
        self._killer_moves = {}

        # History of moves (last 6 moves)
        self._move_history = []
        self._last_positions = None

    def compute_action(
        self, current_state: GameState, remaining_time: int = 1e9, **kwargs
    ) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        # Update history with opponent's move
        current_env = current_state.rep.env
        current_positions = set(current_env.keys())

        if self._last_positions is not None:
            new_moves = current_positions - self._last_positions
            for move in new_moves:
                self._move_history.append(move)
        else:
            # First time we see the board or first move
            for move in current_positions:
                self._move_history.append(move)

        # Keep only last 6 moves
        if len(self._move_history) > 6:
            self._move_history = self._move_history[-6:]

        # Use opening strategy only on first or second move
        if len(current_env) <= 1:
            best_action = self.opening_strategy(current_state)
        else:
            best_action = self.minimax_search(current_state)

        # Update history with my move
        pos = best_action.data["position"]
        self._move_history.append(pos)
        if len(self._move_history) > 6:
            self._move_history = self._move_history[-6:]

        self._last_positions = current_positions | {pos}

        return best_action

    def opening_strategy(self, current_state: GameStateHex) -> LightAction:
        """Stratégie d'ouverture pour Hex"""
        env = current_state.rep.env
        board_size = current_state.rep.get_dimensions()[0]
        center = board_size // 2 - 1

        center_positions = [
            (center, center),
            (center, center + 1),
            (center + 1, center),
            (center + 1, center + 1),
        ]

        row, col = center_positions[0]  # Default fallback

        for pos in center_positions:
            if pos not in env:
                row, col = pos
                break

        return LightAction({"piece": self.piece_type, "position": (row, col)})

    def minimax_search(self, current_state: GameState) -> Action:
        # Pass a copy of the history as a list
        history = list(self._move_history)
        _, best_action = self.max_value(
            current_state, self.max_depth, float("-inf"), float("inf"), history
        )

        if best_action is None:
            return list(current_state.get_possible_light_actions())[0]
        return best_action

    def max_value(
        self,
        current_state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        history: list,
    ):
        if current_state.is_done():
            # Terminal state: return large value based on winner
            score = current_state.get_player_score(self)
            return (10000 if score == 1 else -10000, None)

        if depth == 0:
            return (self.evaluate_state_cost(current_state), None)

        current_score: float = float("-inf")
        best_action: LightAction | None = None

        # Generate relevant moves from history
        relevant_moves = self.get_relevant_moves(current_state, history)

        # Move Ordering
        possible_actions = [
            LightAction({"piece": self.piece_type, "position": pos})
            for pos in relevant_moves
        ]

        actions_with_priority = []

        killer_move = self._killer_moves.get(depth)

        for action in possible_actions:
            priority = 1  # Default

            # Killer Heuristic
            if killer_move and action.data["position"] == killer_move.data["position"]:
                priority = -1  # Highest
            # Proximity Sorting
            elif self.is_adjacent_to_piece(current_state, action):
                priority = 0  # High

            actions_with_priority.append((priority, action))

        if not actions_with_priority and possible_actions:
            for action in possible_actions:
                actions_with_priority.append((1, action))

        actions_with_priority.sort(key=lambda x: x[0])

        for _, action in actions_with_priority:
            potential_state = current_state.apply_action(action)

            if potential_state.is_done():
                # This is a winning move - return immediately with high score
                score = potential_state.get_player_score(self)
                return (10000 if score == 1 else -10000, action)

            # Update history for next level
            new_history = history + [action.data["position"]]
            if len(new_history) > 6:
                new_history.pop(0)

            potential_score, _ = self.min_value(
                potential_state, depth - 1, alpha, beta, new_history
            )

            if potential_score > current_score:
                current_score = potential_score
                best_action = action

            alpha = max(alpha, current_score)
            if current_score >= beta:
                # Killer Heuristic: Store the move that caused the cutoff
                self._killer_moves[depth] = action
                break

        return (current_score, best_action)

    def min_value(
        self,
        current_state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        history: list,
    ):
        if current_state.is_done():
            # Terminal state: return large value based on winner
            score = current_state.get_player_score(self)
            return (10000 if score == 1 else -10000, None)

        if depth == 0:
            return (self.evaluate_state_cost(current_state), None)

        current_score: float = float("inf")
        best_action: LightAction | None = None

        # Generate relevant moves from history
        relevant_moves = self.get_relevant_moves(current_state, history)

        # Move Ordering
        possible_actions = [
            LightAction({"piece": self.opponent_piece, "position": pos})
            for pos in relevant_moves
        ]

        actions_with_priority = []

        killer_move = self._killer_moves.get(depth)

        for action in possible_actions:
            priority = 1  # Default

            # Killer Heuristic
            if killer_move and action.data["position"] == killer_move.data["position"]:
                priority = -1  # Highest
            # Proximity Sorting
            elif self.is_adjacent_to_piece(current_state, action):
                priority = 0  # High

            actions_with_priority.append((priority, action))

        if not actions_with_priority and possible_actions:
            for action in possible_actions:
                actions_with_priority.append((1, action))

        actions_with_priority.sort(key=lambda x: x[0])

        for _, action in actions_with_priority:
            potential_state = current_state.apply_action(action)

            if potential_state.is_done():
                # Opponent's winning move - return with high score from our perspective
                score = potential_state.get_player_score(self)
                return (10000 if score == 1 else -10000, action)

            # Update history for next level
            new_history = history + [action.data["position"]]
            if len(new_history) > 6:
                new_history.pop(0)

            potential_score, _ = self.max_value(
                potential_state, depth - 1, alpha, beta, new_history
            )

            if potential_score < current_score:
                current_score = potential_score
                best_action = action

            beta = min(beta, current_score)
            if current_score <= alpha:
                # Killer Heuristic: Store the move that caused the cutoff
                self._killer_moves[depth] = action
                break

        return (current_score, best_action)

    def get_neighbors_dist_1_2(
        self, current_state: GameStateHex, row: int, col: int
    ) -> set:
        """
        Returns a set of positions within distance 2 of (row, col).
        """
        if (row, col) in self._neighbors_cache:
            return self._neighbors_cache[(row, col)]

        neighbors_1 = set()
        neighbors_2 = set()

        # Dist 1
        n1_dict = current_state.get_neighbours(row, col)
        for n_info in n1_dict.values():
            n_type, n_pos = n_info
            if n_type != "OUTSIDE":
                neighbors_1.add(n_pos)

        # Dist 2
        for r1, c1 in neighbors_1:
            n2_dict = current_state.get_neighbours(r1, c1)
            for n_info in n2_dict.values():
                n_type, n_pos = n_info
                if (
                    n_type != "OUTSIDE"
                    and n_pos != (row, col)
                    and n_pos not in neighbors_1
                ):
                    neighbors_2.add(n_pos)

        result = neighbors_1.union(neighbors_2)
        self._neighbors_cache[(row, col)] = result
        return result

    def get_relevant_moves(self, current_state: GameStateHex, history: list) -> set:
        """
        Compute relevant moves based on the history of the last 6 moves.
        Relevant moves are empty cells within distance 2 of any move in the history.
        """
        env = current_state.get_rep().get_env()
        relevant = set()

        if not history:
            # Fallback if history is empty (should be handled by opening strategy, but just in case)
            board_size = current_state.get_rep().get_dimensions()[0]
            center = board_size // 2
            if (center, center) not in env:
                return {(center, center)}
            else:
                # If center taken and no history (weird), return all empty neighbors of center
                return self.get_neighbors_dist_1_2(current_state, center, center)

        for pos in history:
            neighbors = self.get_neighbors_dist_1_2(current_state, pos[0], pos[1])
            for n in neighbors:
                if n not in env:
                    relevant.add(n)

        return relevant

    def evaluate_state_cost(self, current_state: GameStateHex) -> float:
        """
        Évalue un état non-terminal en utilisant l'algorithme de Dijkstra pour trouver
        le coût du plus court chemin pour chaque joueur.
        L'heuristique est la différence entre le coût du chemin de l'adversaire et celui du joueur.
        """
        my_path_cost = self.bfs_path_cost(current_state, self.piece_type)
        opponent_path_cost = self.bfs_path_cost(current_state, self.opponent_piece)

        if my_path_cost == 0:
            return 10000
        if opponent_path_cost == 0:
            return -10000

        # Un coût plus faible est meilleur. Nous voulons maximiser (opp_cost^2 - my_cost^2).
        return (opponent_path_cost**2) - (my_path_cost**2)

    def is_adjacent_to_piece(
        self, current_state: GameStateHex, action: LightAction
    ) -> bool:
        """
        Check if the move is adjacent to any existing piece.
        """
        new_pos = action.data["position"]
        row, col = new_pos
        board = current_state.get_rep().get_env()
        neighbors = current_state.get_neighbours(row, col)

        for neighbor_info in neighbors.values():
            neighbor_type, neighbor_pos = neighbor_info
            if neighbor_type != "OUTSIDE" and neighbor_pos in board:
                return True
        return False

    def _initialize_cache(self, board_size: int) -> None:
        """Initialise le cache des sources/destinations pour les deux joueurs."""
        self._board_size = board_size
        # Précalculer pour Red (Vertical: Top-Bottom)
        self._sources_cache["R"] = [(0, j) for j in range(board_size)]
        self._destinations_cache["R"] = {(board_size - 1, j) for j in range(board_size)}
        self._is_horizontal_cache["R"] = False

        # Précalculer pour Blue (Horizontal: Left-Right)
        self._sources_cache["B"] = [(i, 0) for i in range(board_size)]
        self._destinations_cache["B"] = {(i, board_size - 1) for i in range(board_size)}
        self._is_horizontal_cache["B"] = True

    def bfs_path_cost(self, current_state: GameStateHex, piece_type: str) -> float:
        """
        Calcule le coût du plus court chemin pour un joueur donné en utilisant 0-1 BFS.
        """
        board = current_state.get_rep()
        board_size = board.get_dimensions()[0]
        env = board.get_env()

        # Initialiser le cache si nécessaire
        if self._board_size != board_size:
            self._initialize_cache(board_size)

        # Utiliser les valeurs en cache
        sources = self._sources_cache[piece_type]
        destinations = self._destinations_cache[piece_type]

        # Nœuds virtuels
        SOURCE = (-1, -1)
        DEST = (-2, -2)

        # Deque: (node, cost)
        dq = deque([(SOURCE, 0)])
        visited = set()

        while dq:
            node, cost = dq.popleft()

            # Si on a déjà visité ce nœud, on passe
            if node in visited:
                continue
            visited.add(node)

            # Si on atteint la destination virtuelle, on retourne le coût
            if node == DEST:
                return cost

            # Génération des voisins
            neighbors = []
            if node == SOURCE:
                # Depuis la source virtuelle, explorer toutes les cases sources
                for pos in sources:
                    cell = env.get(pos)
                    if cell is None:  # Case vide
                        edge_cost = 1
                    elif cell.get_type() == piece_type:  # Notre pièce
                        edge_cost = 0
                    else:  # Pièce adverse (bloquant)
                        continue
                    neighbors.append((pos, edge_cost))
            else:
                # Nœud normal: explorer les voisins hexagonaux
                neighbor_dict = current_state.get_neighbours(node[0], node[1])

                for neighbor_info in neighbor_dict.values():
                    neighbor_type, neighbor_pos = neighbor_info

                    if neighbor_type == "OUTSIDE":
                        continue

                    # Vérifier si c'est une destination
                    if neighbor_pos in destinations:
                        # Connecter à la destination virtuelle
                        cell = env.get(neighbor_pos)
                        if cell is None:
                            edge_cost = 1
                        elif cell.get_type() == piece_type:
                            edge_cost = 0
                        else:
                            continue
                        neighbors.append((DEST, edge_cost))
                    else:
                        # Voisin normal
                        cell = env.get(neighbor_pos)
                        if cell is None:
                            edge_cost = 1
                        elif cell.get_type() == piece_type:
                            edge_cost = 0
                        else:  # Pièce adverse
                            continue
                        neighbors.append((neighbor_pos, edge_cost))

            # Traiter les voisins
            for next_node, edge_cost in neighbors:
                if next_node not in visited:
                    if edge_cost == 0:
                        dq.appendleft((next_node, cost))
                    else:
                        dq.append((next_node, cost + 1))

        return float("inf")
