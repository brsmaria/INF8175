from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.game.light_action import LightAction
from collections import deque
import random

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
            name (str, optional): Name of the player
        """
        super().__init__(piece_type, name)
        self.max_depth = 4
        self.opponent_piece = "B" if piece_type == "R" else "R"
        self.is_opening_move_called = False

        self._sources_cache = {}
        self._destinations_cache = {}
        self._is_horizontal_cache = {}
        self._board_size = None

        self._neighbors = self.precompute_neighbors(radius=2)

        self.killer_moves = {}

        self._zobrist_table = {}
        self._transposition_table = {}

    def init_zobrist(self, board_size: int):
        """
        Initialize the Zobrist table for board state hashing.
        
        Args:
            board_size (int): The size of the game board
        """
        self._zobrist_table = {}
        for r in range(board_size):
            for c in range(board_size):
                self._zobrist_table[(r, c)] = {
                    "R": random.getrandbits(64),
                    "B": random.getrandbits(64),
                }
        self._transposition_table = {}

    def compute_board_hash(self, current_state: GameStateHex) -> int:
        """
        Computes the Zobrist hash of the current board state.
        
        Args:
            current_state (GameStateHex): The current game state
            
        Returns:
            int: The Zobrist hash of the board state
        """
        h = 0
        board = current_state.get_rep().get_env()
        for pos, piece in board.items():
            piece_type = piece.get_type()
            if piece_type in ["R", "B"]:
                h ^= self._zobrist_table[pos][piece_type]
        return h

    def compute_action(
        self, current_state: GameState, remaining_time: int = 1e9, **kwargs
    ) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state
            remaining_time (int, optional): Remaining time for the move. Defaults to 1e9

        Returns:
            Action: The best action as determined by minimax
        """
        env = current_state.rep.env
        if len(env) <= 1:
            return self.opening_strategy(current_state)

        board_size = current_state.rep.get_dimensions()[0]
        if not self._zobrist_table:
            self.init_zobrist(board_size)

        current_hash = self.compute_board_hash(current_state)

        return self.minimax_search(current_state, current_hash)

    def opening_strategy(self, current_state: GameStateHex) -> LightAction:
        """
        Opening strategy for the first move.
        
        Args:
            current_state (GameStateHex): The current game state
            
        Returns:
            LightAction: The opening move action
        """
        env = current_state.rep.env
        board_size = current_state.rep.get_dimensions()[0]
        center = board_size // 2 - 1

        center_positions = [
            (center, center),
            (center, center + 1),
            (center + 1, center),
            (center + 1, center + 1),
        ]

        row, col = center_positions[0] 

        for pos in center_positions:
            if pos not in env:
                row, col = pos
                break

        return LightAction({"piece": self.piece_type, "position": (row, col)})

    def minimax_search(self, current_state: GameState, current_hash: int) -> Action:
        """
        Minimax search with alpha-beta pruning to find the best action.
        
        Args:
            current_state (GameState): The current game state
            current_hash (int): The Zobrist hash of the current state
            
        Returns:
            Action: The best action found by minimax
        """
        _, best_action = self.max_value(
            current_state, self.max_depth, float("-inf"), float("inf"), current_hash
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
        current_hash: int,
    ):
        """
        Maximizer node in the minimax algorithm with alpha-beta pruning.
        
        Args:
            current_state (GameState): The current game state
            depth (int): Remaining depth to search
            alpha (float): Alpha value for alpha-beta pruning
            beta (float): Beta value for alpha-beta pruning
            current_hash (int): The Zobrist hash of the current state
            
        Returns:
            tuple: (score, best_action) where score is the evaluation and best_action is the best move
        """
        if current_hash in self._transposition_table:
            tt_score, tt_depth, tt_flag, tt_move = self._transposition_table[current_hash]
            if tt_depth >= depth:
                if tt_flag == 0: 
                    return (tt_score, tt_move)
                elif tt_flag == 1:  
                    alpha = max(alpha, tt_score)
                elif tt_flag == 2: 
                    beta = min(beta, tt_score)

                if alpha >= beta:
                    return (tt_score, tt_move)

        if current_state.is_done():
            score = current_state.get_player_score(self)
            return (10000 if score == 1 else -10000, None)

        if depth == 0:
            val = self.evaluate_state_cost(current_state)
            self._transposition_table[current_hash] = (val, depth, 0, None)
            return (val, None)

        current_score: float = float("-inf")
        best_action: LightAction | None = None

        possible_actions = list(current_state.get_possible_light_actions())
        actions_with_priority = []

        killer_move = self.killer_moves.get(depth)
        tt_move = None
        if current_hash in self._transposition_table:
            tt_move = self._transposition_table[current_hash][3]

        for action in possible_actions:
            priority = 1  # Default

            # TT Move
            if tt_move and action.data["position"] == tt_move.data["position"]:
                priority = -2  # Highest
            # Killer Heuristic
            elif (
                killer_move and action.data["position"] == killer_move.data["position"]
            ):
                priority = -1  # Highest
            # Proximity Sorting
            elif self.is_relevant(current_state, action):
                priority = 0  # High
            else:
                continue  # Skip non-relevant moves

            actions_with_priority.append((priority, action))

        actions_with_priority.sort(key=lambda x: x[0])

        original_alpha = alpha

        for _, action in actions_with_priority:
            potential_state = current_state.apply_action(action)

            pos = action.data["position"]
            new_hash = current_hash ^ self._zobrist_table[pos][self.piece_type]

            if potential_state.is_done():
                score = potential_state.get_player_score(self)
                return (10000 if score == 1 else -10000, action)

            potential_score, _ = self.min_value(
                potential_state, depth - 1, alpha, beta, new_hash
            )

            if potential_score > current_score:
                current_score = potential_score
                best_action = action

            alpha = max(alpha, current_score)
            if current_score >= beta:
                self.killer_moves[depth] = action
                break

        flag = 0
        if current_score <= original_alpha:
            flag = 2 
        elif current_score >= beta:
            flag = 1 
        else:
            flag = 0

        self._transposition_table[current_hash] = (
            current_score,
            depth,
            flag,
            best_action,
        )

        return (current_score, best_action)

    def min_value(
        self,
        current_state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        current_hash: int,
    ):
        """
        Minimizer node in the minimax algorithm with alpha-beta pruning.
        
        Args:
            current_state (GameState): The current game state
            depth (int): Remaining depth to search
            alpha (float): Alpha value for alpha-beta pruning
            beta (float): Beta value for alpha-beta pruning
            current_hash (int): The Zobrist hash of the current state
            
        Returns:
            tuple: (score, best_action) where score is the evaluation and best_action is the best move
        """
        # TT Lookup
        if current_hash in self._transposition_table:
            tt_score, tt_depth, tt_flag, tt_move = self._transposition_table[
                current_hash
            ]
            if tt_depth >= depth:
                if tt_flag == 0: 
                    return (tt_score, tt_move)
                elif tt_flag == 1:  
                    alpha = max(alpha, tt_score)
                elif tt_flag == 2: 
                    beta = min(beta, tt_score)

                if alpha >= beta:
                    return (tt_score, tt_move)

        if current_state.is_done():
            score = current_state.get_player_score(self)
            return (10000 if score == 1 else -10000, None)

        if depth == 0:
            val = self.evaluate_state_cost(current_state)
            self._transposition_table[current_hash] = (val, depth, 0, None)
            return (val, None)

        current_score: float = float("inf")
        best_action: LightAction | None = None

        possible_actions = list(current_state.get_possible_light_actions())
        actions_with_priority = []

        killer_move = self.killer_moves.get(depth)
        tt_move = None
        if current_hash in self._transposition_table:
            tt_move = self._transposition_table[current_hash][3]

        for action in possible_actions:
            priority = 1 

            # TT Move
            if tt_move and action.data["position"] == tt_move.data["position"]:
                priority = -2  # Highest
            # Killer Heuristic
            elif (
                killer_move and action.data["position"] == killer_move.data["position"]
            ):
                priority = -1  # Highest
            # Proximity Sorting
            elif self.is_relevant(current_state, action):
                priority = 0  # High
            else:
                continue  # Skip non-relevant moves

            actions_with_priority.append((priority, action))

        actions_with_priority.sort(key=lambda x: x[0])

        original_beta = beta

        for _, action in actions_with_priority:
            potential_state = current_state.apply_action(action)

            pos = action.data["position"]
            new_hash = current_hash ^ self._zobrist_table[pos][self.opponent_piece]

            if potential_state.is_done():
                score = potential_state.get_player_score(self)
                return (10000 if score == 1 else -10000, action)

            potential_score, _ = self.max_value(
                potential_state, depth - 1, alpha, beta, new_hash
            )

            if potential_score < current_score:
                current_score = potential_score
                best_action = action

            beta = min(beta, current_score)
            if current_score <= alpha:
                # Killer Heuristic: Store the move that caused the cutoff
                self.killer_moves[depth] = action
                break

        flag = 0
        if current_score <= alpha:
            flag = 2  # UPPERBOUND
        elif current_score >= original_beta:
            flag = 1  # LOWERBOUND
        else:
            flag = 0  # EXACT

        self._transposition_table[current_hash] = (
            current_score,
            depth,
            flag,
            best_action,
        )

        return (current_score, best_action)

    def evaluate_state_cost(self, current_state: GameStateHex) -> float:
        """
        Evaluates a non-terminal state using BFS algorithm to find
        the shortest path cost for each player.
        The heuristic is the difference between the opponent's path cost and the player's path cost.
        
        Args:
            current_state (GameStateHex): The current game state
            
        Returns:
            float: The heuristic evaluation of the state
        """
        my_path_cost = self.bfs_path_cost(current_state, self.piece_type)
        opponent_path_cost = self.bfs_path_cost(current_state, self.opponent_piece)

        if my_path_cost == 0:
            return 10000
        if opponent_path_cost == 0:
            return -10000

        return (opponent_path_cost**2) - (my_path_cost**2)

    def is_relevant(self, current_state: GameStateHex, action: LightAction) -> bool:
        """
        Checks if the move is relevant by being near an existing piece.
        
        Args:
            current_state (GameStateHex): The current game state
            action (LightAction): The action to evaluate
            
        Returns:
            bool: True if the move is relevant, False otherwise
        """
        new_pos = action.data["position"]
        row, col = new_pos
        board = current_state.get_rep().get_env()

        for dr, dc in self._neighbors:
            neighbor_pos = (row + dr, col + dc)
            if neighbor_pos in board:
                return True
        return False

    def _initialize_cache(self, board_size: int) -> None:
        """
        Initialize the cache of sources/destinations for both players.
        
        Args:
            board_size (int): The size of the game board
        """
        self._board_size = board_size
        # Precompute for Red (Vertical: Top-Bottom)
        self._sources_cache["R"] = [(0, j) for j in range(board_size)]
        self._destinations_cache["R"] = {(board_size - 1, j) for j in range(board_size)}
        self._is_horizontal_cache["R"] = False

        # Precompute for Blue (Horizontal: Left-Right)
        self._sources_cache["B"] = [(i, 0) for i in range(board_size)]
        self._destinations_cache["B"] = {(i, board_size - 1) for i in range(board_size)}
        self._is_horizontal_cache["B"] = True

    def bfs_path_cost(self, current_state: GameStateHex, piece_type: str) -> float:
        """
        Calculates the shortest path cost for a given player using 0-1 BFS.
        
        Args:
            current_state (GameStateHex): The current game state
            piece_type (str): The piece type to calculate path cost for ("R" or "B")
            
        Returns:
            float: The shortest path cost for the player
        """
        board = current_state.get_rep()
        board_size = board.get_dimensions()[0]
        env = board.get_env()

        if self._board_size != board_size:
            self._initialize_cache(board_size)

        sources = self._sources_cache[piece_type]
        destinations = self._destinations_cache[piece_type]

        # Virtual nodes
        SOURCE = (-1, -1)
        DEST = (-2, -2)

        dq = deque([(SOURCE, 0)])
        visited = set()

        while dq:
            node, cost = dq.popleft()

            if node in visited:
                continue
            visited.add(node)

            if node == DEST:
                return cost

            neighbors = []
            if node == SOURCE:
                for pos in sources:
                    cell = env.get(pos)
                    if cell is None:  # Empty cell
                        edge_cost = 1
                    elif cell.get_type() == piece_type:
                        edge_cost = 0
                    else: 
                        continue
                    neighbors.append((pos, edge_cost))
            else:
                neighbor_dict = current_state.get_neighbours(node[0], node[1])

                for neighbor_info in neighbor_dict.values():
                    neighbor_type, neighbor_pos = neighbor_info

                    if neighbor_type == "OUTSIDE":
                        continue

                    if neighbor_pos in destinations:
                        cell = env.get(neighbor_pos)
                        if cell is None:
                            edge_cost = 1
                        elif cell.get_type() == piece_type:
                            edge_cost = 0
                        else:
                            continue
                        neighbors.append((DEST, edge_cost))
                    else:
                        cell = env.get(neighbor_pos)
                        if cell is None:
                            edge_cost = 1
                        elif cell.get_type() == piece_type:
                            edge_cost = 0
                        else:
                            continue
                        neighbors.append((neighbor_pos, edge_cost))

            for next_node, edge_cost in neighbors:
                if next_node not in visited:
                    if edge_cost == 0:
                        dq.appendleft((next_node, cost))
                    else:
                        dq.append((next_node, cost + 1))

        return float("inf")

    def _precompute_neighbors(self, radius: int):
        """
        Precomputes neighbor offsets within a given radius.
        
        Args:
            radius (int): The radius within which to compute neighbors
            
        Returns:
            list: List of (dr, dc) tuples representing neighbor offsets
        """
        neighbors = []

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr + dc) <= radius and (dr != 0 or dc != 0):
                    neighbors.append((dr, dc))

        return neighbors
