from collections import deque
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

    DIRECTIONS = ("top_right", "top_left", "right", "left", "bot_right", "bot_left")
        
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)
        self._max_depth = 1

    def _in_bounds(self, i: int, j: int, dim: int) -> bool:
        return 0 <= i < dim and 0 <= j < dim

    def _cell_cost(self, state: GameStateHex, piece: str, i: int, j: int) -> int:
        """
        Coût 0 pour mes pierres, 1 pour case vide, très grand pour pierre adverse.
        """
        p = state.rep.env.get((i, j))
        if p is None:
            return 1
        t = p.get_type()
        if t == piece:
            return 0
        if t in ("R", "B"):
            return 10**9  # mur
        return 1

    def _sources_targets_for(self, piece: str, dim: int):
        #TODO: vérifier que c'est toujours le cas : Rouge de haut en bas et bleu de gache 
        if piece == "R":
            sources = [(0, j) for j in range(dim)]
            targets = [(dim - 1, j) for j in range(dim)]
        else:  # "B"
            sources = [(i, 0) for i in range(dim)]
            targets = [(i, dim - 1) for i in range(dim)]
        return sources, targets

    def _shannon_path_empty_cells(self, state: GameStateHex, piece: str) -> tuple[int, set[tuple[int, int]]]:
        """
        0–1 BFS (coût 0 sur 'piece', 1 sur vides, inf sur adversaire).
        Retourne (distance minimale, ensemble des cases vides appartenant à UN chemin minimal).
        Si aucun chemin n'est possible sans traverser l'adversaire, distance = +inf et set vide.
        """
        dim = state.get_rep().get_dimensions()[0]

        #source et target sont des listes avec les cordonéées de la ligne de départ et celle d'arrivée
        sources, targets = self._sources_targets_for(piece, dim)

        dist = { (i, j): inf for i in range(dim) for j in range(dim) }
        prev: dict[tuple[int,int], tuple[int,int]] = {} # pour remonter un chemin minimal

        dq = deque()
        
        #déterminer s'il y a des pièces dans la première rangée (départ))
        for s in sources:
            c = self._cell_cost(state, piece, s[0], s[1])
            if c < inf:
                dist[s] = c
                dq.appendleft(s) # on a trouvé une source valide

        while dq: #on avance jusqu'à être boquées par le joueur adverse ou atteindre la fin
            current_i, current_j = dq.popleft()
            neigh = state.get_neighbours(current_i, current_j)
            for k in self.DIRECTIONS:
                _, (neigh_i, neigh_j) = neigh[k]
                if not self._in_bounds(neigh_i, neigh_j, dim):
                    continue
                w = self._cell_cost(state, piece, neigh_i, neigh_j)
                if w >= inf:
                    continue
                distance_from_source = dist[(current_i, current_j)] + w
                if distance_from_source < dist[(neigh_i, neigh_j)]:
                    dist[(neigh_i, neigh_j)] = distance_from_source
                    prev[(neigh_i, neigh_j)] = (current_i, current_j)
                    # 0–1 BFS: coût 0 en tête, coût 1 en queue => les voisins avec ma case sont évalués en premier
                    if w == 0:
                        dq.appendleft((neigh_i, neigh_j))
                    else:
                        dq.append((neigh_i, neigh_j))

        # choisir la meilleure target atteinte
        # la distance ici est avec les poids mis en place (0, 1, inf)
        best_target, best_distance = None, inf
        for t in targets:
            if dist.get(t, inf) < best_distance:
                best_distance, best_target = dist[t], t

        if best_distance >= inf or best_target is None: #aucun chemin trouvé
            return (inf, set())

        # remonter UN chemin minimal
        path = []
        current_cell = best_target
        while current_cell in prev:
            path.append(current_cell)
            current_cell = prev[current_cell]
        path.append(current_cell) #source
        path.reverse()

        # extraire les cases vides du chemin -> les cases qu'on veut remplir en priorité
        empties = set()
        for (i, j) in path:
            if state.rep.env.get((i, j)) is None:
                empties.add((i, j))

        return (best_distance, empties)

    def _order_actions_finishers_first(self, state: GameStateHex, actions: list[HeavyAction], piece: str) -> list[HeavyAction]:
        """
        Trie les actions en plaçant d’abord celles qui jouent sur une case vide
        d’un chemin minimal (Shannon) pour le joueur `piece`.
        """
        _, finishers = self._shannon_path_empty_cells(state, piece)
        if not finishers:
            return actions  # rien de spécial à privilégier

        ordered = []
        tail = []
        for ha in actions:
            ns = ha.get_next_game_state()
            # détecter la case jouée: différence entre env avant/après (une seule case en plus)
            # (robuste si l’API n’expose pas directement la coord du coup)
            before = set(state.rep.env.keys())
            after = set(ns.rep.env.keys())
            placed = list(after - before)
            if placed and placed[0] in finishers:
                ordered.append(ha)
            else:
                tail.append(ha)
        return ordered + tail

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        return self.minimax_search(current_state)

    def minimax_search(self, current_state: GameState, alpha: int = -inf, beta: int = inf) -> Action:
        _, best_heavy_action = self.max_value(current_state, self._max_depth, alpha, beta)
        return current_state.convert_heavy_action_to_light_action(best_heavy_action)

    def max_value(self, current_state: GameState, depth: int, alpha: int, beta: int):
        if current_state.is_done():
            return (current_state.get_player_score(self), None)
        if depth == 0:
            return (self.heuristic_evalutation(current_state), None)

        best_score: float = -inf
        best_heavy_action: HeavyAction | None = None

        for heavy_action in self._order_actions_finishers_first(
                current_state,
                list(current_state.generate_possible_heavy_actions()),
                self.piece_type):
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

        for heavy_action in self._order_actions_finishers_first(
                current_state,
                list(current_state.generate_possible_heavy_actions()),
                "B" if self.piece_type == "R" else "R"):  # on privilégie les "finishers" de l’adversaire pour les bloquer
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
        my_piece  = self.piece_type
        opp_piece = "B" if my_piece == "R" else "R"

        env = state.rep.env
        my_count  = sum(1 for p in env.values() if p.get_type() == my_piece)
        opp_count = sum(1 for p in env.values() if p.get_type() == opp_piece)

        d_me, _  = self._shannon_path_empty_cells(state, my_piece)
        d_opp, _ = self._shannon_path_empty_cells(state, opp_piece)

        # si pas de chemin (bloqué par murs), traite comme très défavorable/favorable
        if d_me >= 10**9: d_me = 1000
        if d_opp >= 10**9: d_opp = 1000


        return d_opp - d_me 


    