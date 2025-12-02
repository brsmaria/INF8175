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
        self._max_depth = 3
        self.last_d_me = 14
        self.end_game = False

        self.next_light_action: LightAction | None = None
        self.last_opp_action: LightAction | None = None
        self.last_game_state: GameStateHex | None = None
        


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

    def _shannon_path_with_path(self, state: GameStateHex, piece: str) -> tuple[int, set[tuple[int, int]]]:
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

        return (best_distance, empties, path)
    
    def _shannon_path_empty_cells(self, state: GameStateHex, piece: str) -> tuple[int, set[tuple[int, int]]]:
        d, empties, _ = self._shannon_path_with_path(state, piece)
        return d, empties

    def _order_actiosns_finishers_first(
            self,
            state: GameStateHex,
            actions: list[HeavyAction],
            piece: str
        ) -> list[HeavyAction]:

        if not actions:
            return actions

        before_keys = set(state.rep.env.keys())

        scored_actions = []  # list of (score, heavy_action)
        action_score = 0
        for heavy_action in actions:
            next_state = heavy_action.get_next_game_state()
            action_score = self.evaluate(next_state, before_keys, piece)

            # garder seulement coups intéressants
            if action_score <= 0:
                continue

            scored_actions.append((action_score, heavy_action))

        # si rien de filtré, retourne 10 premiers coups bruts
        if not scored_actions:
            return actions[:10]

        # --- TRI PAR SCORE DECROISSANT ---
        scored_actions.sort(key=lambda x: x[0], reverse=True)

        # --- GARDER SEULEMENT LES 10 MEILLEURS ---
        top10 = [ha for (_, ha) in scored_actions[:10]]

        return top10



    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        print('------------------------------------------------------------------------------')
        if self.is_opening_move(current_state):
            return self.opening_strategy(current_state)
        elif self.end_game:
            return self.end_game_strategy(current_state)

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

            center_positions = [(center, center), (center, center + 1),
                                (center + 1, center), (center + 1, center + 1)]

            row, col = center_positions[0]  # Default fallback

            for pos in center_positions:
                if pos not in env:
                    row, col = pos
                    break

            return LightAction({"piece": self.piece_type, "position": (row, col)})

    def minimax_search(self, current_state: GameState, alpha: int = -inf, beta: int = inf) -> Action:
        _, best_heavy_action = self.max_value(current_state, self._max_depth, alpha, beta)
        return current_state.convert_heavy_action_to_light_action(best_heavy_action)


    def max_value(self, current_state: GameState, depth: int, alpha: int, beta: int):
        if current_state.is_done():
            return (current_state.get_player_score(self), None)
        if depth <= 0:
            return (self.heuristic_evalutation(
                current_state, set(current_state.rep.env.keys())
            ), None)

        best_score: float = -inf
        best_heavy_action: HeavyAction | None = None

        for heavy_action in self._order_actions_finishers_first(
                current_state,
                list(current_state.generate_possible_heavy_actions()),
                self.piece_type):
            if heavy_action.get_next_game_state().is_done():
                return (current_state.get_player_score(self), heavy_action)
            self.next_light_action = current_state.convert_heavy_action_to_light_action(heavy_action)
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
        # if depth == 0:
        #     return (self.heuristic_evalutation(
        #         current_state, set(current_state.rep.env.keys())
        #     ), None)

        best_score: float = inf
        best_heavy_action: HeavyAction | None = None

        for heavy_action in self._order_actions_finishers_first(
                current_state,
                list(current_state.generate_possible_heavy_actions()),
                "B" if self.piece_type == "R" else "R"):  # on privilégie les "finishers" de l’adversaire pour les bloquer
            if heavy_action.get_next_game_state().is_done():
                return (current_state.get_player_score(self), heavy_action)
            self.next_light_action = current_state.convert_heavy_action_to_light_action(heavy_action)
            next_state = heavy_action.get_next_game_state()
            score, _ = self.max_value(next_state, depth - 1, alpha, beta)
            if score < best_score:
                best_score = score
                best_heavy_action = heavy_action
                beta = min(beta, best_score)
            if beta <= alpha:
                break
        return (best_score, best_heavy_action)

    def heuristic_evalutation(self, state: GameStateHex, before_keys) -> float:
        my_piece  = self.piece_type
        opp_piece = "B" if my_piece == "R" else "R"

        dim = state.get_rep().get_dimensions()[0]

        d_me,  _ = self._shannon_path_empty_cells(state, my_piece)
        d_opp, _ = self._shannon_path_empty_cells(state, opp_piece)

        if d_me  >= 10**9: d_me  = dim * 5
        if d_opp >= 10**9: d_opp = dim * 5

        shannon_score = d_opp - d_me          # positif si je suis mieux

        if d_me < self.last_d_me :
            shannon_score += 50
            self.last_d_me = d_me
        move_score    = self.evaluate(state, before_keys, my_piece)

        if d_me <=3: 
            shannon_score += 50
            self.end_game = True
        else:
            self.end_game = False

        w1 = 0.7  # poids du terme global (Shannon)
        w2 = 0.3  # poids du terme local (patterns)
        return w1 * shannon_score + w2 * move_score


    def _order_actions_finishers_first(
            self,
            state: GameStateHex,
            actions: list[HeavyAction],
            piece: str
        ) -> list[HeavyAction]:

        if not actions:
            return actions

        before_keys = set(state.rep.env.keys())

        scored_actions: list[tuple[float, HeavyAction]] = []

        for heavy_action in actions:
            next_state = heavy_action.get_next_game_state()

            if next_state.is_done():
                return [heavy_action]

            score = self.evaluate(next_state, before_keys, piece)

            # On garde seulement les coups vraiment intéressants
            if score <= 0:
                continue

            scored_actions.append((score, heavy_action))

        # si rien de filtré, retourne au moins quelques coups bruts
        if not scored_actions:
            return actions[:5]

        # tri par score décroissant
        scored_actions.sort(key=lambda x: x[0], reverse=True)

        # top 10
        return [ha for (s, ha) in scored_actions[:10]]


    def evaluate(self, next_state: GameState, before_keys, piece: str) -> float:
        bridge_offsets = [
            ((-2,  1), (-1, 0), (-1, 1)),
            ((-1,  2), (-1, 1), ( 0, 1)),
            ((-1, -1), (-1, 0), ( 0,-1)),
            (( 1, -2), ( 0,-1), ( 1,-1)),
            (( 2, -1), ( 1,-1), ( 1, 0)),
            (( 1,  1), ( 0, 1), ( 1, 0)),
        ]

        neighbor_offsets = [
            (-1,  0), (-1,  1),
            ( 0, -1), ( 0,  1),
            ( 1, -1), ( 1,  0),
        ]

        middle_pairs_for_bridge = [
            ((-1,0),(1,-1),(0,-1)),
            ((-1,1),(1,0),(0,1)),
            ((-1,0),(0,1),(-1,1)),
            ((-1,1),(0,-1),(-1,0)),
            ((1,0),(0,-1),(1,-1)),
            ((1,-1),(0,1),(1,0)),
        ]

        useless_flower_offset_R = [
            ((-1, -1), (0,-2), (1, -2)),
            ((-1, 2), (0, 2), (1, 1)),
        ]

        useless_flower_offset_B = [
            ((-2, 0), (-2, 1), (-2, 2)),
            ((2, -2), (2, -1), (2, 0)),
        ]


        dim = next_state.get_rep().get_dimensions()[0]
        my_piece  = piece
        opp_piece = "B" if my_piece == "R" else "R"
        
        useless_flower_offset = useless_flower_offset_R if my_piece == "R" else useless_flower_offset_B
        env_ns = next_state.rep.env

        # retrouver la case jouée :
        after_keys = set(env_ns.keys())
        diff = list(after_keys - before_keys)
        if not diff:
            # sécurité : si on ne trouve pas le coup, on considère neutre
            return 0.0

        i, j = diff[0]
        action_score = 0.0

        # triangle filter
        empty_neighbors = 0
        opp_neighbors = 0
        for di, dj in neighbor_offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < dim and 0 <= nj < dim:
                p = env_ns.get((ni, nj))
                if p is None:
                    empty_neighbors += 1
                elif p.get_type() == opp_piece:
                    opp_neighbors += 1

        if empty_neighbors < 2:
            action_score -= 50

        # Malus si on s'expose à trop d'adversaires autour
        if opp_neighbors >= 3:
            action_score -= 50

        # useless flower
        for side in useless_flower_offset:
            useless_flower_count = 0
            for di, dj in side:
                ni, nj = i + di, j + dj
                if 0 <= ni < dim and 0 <= nj < dim:
                    p = env_ns.get((ni, nj))
                    if p and p.get_type() == my_piece:
                        useless_flower_count +=1
            if useless_flower_count >= 2:
                action_score -=20

        # bridge / block patterns
        bridge_bonus = 0
        block_bonus  = 0
        complete_bridge_bonus = 0

        for (di1, dj1), (di2, dj2), (di3, dj3) in bridge_offsets:
            ni1, nj1 = i + di1, j + dj1
            ni2, nj2 = i + di2, j + dj2
            ni3, nj3 = i + di3, j + dj3

            if 0 <= ni1 < dim and 0 <= nj1 < dim:
                p1 = env_ns.get((ni1, nj1))
                p2 = env_ns.get((ni2, nj2))
                p3 = env_ns.get((ni3, nj3))

                if p1 and (p2 is None) and (p3 is None):
                    if p1.get_type() == my_piece:
                        bridge_bonus += 25
                    elif p1.get_type() == opp_piece:
                        block_bonus += 15
                    break
                elif p1 and p2 and (p3 is None ):
                    if p1.get_type() == my_piece and p2.get_type() == opp_piece:
                        block_bonus -= 20

                elif p1 and p3 and (p2 is None ):
                    if p1.get_type() == my_piece and p3.get_type() == opp_piece:
                        block_bonus -= 20
                

        for (di1,dj1),(di2,dj2),(di3,dj3) in middle_pairs_for_bridge:
            ni1, nj1 = i + di1, j + dj1
            ni2, nj2 = i + di2, j + dj2
            ni3, nj3 = i + di3, j + dj3

            if not (0 <= ni1 < dim and 0 <= nj1 < dim and
                    0 <= ni2 < dim and 0 <= nj2 < dim):
                continue

            p1 = env_ns.get((ni1, nj1))
            p2 = env_ns.get((ni2, nj2))
            p3 = env_ns.get((ni3, nj3))

            # bridge à moi
            if p1 and p2 and p1.get_type()==my_piece and p2.get_type()==my_piece:
                if p3 is None:
                    complete_bridge_bonus += 25
                elif p3.get_type() == opp_piece:
                    complete_bridge_bonus += 50  # URGENT à défendre / exploiter

            # bridge adverse
            if p1 and p2 and p1.get_type()== opp_piece and p2.get_type()==opp_piece:
                if p3 is None or p3.get_type() == opp_piece:
                    complete_bridge_bonus -= 40  # très dangereux
                elif p3.get_type() == my_piece:
                    complete_bridge_bonus  += 25  # bon coup de blocage

        action_score += bridge_bonus + block_bonus + complete_bridge_bonus
        
        print(f'next move {i} {j}')
        print(action_score)
        return action_score

    def end_game_strategy(self, current_state: GameStateHex) -> LightAction:
        """
        Stratégie de fin de partie :
        - parcourt tous les coups possibles
        - évalue pour chacun :
            * la distance de Shannon (d_me, d_opp)
            * les patterns locaux (bridges, blocages) via evaluate
        - choisit le coup avec le meilleur score.
        """

        my_piece  = self.piece_type
        opp_piece = "B" if my_piece == "R" else "R"
        dim       = current_state.get_rep().get_dimensions()[0]

        env = current_state.rep.env

        total_cells       = dim * dim
        remaining_empties = total_cells - len(env)
        print(f"[ENDGAME] Cases vides restantes : {remaining_empties}")

        before_keys = set(env.keys())

        best_score: float = -inf
        best_heavy_action: HeavyAction | None = None

        for heavy_action in current_state.generate_possible_heavy_actions():
            next_state = heavy_action.get_next_game_state()

            if next_state.is_done():
                print("[ENDGAME] Coup gagnant trouvé.")
                return current_state.convert_heavy_action_to_light_action(heavy_action)

            d_me,  _, my_path  = self._shannon_path_with_path(next_state, my_piece)
            d_opp, _, opp_path = self._shannon_path_with_path(next_state, opp_piece)

            if d_me  >= 10**9: d_me  = dim * 5
            if d_opp >= 10**9: d_opp = dim * 5

            shannon_score = d_opp - d_me   # > 0 si on est mieux

            local_score = self.evaluate(next_state, before_keys, my_piece)

            # >>>>> NOUVEAU : bonus de bridge autour de my_path[-1]
            bridge_last_path_bonus = self._bridge_bonus_last_path(
                next_state,
                before_keys,
                my_piece,
                my_path
            )

            w_shannon = 1.0
            w_local   = 0.4

            total_score = (
                w_shannon * shannon_score +
                w_local   * local_score +
                bridge_last_path_bonus
            )

            print("[ENDGAME] coup testé, score =", total_score)

            if total_score > best_score:
                best_score = total_score
                best_heavy_action = heavy_action

        if best_heavy_action is None:
            print("[ENDGAME] Aucun coup choisi, fallback minimax.")
            _, best_heavy_action = self.max_value(current_state, 1, -inf, inf)

        return current_state.convert_heavy_action_to_light_action(best_heavy_action)



    def _bridge_bonus_last_path(
        self,
        next_state: GameStateHex,
        before_keys: set[tuple[int, int]],
        my_piece: str,
        my_path: list[tuple[int, int]]
    ) -> float:
        """
        Donne un bonus si le coup joué :
        - forme un bridge avec la dernière case du chemin my_path[-1]
        - et éventuellement 'complète' un bridge formé par my_path[-1] avec une autre de mes pierres du chemin.
        """

        if not my_path:
            return 0.0

        env_ns = next_state.rep.env
        after_keys = set(env_ns.keys())
        diff = list(after_keys - before_keys)
        if not diff:
            return 0.0

        # case jouée par ce HeavyAction
        i, j = diff[0]

        pivot_i, pivot_j = my_path[-1]

        # vecteurs de bridge (distance entre deux extrémités de bridge)
        bridge_vectors = [
            (-2,  1), (-1,  2),
            (-1, -1), ( 1,  1),
            ( 1, -2), ( 2, -1),
        ]

        # 1) Bonus si la case jouée est en relation de bridge avec my_path[-1]
        di = pivot_i - i
        dj = pivot_j - j

        bonus = 0.0

        pivot_p = env_ns.get((pivot_i, pivot_j))
        played_p = env_ns.get((i, j))

        if (di, dj) in bridge_vectors and pivot_p and played_p:
            if pivot_p.get_type() == my_piece and played_p.get_type() == my_piece:
                # bridge direct entre le coup et my_path[-1]
                bonus += 30.0  # à ajuster selon l'impact souhaité

                # 2) Bonus supplémentaire si my_path[-1] forme aussi un bridge avec
                #    une autre de mes pierres sur le chemin -> on interprète ça
                #    comme "compléter" un pattern autour de my_path[-1].
                for qi, qj in my_path[:-1]:
                    dqi = qi - pivot_i
                    dqj = qj - pivot_j
                    if (dqi, dqj) in bridge_vectors:
                        qp = env_ns.get((qi, qj))
                        if qp and qp.get_type() == my_piece:
                            bonus += 20.0  # "compléter un bridge" autour de my_path[-1]
                            break

        return bonus
