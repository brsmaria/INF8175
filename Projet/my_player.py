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
        self._extension_threshold = 5 # à ajuster après tests
        self._tt = {}
        self._shannon_cache = {}

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

    def _order_actions_finishers_first(
        self,
        state: GameStateHex,
        actions: list[HeavyAction],
        piece: str
    ) -> list[HeavyAction]:
        """
        Filtre les actions :
          - enlève les triangles (<= 2 voisins vides)
          - enlève les bridge peeps (ma pierre entre deux pierres adverses)
          - garde seulement :
                * les coups dans 'finishers' (chemin de Shannon pour `piece`)
                * OU ceux qui créent un bridge, bloquent un bridge adverse
                  ou complètent un bridge pour `piece`.

        Les actions retenues sont ensuite ordonnées :
          - d'abord celles qui jouent dans 'finishers'
          - puis les autres coups "intéressants".
        Si tout est filtré, on renvoie la liste originale pour ne pas tuer la recherche.
        """
        dim = state.get_rep().get_dimensions()[0]

        # perspective
        my_piece  = piece
        opp_piece = "B" if my_piece == "R" else "R"

        # cases vides d'un chemin minimal pour `piece`
        _, finishers = self._shannon_path_empty_cells(state, my_piece)

        if not actions:
            return actions

        # offsets identiques à ceux de ta heuristique
        bridge_offsets = [
            (-2,  1),
            (-1,  2),
            (-1, -1),
            ( 1, -2),
            ( 2, -1),
            ( 1,  1),
        ]

        neighbor_offsets = [
            (-1,  0),
            (-1,  1),
            ( 0, -1),
            ( 0,  1),
            ( 1, -1),
            ( 1,  0),
        ]

        middle_pairs = [
            ((-1,  0), ( 1, -1)),  # (i-1, j)   et (i+1, j-1)
            ((-1,  1), ( 1,  0)),  # (i-1, j+1) et (i+1, j)
            (( 0,  1), ( 0, -1)),  # (i, j+1)   et (i, j-1)
        ]

        before_keys = set(state.rep.env.keys())

        finishers_actions: list[HeavyAction] = []
        other_good_actions: list[HeavyAction] = []

        for ha in actions:
            ns = ha.get_next_game_state()
            env_ns = ns.rep.env

            # 1) retrouver la case jouée
            after_keys = set(env_ns.keys())
            diff = list(after_keys - before_keys)
            if not diff:
                # sécurité, on ne s'attend pas à ça mais on ne crash pas
                continue
            i, j = diff[0]

            # 2) filtrer les triangles (<= 2 voisins vides)
            empty_neighbors = 0
            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                if not self._in_bounds(ni, nj, dim):
                    continue
                if env_ns.get((ni, nj)) is None:
                    empty_neighbors += 1
            if empty_neighbors <= 2:
                # on enlève ce coup : triangle / coup très enfermé
                continue

            # 3) filtrer les bridge peeps :
            #    ma pierre au milieu de deux pierres adverses
            is_bridge_peep = False
            for (di1, dj1), (di2, dj2) in middle_pairs:
                ni1, nj1 = i + di1, j + dj1
                ni2, nj2 = i + di2, j + dj2

                if not (self._in_bounds(ni1, nj1, dim) and self._in_bounds(ni2, nj2, dim)):
                    continue

                p1 = env_ns.get((ni1, nj1))
                p2 = env_ns.get((ni2, nj2))

                if (
                    p1 is not None
                    and p2 is not None
                    and p1.get_type() == opp_piece
                    and p2.get_type() == opp_piece
                ):
                    is_bridge_peep = True
                    break

            if is_bridge_peep:
                # on jette les bridge peeps
                continue

            # 4) calculer les patterns intéressants :
            #    bridge pour moi, blocage de bridge adverse, complete bridge
            bridge_bonus = 0
            block_bridge_bonus = 0
            complete_bridge_bonus = 0

            # 4a) bridges
            for di, dj in bridge_offsets:
                ni, nj = i + di, j + dj
                if not self._in_bounds(ni, nj, dim):
                    continue
                p = env_ns.get((ni, nj))
                if p is None:
                    continue
                t = p.get_type()
                if t == my_piece:
                    bridge_bonus += 3
                elif t == opp_piece:
                    block_bridge_bonus += 3

            # 4b) complete bridge (pour `my_piece`)
            for (di1, dj1), (di2, dj2) in middle_pairs:
                ni1, nj1 = i + di1, j + dj1
                ni2, nj2 = i + di2, j + dj2

                if not (self._in_bounds(ni1, nj1, dim) and self._in_bounds(ni2, nj2, dim)):
                    continue

                p1 = env_ns.get((ni1, nj1))
                p2 = env_ns.get((ni2, nj2))

                if (
                    p1 is not None
                    and p2 is not None
                    and p1.get_type() == my_piece
                    and p2.get_type() == my_piece
                ):
                    complete_bridge_bonus += 6

            # 5) critère de conservation :
            #    on garde seulement :
            #      - cases dans finishers
            #      - OU coups avec x bonus de pattern
            is_finisher = (i, j) in finishers
            has_pattern = (
                bridge_bonus > 0
                or block_bridge_bonus > 0
                or complete_bridge_bonus > 0
            )

            if not is_finisher and not has_pattern:
                # on enlève les coups "neutres" qui ne font ni finisher ni pattern
                continue

            # 6) rangement : finishers d'abord, puis le reste
            if is_finisher:
                finishers_actions.append(ha)
            else:
                other_good_actions.append(ha)

        # Si on a trouvé des coups intéressants, on les renvoie.
        # Sinon, on retombe sur la liste originale pour ne pas bloquer la recherche.
        if finishers_actions or other_good_actions:
            return finishers_actions + other_good_actions
        return actions


    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        dim = current_state.get_rep().get_dimensions()[0]
        total_cells = dim * dim
        filled = len(current_state.rep.env)
        empties = total_cells - filled

        if empties > total_cells * 0.6:
            self._max_depth = 1
        elif empties > total_cells * 0.3:
            self._max_depth = 3
        else:
            self._max_depth = 5

        self._shannon_cache.clear()

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

            center_positions = [(center, center), (center, center + 1),
                                (center + 1, center), (center + 1, center + 1)]

            row, col = center_positions[0]  # Default fallback

            for pos in center_positions:
                if pos not in env:
                    row, col = pos
                    break

            return LightAction({"piece": self.piece_type, "position": (row, col)})

    def minimax_search(self, current_state: GameState, alpha: int = -inf, beta: int = inf) -> Action:
        _, best_heavy_action = self.max_value(current_state, self._max_depth, alpha, beta, can_extend=True)
        return current_state.convert_heavy_action_to_light_action(best_heavy_action)


    def max_value(self, current_state: GameState, depth: int, alpha: int, beta: int, can_extend: bool = False):
        key = hash(current_state)
        if key in self._tt:
            return (self._tt[key], None)
        # Si la partie est déjà terminée
        if current_state.is_done():
            return (current_state.get_player_score(self), None)

        # -----------------------------
        # Feuille : profondeur 0
        # -----------------------------
        if depth == 0:
            score = self.heuristic_evalutation(current_state)

            # Extension sélective : si la position est très bonne pour moi
            if can_extend and score > self._extension_threshold:
                best_score = -inf
                best_action = None

                # Je simule MES coups (car max joue ici)
                actions = self._order_actions_finishers_first(
                    current_state,
                    list(current_state.generate_possible_heavy_actions()),
                    self.piece_type
                )

                for heavy_action in actions:
                    next_state = heavy_action.get_next_game_state()

                    if next_state.is_done():
                        child_score = next_state.get_player_score(self)
                    else:
                        # max joue → l'adversaire va répondre (min_value)
                        self.next_light_action = current_state.convert_heavy_action_to_light_action(heavy_action)
                        child_score, _ = self.min_value(next_state, 0, alpha, beta, can_extend=False)

                    if child_score > best_score:
                        best_score = child_score
                        best_action = heavy_action

                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break

                return (best_score, best_action)

            # Pas d’extension : heuristique brute
            return (score, None)

        # -----------------------------
        # Cas général : profondeur > 0
        # -----------------------------
        best_score: float = -inf
        best_heavy_action: HeavyAction | None = None

        actions = self._order_actions_finishers_first(
            current_state,
            list(current_state.generate_possible_heavy_actions()),
            self.piece_type
        )

        for heavy_action in actions:
            next_state = heavy_action.get_next_game_state()

            if next_state.is_done():
                return (next_state.get_player_score(self), heavy_action)

            self.next_light_action = current_state.convert_heavy_action_to_light_action(heavy_action)

            score, _ = self.min_value(next_state, depth - 1, alpha, beta, can_extend=can_extend)

            if score > best_score:
                best_score = score
                best_heavy_action = heavy_action
                alpha = max(alpha, best_score)

            if beta <= alpha:
                break
        self._tt[key] = best_score
        return (best_score, best_heavy_action)


    def min_value(self, current_state: GameState, depth: int, alpha: int, beta: int, can_extend: bool = False):
        key = hash(current_state)
        if key in self._tt:
            return (self._tt[key], None)
        # Si la partie est déjà terminée, on renvoie le score final
        if current_state.is_done():
            return (current_state.get_player_score(self), None)

        # -----------------------------
        # Feuille : profondeur 0
        # -----------------------------
        if depth == 0:
            score = self.heuristic_evalutation(current_state)

            # Extension sélective : la position est très mauvaise pour moi,
            # on regarde si l'adversaire (max) a vraiment une si bonne suite
            if can_extend and score < -self._extension_threshold:
                best_score = inf

                # c'est à l'adversaire de jouer ici → on simule ses coups
                opp_piece = "B" if self.piece_type == "R" else "R"
                actions = self._order_actions_finishers_first(
                    current_state,
                    list(current_state.generate_possible_heavy_actions()),
                    opp_piece
                )

                for heavy_action in actions:
                    if heavy_action.get_next_game_state().is_done():
                        child_score = heavy_action.get_next_game_state().get_player_score(self)
                    else:
                        # on regarde la réponse de "max" à profondeur 0, sans nouvelle extension
                        self.next_light_action = current_state.convert_heavy_action_to_light_action(heavy_action)
                        next_state = heavy_action.get_next_game_state()
                        child_score, _ = self.max_value(next_state, 0, alpha, beta, can_extend=False)

                    if child_score < best_score:
                        best_score = child_score
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break

                self._tt[key] = best_score
                return (best_score, None)

            # pas d’extension : on renvoie simplement l’heuristique
            return (score, None)

        # -----------------------------
        # Cas général : profondeur > 0
        # -----------------------------
        best_score: float = inf
        best_heavy_action: HeavyAction | None = None

        opp_piece = "B" if self.piece_type == "R" else "R"
        for heavy_action in self._order_actions_finishers_first(
                current_state,
                list(current_state.generate_possible_heavy_actions()),
                opp_piece):

            if heavy_action.get_next_game_state().is_done():
                return (heavy_action.get_next_game_state().get_player_score(self), heavy_action)

            self.next_light_action = current_state.convert_heavy_action_to_light_action(heavy_action)
            next_state = heavy_action.get_next_game_state()
            score, _ = self.max_value(next_state, depth - 1, alpha, beta, can_extend=can_extend)

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
        dim = state.get_rep().get_dimensions()[0]

        # -------------------------
        # 1) Termes globaux : Shannon
        # -------------------------
        d_me,  empties_me  = self._shannon_path_empty_cells(state, my_piece)
        d_opp, empties_opp = self._shannon_path_empty_cells(state, opp_piece)

        # clamp pour éviter les infs énormes
        if d_me  >= 10**9: d_me  = dim * 5
        if d_opp >= 10**9: d_opp = dim * 5

        # terme de base : positif si je suis mieux placé
        score = (d_opp - d_me)

        # -------------------------
        # 2) Termes locaux autour du coup joué
        # -------------------------
        if self.next_light_action is None:
            return score  # sécurité, mais normalement ne devrait pas arriver

        i, j = self.next_light_action.data["position"]

        # a) bonus si on joue dans un des empties de mon chemin Shannon
        if (i, j) in empties_me:
            score += 8  # remplir mon propre chemin minimal

        # b) bonus si on joue dans un empty du chemin adverse (on le bloque)
        if (i, j) in empties_opp:
            score += 6  # perturber son meilleur chemin

        # offsets utilisés pour les bridges (comme avant)
        bridge_offsets = [
            (-2,  1),
            (-1,  2),
            (-1, -1),
            ( 1, -2),
            ( 2, -1),
            ( 1,  1),
        ]

        middle_pairs = [
            ((-1,  0), ( 1, -1)),  # (i-1, j)   et (i+1, j-1)
            ((-1,  1), ( 1,  0)),  # (i-1, j+1) et (i+1, j)
            (( 0,  1), ( 0, -1)),  # (i, j+1)   et (i, j-1)
        ]

        bridge_bonus = 0
        block_bridge_bonus = 0
        complete_bridge_bonus = 0

        # c) bridges / blocage de bridge
        for di, dj in bridge_offsets:
            ni, nj = i + di, j + dj
            if not self._in_bounds(ni, nj, dim):
                continue
            p = env.get((ni, nj))
            if p is None:
                continue
            t = p.get_type()
            if t == my_piece:
                bridge_bonus += 3
            elif t == opp_piece:
                block_bridge_bonus += 3

        # d) complete bridge pour moi
        for (di1, dj1), (di2, dj2) in middle_pairs:
            ni1, nj1 = i + di1, j + dj1
            ni2, nj2 = i + di2, j + dj2

            if not (self._in_bounds(ni1, nj1, dim) and self._in_bounds(ni2, nj2, dim)):
                continue

            p1 = env.get((ni1, nj1))
            p2 = env.get((ni2, nj2))

            if (
                p1 is not None
                and p2 is not None
                and p1.get_type() == my_piece
                and p2.get_type() == my_piece
            ):
                complete_bridge_bonus += 6

        # on ajoute les patterns comme des “micro-termes”, pas comme des hard filters
        score += bridge_bonus
        score += block_bridge_bonus
        score += complete_bridge_bonus

        return score




    