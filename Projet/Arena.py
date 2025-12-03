import sys
import os
import csv
import itertools
import concurrent.futures
from os.path import splitext, basename

# Add current directory to sys.path to allow imports
sys.path.append(os.getcwd())

# List of players to include in the tournament
# You can add or remove players here
PLAYERS = [
    "boo4.py",
    "baba3.py",
    "baudelaire.py",
    "gumball.py",
    "bobynettebridge.py",
    "bob.py",
]

def run_single_match(args):
    import sys
    import os
    import time
    from importlib import util
    from os.path import splitext, basename
    from board_hex import BoardHex
    from game_state_hex import GameStateHex
    from master_hex import MasterHex

    p1_file, p2_file, c1, c2, match_num, total_matches, port = args

    # Ensure sys.path has current directory
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    def load_player_class(file_path):
        module_name = splitext(basename(file_path))[0]
        spec = util.spec_from_file_location(module_name, file_path)
        module = util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.MyPlayer

    p1_name = splitext(basename(p1_file))[0]
    p2_name = splitext(basename(p2_file))[0]

    print(
        f"[{match_num}/{total_matches}] Starting {p1_name} ({c1}) vs {p2_name} ({c2}) on port {port}..."
    )

    start_time = time.time()
    winner_file = None
    error = None

    try:
        p1_class = load_player_class(p1_file)
        p2_class = load_player_class(p2_file)

        player1 = p1_class(c1, name=f"{p1_name}")
        player2 = p2_class(c2, name=f"{p2_name}")

        list_players = [player1, player2]
        init_scores = {player1.get_id(): 0, player2.get_id(): 0}
        dim = [14, 14]
        env = {}

        init_rep = BoardHex(env=env, dim=dim)
        initial_game_state = GameStateHex(
            scores=init_scores,
            next_player=player1,
            players=list_players,
            rep=init_rep,
            step=0,
        )

        master = MasterHex(
            name="Hex",
            initial_game_state=initial_game_state,
            players_iterator=list_players,
            log_level="ERROR",
            port=port,
            hostname="localhost",
            time_limit=60 * 15,
        )

        master.record_game(listeners=[])
        winners = master.compute_winner()

        if len(winners) == 1:
            if winners[0].get_id() == player1.get_id():
                winner_file = p1_file
            else:
                winner_file = p2_file

    except Exception as e:
        error = str(e)
        print(f"Error in match {match_num} ({p1_name} vs {p2_name}): {e}")

    duration = time.time() - start_time

    # Return simple types that are picklable
    return {
        "p1_file": p1_file,
        "p2_file": p2_file,
        "winner_file": winner_file,
        "duration": duration,
        "error": error,
        "match_num": match_num,
        "p1_name": p1_name,
        "p2_name": p2_name,
    }


class Arena:
    def __init__(self, players_files):
        self.players_files = players_files
        self.results = []
        self.stats = {
            p: {"wins": 0, "losses": 0, "draws": 0, "points": 0} for p in players_files
        }

    def run_tournament(self):
        print(f"Starting tournament with {len(self.players_files)} players...")
        print("-" * 60)

        pairs = list(itertools.combinations(self.players_files, 2))
        matches = []

        match_id = 0
        for p1, p2 in pairs:
            # Match 1: p1 is Red (starts), p2 is Blue
            match_id += 1
            matches.append((p1, p2, "R", "B", match_id))

            # Match 2: p2 is Red (starts), p1 is Blue
            match_id += 1
            matches.append((p2, p1, "R", "B", match_id))

        total_matches = len(matches)

        # Prepare arguments for workers
        worker_args = []
        for i, (p1, p2, c1, c2, m_id) in enumerate(matches):
            port = 8080 + i
            worker_args.append((p1, p2, c1, c2, m_id, total_matches, port))

        max_workers = os.cpu_count()
        print(f"Running matches in parallel with {max_workers} workers...")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for result in executor.map(run_single_match, worker_args):
                self._process_result(result)

    def _process_result(self, result):
        p1_file = result["p1_file"]
        p2_file = result["p2_file"]
        winner_file = result["winner_file"]
        duration = result["duration"]
        error = result["error"]
        match_num = result["match_num"]
        p1_name = result["p1_name"]
        p2_name = result["p2_name"]

        if error:
            print(f"Match {match_num} failed: {error}")
            return

        winner_name = "Draw"
        if winner_file:
            winner_name = splitext(basename(winner_file))[0]
            self.stats[winner_file]["wins"] += 1
            self.stats[winner_file]["points"] += 1
            loser = p2_file if winner_file == p1_file else p1_file
            self.stats[loser]["losses"] += 1
        else:
            self.stats[p1_file]["draws"] += 1
            self.stats[p2_file]["draws"] += 1
            self.stats[p1_file]["points"] += 0.5
            self.stats[p2_file]["points"] += 0.5

        print(
            f"[{match_num}] Finished {p1_name} vs {p2_name} -> Winner: {winner_name} ({duration:.2f}s)"
        )

        self.results.append(
            {
                "Player 1": p1_name,
                "Player 2": p2_name,
                "Winner": winner_name,
                "Duration": f"{duration:.2f}",
            }
        )

    def export_csv(self, filename="tournament_results.csv"):
        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["Player 1", "Player 2", "Winner", "Duration"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)
        print(f"\nResults exported to {filename}")

    def print_standings(self):
        print("\n" + "=" * 65)
        print(f"{'TOURNAMENT STANDINGS':^65}")
        print("=" * 65)
        print(
            f"{'Rank':<6} {'Player':<20} {'Points':<8} {'Wins':<6} {'Losses':<8} {'Draws':<6}"
        )
        print("-" * 65)

        sorted_stats = sorted(
            self.stats.items(), key=lambda x: x[1]["points"], reverse=True
        )

        for rank, (player_file, stat) in enumerate(sorted_stats, 1):
            name = splitext(basename(player_file))[0]
            print(
                f"{rank:<6} {name:<20} {stat['points']:<8} {stat['wins']:<6} {stat['losses']:<8} {stat['draws']:<6}"
            )
        print("=" * 65)


if __name__ == "__main__":
    # Filter out files that don't exist
    players = [p for p in PLAYERS if os.path.exists(p)]

    if len(players) < 2:
        print("Not enough players found to start a tournament.")
    else:
        arena = Arena(players)
        arena.run_tournament()
        arena.export_csv()
        arena.print_standings()
