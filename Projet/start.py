import subprocess
import sys

def main():
    if len(sys.argv) > 1 :
        arg = sys.argv[1]
        if arg == "-twoRandom":
            cmd = ["python", "main_hex.py", "-t", "local", "random_player_hex.py", "random_player_hex.py"]

            result = subprocess.run(cmd, text=True)

            if result.returncode != 0:
                print("Error: Command failed with return code", result.returncode)

        elif arg == "-hostGame":
            ipAddress = sys.argv[2] 
            cmd = ["python", "main_hex.py", "-t", "host_game", "-a", ipAddress, "random_player_hex.py"]
            result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                print("Error: Command failed with return code", result.returncode)

        elif arg == "-connect":
            ipAddress = sys.argv[2] 
            cmd = ["python", "main_hex.py", "-t", "connect", "-a", ipAddress, "random_player_hex.py"]
            result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                print("Error: Command failed with return code", result.returncode)
        
        elif arg == "-humanGame":
            cmd = ["python", "main_hex.py", "-t", "human_vs_human"]
            result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                print("Error: Command failed with return code", result.returncode)

        elif arg == "-humanVsComputer":
            cmd = ["python", "main_hex.py", "-t", "human_vs_computer", "random_player_hex.py"]
            result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                print("Error: Command failed with return code", result.returncode)
        else:
            print("Argument unrecognized")
    else:
        print("No argument provided")

if __name__ == "__main__":
    main()
    