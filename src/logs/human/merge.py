import json
import os

GREEDY_FILENAME = "vs_greedy.json"
OUTPUT_FILENAME = "human_total.json"


def merge_json_files():
    total_wins = 0
    total_games = 0

    # List all json files in the current directory
    files = [f for f in os.listdir(".") if f.endswith(".json")]

    # Exclude vs_greedy.json and the output file itself
    excluded_files = {GREEDY_FILENAME, OUTPUT_FILENAME}
    files_to_merge = [f for f in files if f not in excluded_files]

    print(f"Merging files: {files_to_merge}")

    for filename in files_to_merge:
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                total_wins += data.get("wins", 0)
                total_games += data.get("total", 0)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {filename}: {e}")

    result = {"wins": total_wins, "total": total_games}

    with open("human_total.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"Successfully merged into human_total.json: {result}")
    print(f"Final win rate of human players: {total_wins / total_games:.2%}")


if __name__ == "__main__":
    merge_json_files()
