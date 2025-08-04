import pandas as pd

data_folder = "dota-2-matches/versions/2/"

try:
    players_df = pd.read_csv(data_folder + "players.csv")
    match_df = pd.read_csv(data_folder + "match.csv")
    print("players.csv and match.csv was uploaded succesfull")
except FileNotFoundError as e:
    print("{e} error. Make sure that files are located in right directive.")
    exit()

print("\n---1. General information players.csv ---")
players_df.info()

print("\n---2. First 5 rows players.csv ---")
print(players_df.head(5))

print("\n---3. Discriptic statistics players.csv ---")
print(players_df.describe().T)

print("\n---4. Winrate from match.csv ---")
win_counts = match_df["radiant_win"].value_counts()
print(win_counts)

radiant_winrate = win_counts[1] / len(match_df)
print(f"Radiant winrate: {radiant_winrate:.2%}")