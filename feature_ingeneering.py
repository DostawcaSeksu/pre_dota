import pandas as pd
import numpy as np

data_folder = 'dota-2-matches/versions/2/'

try:
    players_df = pd.read_csv(data_folder + 'players.csv')
    match_df = pd.read_csv(data_folder + 'match.csv')
    print('players.csv and match.csv was uploaded succesfull')
except FileNotFoundError as e:
    print('{e} error. Make sure that files are located in right directive.')
    exit()

features_to_use = [
    'match_id', 'player_slot', 'gold_per_min', 'xp_per_min',
    'kills', 'deaths', 'assists', 'denies', 'last_hits',
    'hero_damage', 'hero_healing', 'tower_damage'
]
players_df = players_df[features_to_use]

players_df['team'] = np.where(players_df['player_slot'] <= 4, 'radiant', 'dire')
players_df.drop('player_slot', axis=1, inplace=True)

team_stats = players_df.groupby(['match_id', 'team']).sum()
wide_stats = team_stats.unstack()

advantage_features = pd.DataFrame()
for col in team_stats.columns:
    advantage_features[f'{col}_adv'] = wide_stats[(col, 'radiant')] - wide_stats[(col, 'dire')]

match_df.set_index('match_id', inplace=True)
final_df = advantage_features.join(match_df['radiant_win'])

final_df.dropna(inplace=True)

print('\n--- Final DataFrame is ready ---')
print('\nGeneral information: ')
final_df.info()

print('\nFirst 5 rows: ')
print(final_df.head())

final_df.to_csv('dota_preprocessed_data.csv')
print('\nFinal data is saved in "dota_preprocessed_data.csv".')