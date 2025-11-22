import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / 'data' / 'madden_game_log.parquet'
RATINGS_PATH = ROOT / 'data' / 'madden_ratings.csv'

if not LOG_PATH.exists():
    raise FileNotFoundError("Run scripts/step4_build_game_log.py to create the Madden game log.")

log = pd.read_parquet(LOG_PATH).rename(columns={'Slate_x': 'Slate'})
if 'Slate_y' in log.columns:
    log = log.drop(columns=['Slate_y'])

for col in ['FPTS', 'Salary', 'AvgPointsPerGame']:
    log[col] = pd.to_numeric(log[col], errors='coerce')

ratings_map = {}
if RATINGS_PATH.exists():
    ratings_df = pd.read_csv(RATINGS_PATH)
    ratings_df['Player'] = ratings_df['Player'].astype(str).str.strip()
    ratings_map = ratings_df.set_index('Player')['Rating'].to_dict()

entry_team_counts = log.dropna(subset=['TeamAbbrev']).groupby(['EntryId', 'TeamAbbrev']).size().reset_index(name='TeamCount')
stack_base = log.dropna(subset=['TeamAbbrev']).merge(entry_team_counts, on=['EntryId', 'TeamAbbrev'], how='left')
stack_lookup = (stack_base['TeamCount'] > 1).groupby(stack_base['Player']).mean()

unique_cols = ['Player', 'Slate', 'TeamAbbrev', 'Position', 'Opponent', 'IsHome', 'Salary', 'AvgPointsPerGame', 'FPTS']
unique = log[unique_cols].drop_duplicates()

player_mean = unique.groupby('Player')['FPTS'].mean().to_dict()
player_opp_mean = unique.dropna(subset=['Opponent']).groupby(['Player', 'Opponent'])['FPTS'].mean().to_dict()
team_pos_mean = unique.dropna(subset=['TeamAbbrev']).groupby(['TeamAbbrev', 'Position'])['FPTS'].mean().to_dict()
team_pos_opp_mean = unique.dropna(subset=['TeamAbbrev', 'Opponent']).groupby(['TeamAbbrev', 'Opponent', 'Position'])['FPTS'].mean().to_dict()
team_total_mean = unique.groupby(['TeamAbbrev', 'Slate'])['FPTS'].sum().groupby('TeamAbbrev').mean().to_dict()

feature_records = []
for row in unique.itertuples(index=False):
    feature_records.append({
        'Player': row.Player,
        'Slate': row.Slate,
        'TeamAbbrev': row.TeamAbbrev,
        'Position': row.Position,
        'Opponent': row.Opponent,
        'IsHome': float(row.IsHome) if pd.notna(row.IsHome) else np.nan,
        'Salary': row.Salary,
        'AvgPointsPerGame': row.AvgPointsPerGame,
        'FPTS': row.FPTS,
        'StackRate': stack_lookup.get(row.Player, 0.0),
        'PlayerMean': player_mean.get(row.Player, np.nan),
        'PlayerOpponentMean': player_opp_mean.get((row.Player, row.Opponent), np.nan),
        'TeamMean': team_pos_mean.get((row.TeamAbbrev, row.Position), np.nan),
        'TeamOpponentMean': team_pos_opp_mean.get((row.TeamAbbrev, row.Opponent, row.Position), np.nan),
        'TeamTotalMean': team_total_mean.get(row.TeamAbbrev, np.nan),
        'Rating': ratings_map.get(row.Player, np.nan),
    })

feature_df = pd.DataFrame(feature_records)

feature_cols = [
    'Salary',
    'AvgPointsPerGame',
    'IsHome',
    'StackRate',
    'PlayerMean',
    'PlayerOpponentMean',
    'TeamMean',
    'TeamOpponentMean',
    'TeamTotalMean',
    'Rating',
]

predictions = []
slates = sorted(feature_df['Slate'].unique())

for slate in slates:
    holdout_mask = feature_df['Slate'] == slate
    train_subset = feature_df.loc[~holdout_mask]
    holdout_subset = feature_df.loc[holdout_mask]
    if train_subset.empty or holdout_subset.empty:
        continue
    slate_preds = []
    for position, group in train_subset.groupby('Position'):
        holdout_group = holdout_subset[holdout_subset['Position'] == position]
        if holdout_group.empty or len(group) < 10:
            continue
        X_train = group[feature_cols].copy()
        pos_means = X_train.mean().fillna(0)
        X_train = X_train.fillna(pos_means)
        y_train = group['FPTS']
        model = Ridge(alpha=5.0)
        model.fit(X_train, y_train)
        X_holdout = holdout_group[feature_cols].copy().fillna(pos_means)
        preds = model.predict(X_holdout)
        slate_preds.append(pd.DataFrame({
            'Player': holdout_group['Player'].values,
            'Slate': slate,
            'Position': position,
            'PredFPTS': preds,
        }))
    if slate_preds:
        predictions.append(pd.concat(slate_preds, ignore_index=True))

if not predictions:
    raise RuntimeError("No predictions generated; cannot continue.")

pred_df = pd.concat(predictions, ignore_index=True)
log_pred = log.merge(pred_df, on=['Player', 'Slate'], how='left')
if 'Position_x' in log_pred.columns:
    log_pred = log_pred.rename(columns={'Position_x': 'Position'})
if 'Position_y' in log_pred.columns:
    log_pred = log_pred.drop(columns=['Position_y'])

missing_preds = log_pred['PredFPTS'].isna().sum()
if missing_preds:
    log_pred = log_pred.dropna(subset=['PredFPTS'])

log_pred['Residual'] = log_pred['FPTS'] - log_pred['PredFPTS']

# Variance analysis
variance_df = log_pred.groupby('Player')['FPTS'].agg(['mean', 'std', 'count']).rename(columns={'std': 'StdDev'})
variance_df['CoeffVar'] = variance_df['StdDev'] / variance_df['mean'].replace(0, np.nan)
print("Player variance snapshot (top 15 by appearances):")
print(variance_df[variance_df['count'] >= 5].sort_values('StdDev').head(15).to_string(float_format=lambda v: f"{v:.2f}"))

# Contest data
pattern = re.compile(r'(DST|FLEX|QB|RB|WR|TE)\s+')
entry_rows, player_rows = [], []
contest_files = sorted((ROOT / 'contest').glob('DKcontest_*.csv'))
for cpath in contest_files:
    slate = cpath.stem.split('DKcontest_')[1]
    df = pd.read_csv(cpath)
    df['Slate'] = slate
    df['%Drafted'] = df['%Drafted'].str.rstrip('%').astype(float) / 100
    entry_rows.append(df[['EntryId', 'Points', 'Slate']])
    for _, row in df[['EntryId', 'Lineup', '%Drafted', 'Slate']].dropna(subset=['Lineup']).iterrows():
        matches = list(pattern.finditer(row['Lineup']))
        for i, match in enumerate(matches):
            slot = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(row['Lineup'])
            name = row['Lineup'][start:end].strip()
            if name and name.upper() != 'LOCKED':
                player_rows.append({
                    'EntryId': row['EntryId'],
                    'Slate': row['Slate'],
                    'Slot': slot,
                    'Player': name,
                    'Ownership': row['%Drafted'],
                })

contest_entries = pd.concat(entry_rows, ignore_index=True).drop_duplicates()
contest_players = pd.DataFrame(player_rows)

contest_entries['Rank'] = contest_entries.groupby('Slate')['Points'].rank(ascending=False, method='first')
contest_entries['Cut'] = (contest_entries.groupby('Slate')['EntryId'].transform('count') * 0.01).clip(lower=1)
top_ids = set(contest_entries[contest_entries['Rank'] <= contest_entries['Cut']]['EntryId'])
contest_players['IsTop1'] = contest_players['EntryId'].isin(top_ids)

ownership_df = contest_players.groupby(['Player', 'Slate']).agg(ActualOwnership=('Ownership', 'mean')).reset_index()
pred_ownership = pred_df.merge(ownership_df, on=['Player', 'Slate'], how='inner')
print("\nCorrelation between predicted FPTS and actual ownership (per slate):")
correlations = []
for slate, group in pred_ownership.groupby('Slate'):
    if group['PredFPTS'].nunique() < 2:
        continue
    corr = group['PredFPTS'].corr(group['ActualOwnership'])
    correlations.append(corr)
    print(f"Slate {slate}: corr={corr:.3f}")
if correlations:
    print(f"Average correlation: {np.nanmean(correlations):.3f}")

# Game environment stability
matchup_stats = log_pred.dropna(subset=['Opponent']).groupby(['TeamAbbrev', 'Opponent', 'Position'])['FPTS'].agg(['mean', 'std', 'count']).reset_index()
repeatable = matchup_stats[(matchup_stats['count'] >= 3)].sort_values('std')
print("\nMost stable matchups (>=3 samples, lowest stdev):")
print(repeatable.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# Ownership in top lineups vs field
top_lineup_ownership = contest_players[contest_players['IsTop1']]['Ownership'].mean()
field_ownership = contest_players['Ownership'].mean()
print(f"\nAverage player ownership (top 1% lineups): {top_lineup_ownership:.3f}")
print(f"Average player ownership (all lineups): {field_ownership:.3f}")

# Salary inefficiencies by position
salary_stats = log_pred.groupby(['Position'])[['FPTS', 'Salary']].mean()
salary_stats['FPTS_per_1k'] = salary_stats['FPTS'] / (salary_stats['Salary'] / 1000)
print("\nAverage FPTS per $1k salary by position:")
print(salary_stats[['FPTS_per_1k']].sort_values('FPTS_per_1k', ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))

# Ownership vs projection to highlight under-owned hits
merged = pred_df.merge(ownership_df, on=['Player', 'Slate'], how='left').merge(log_pred[['Player', 'Slate', 'FPTS']], on=['Player', 'Slate'], how='left')
merged['ValueScore'] = merged['FPTS'] - merged['PredFPTS']
under_owned_hits = merged[(merged['ActualOwnership'] <= 0.10) & (merged['ValueScore'] > 5)].sort_values('ValueScore', ascending=False).head(15)
print("\nUnder-owned overperformers (<=10% ownership, value > 5 FPTS):")
print(under_owned_hits[['Player', 'Slate', 'ActualOwnership', 'PredFPTS', 'FPTS', 'ValueScore']].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


