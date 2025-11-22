import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

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

entry_team_counts = log.dropna(subset=['TeamAbbrev']).groupby(['EntryId', 'TeamAbbrev']).size().reset_index(name='TeamCount')
stack_base = log.dropna(subset=['TeamAbbrev']).merge(entry_team_counts, on=['EntryId', 'TeamAbbrev'], how='left')
stack_lookup = (stack_base['TeamCount'] > 1).groupby(stack_base['Player']).mean()

unique_cols = ['Player', 'Slate', 'TeamAbbrev', 'Position', 'Opponent', 'IsHome', 'Salary', 'AvgPointsPerGame', 'FPTS']
unique = log[unique_cols].drop_duplicates()

ratings_map = {}
if RATINGS_PATH.exists():
    ratings_df = pd.read_csv(RATINGS_PATH)
    ratings_df['Player'] = ratings_df['Player'].astype(str).str.strip()
    ratings_map = ratings_df.set_index('Player')['Rating'].to_dict()

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
        if holdout_group.empty:
            continue
        if len(group) < 10:
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
    raise RuntimeError("No predictions generated; check data coverage.")

pred_df = pd.concat(predictions, ignore_index=True)
log_pred = log.merge(pred_df, on=['Player', 'Slate'], how='left')
if 'Position_x' in log_pred.columns:
    log_pred = log_pred.rename(columns={'Position_x': 'Position'})
if 'Position_y' in log_pred.columns:
    log_pred = log_pred.drop(columns=['Position_y'])

log_pred['PredFPTS'] = log_pred['PredFPTS'].astype(float)

missing_preds = log_pred['PredFPTS'].isna().sum()
if missing_preds:
    print(f"Warning: {missing_preds} rows missing predictions (likely due to small training samples).")
    log_pred = log_pred.dropna(subset=['PredFPTS'])

log_pred['Residual'] = log_pred['FPTS'] - log_pred['PredFPTS']

entry_summary = log_pred.groupby('EntryId').agg(
    EntryActual=('FPTS', 'sum'),
    EntryPred=('PredFPTS', 'sum'),
    EntryResidual=('Residual', 'sum'),
).reset_index()

lineup = log_pred[['EntryId', 'Player', 'Position', 'TeamAbbrev', 'Opponent', 'IsHome', 'Residual']]

entry_groups = lineup.groupby('EntryId')


def qb_stack(group):
    qb_teams = group.loc[group['Position'] == 'QB', 'TeamAbbrev'].dropna().unique()
    if qb_teams.size == 0:
        return False
    catchers = group[group['Position'].isin(['WR', 'TE'])]
    return catchers['TeamAbbrev'].isin(qb_teams).any()


def bring_back(group):
    qb_rows = group[group['Position'] == 'QB']
    catchers = group[group['Position'].isin(['WR', 'TE'])]
    if qb_rows.empty or catchers.empty:
        return False
    for _, qb in qb_rows.iterrows():
        team = qb['TeamAbbrev']
        opp = qb['Opponent']
        if pd.isna(team) or pd.isna(opp):
            continue
        same_team = catchers[catchers['TeamAbbrev'] == team]
        if same_team.empty:
            continue
        if not catchers[catchers['TeamAbbrev'] == opp].empty:
            return True
    return False


def rb_dst(group):
    rb = set(group.loc[group['Position'] == 'RB', 'TeamAbbrev'].dropna())
    dst = set(group.loc[group['Position'] == 'DST', 'TeamAbbrev'].dropna())
    return bool(rb & dst)


def max_team_stack(group):
    counts = group['TeamAbbrev'].value_counts()
    return counts.max() if not counts.empty else 0

summary_records = []
for entry_id, group in entry_groups:
    summary_records.append({
        'EntryId': entry_id,
        'QBStack': qb_stack(group),
        'BringBack': bring_back(group),
        'RBDST': rb_dst(group),
        'MaxTeamStack': max_team_stack(group),
    })

flags_df = pd.DataFrame(summary_records)
stack_df = entry_summary.merge(flags_df, on='EntryId', how='left')

print("\nLineup residuals by stack feature:")
for col in ['QBStack', 'BringBack', 'RBDST']:
    stats = stack_df.groupby(col)['EntryResidual'].agg(['mean', 'count']).rename(columns={'mean': 'MeanResidual', 'count': 'Lineups'})
    print(f"\n{col}:")
    print(stats.to_string(float_format=lambda v: f"{v:.2f}"))

stack_stats = stack_df.groupby('MaxTeamStack')['EntryResidual'].agg(['mean', 'count']).rename(columns={'mean': 'MeanResidual', 'count': 'Lineups'})
print("\nResiduals by maximum teammates from one NFL team:")
print(stack_stats.to_string(float_format=lambda v: f"{v:.2f}"))

top_positive = stack_df.sort_values('EntryResidual', ascending=False).head(10)
top_negative = stack_df.sort_values('EntryResidual').head(10)

print("\nTop 10 lineups beating projection:")
print(top_positive.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

print("\nBottom 10 lineups underperforming projection:")
print(top_negative.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
