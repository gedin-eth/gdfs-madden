import numpy as np
import pandas as pd
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data' / 'madden_game_log.parquet'
if not data_path.exists():
    raise FileNotFoundError("Run scripts/step4_build_game_log.py first to materialize the game log.")

log = pd.read_parquet(data_path)
log = log.rename(columns={'Slate_x': 'Slate'})
if 'Slate_y' in log.columns:
    log = log.drop(columns=['Slate_y'])
log['FPTS'] = log['FPTS'].astype(float)
log['Salary'] = log['Salary'].astype(float)
log['AvgPointsPerGame'] = log['AvgPointsPerGame'].astype(float)

ratings_path = Path(__file__).resolve().parents[1] / 'data' / 'madden_ratings.csv'
rating_map = {}
rating_scale = {}
if ratings_path.exists():
    ratings_df = pd.read_csv(ratings_path)
    ratings_df['Player'] = ratings_df['Player'].astype(str).str.strip()
    rating_map = ratings_df.set_index('Player')['Rating'].to_dict()
    log['Rating'] = log['Player'].map(rating_map)
    valid = log.dropna(subset=['Rating', 'FPTS'])
    if not valid.empty:
        pos_scale = (valid.groupby('Position')['FPTS'].mean() / valid.groupby('Position')['Rating'].mean()).to_dict()
        global_scale = valid['FPTS'].mean() / valid['Rating'].mean()
        rating_scale = {pos: scale for pos, scale in pos_scale.items() if not np.isnan(scale)}
        rating_scale['__global__'] = global_scale
else:
    log['Rating'] = np.nan

entries = log[['EntryId', 'Slate', 'LineupScore']].drop_duplicates()
entries['Rank'] = entries.groupby('Slate')['LineupScore'].rank(ascending=False, method='first')
entries['Cut'] = (entries.groupby('Slate')['EntryId'].transform('count') * 0.01).clip(lower=1)
entries['IsTop1'] = entries['Rank'] <= entries['Cut']
log = log.merge(entries[['EntryId', 'IsTop1']], on='EntryId', how='left')

# Projection features
match_mean = log.dropna(subset=['Opponent']).groupby(['Player', 'Opponent'])['FPTS'].mean()
player_mean = log.groupby('Player')['FPTS'].mean()
team_pos_mean = log.dropna(subset=['TeamAbbrev']).groupby(['TeamAbbrev', 'Position'])['FPTS'].mean()
home_mean = log.dropna(subset=['IsHome']).groupby(['Player', 'IsHome'])['FPTS'].mean()

recent_mean = log.sort_values('Slate').groupby('Player')['FPTS'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
log['RecentAvg'] = recent_mean
recent_lookup = log.groupby('Player')['RecentAvg'].last()

contested = {idx: val for idx, val in match_mean.items()}
player_avg = player_mean.to_dict()
team_avg = team_pos_mean.to_dict()
home_avg = home_mean.to_dict()
recent_avg = recent_lookup.to_dict()


def project(row):
    key = (row['Player'], row['Opponent'])
    if key in contested:
        return contested[key]
    key_home = (row['Player'], row['IsHome'])
    if key_home in home_avg:
        return home_avg[key_home]
    if row['Player'] in recent_avg:
        return recent_avg[row['Player']]
    if row['Player'] in player_avg:
        return player_avg[row['Player']]
    tkey = (row['TeamAbbrev'], row['Position'])
    if tkey in team_avg:
        return team_avg[tkey]
    base = row['AvgPointsPerGame']
    if pd.notna(base) and base > 0:
        return base
    rating = row.get('Rating')
    if pd.notna(rating):
        pos_scale = rating_scale.get(row['Position'])
        scale = pos_scale if pos_scale is not None else rating_scale.get('__global__', 0.2)
        return rating * scale
    return 0.0

log['ProjPoints'] = log.apply(project, axis=1)

# Lineup appearance probability
field_counts = log.groupby('Player')['EntryId'].nunique()
top_counts = log[log['IsTop1']].groupby('Player')['EntryId'].nunique()
usage = field_counts.to_frame('FieldCount').join(top_counts.to_frame('TopCount'), how='left').fillna(0)

entries_total = entries['EntryId'].nunique() or 1
entries_top = entries[entries['IsTop1']]['EntryId'].nunique() or 1
usage['FieldRate'] = usage['FieldCount'] / entries_total
usage['TopRate'] = usage['TopCount'] / entries_top
usage['OwnershipLeverage'] = ((usage['TopRate'] - usage['FieldRate']) / usage['FieldRate'].replace(0, np.nan))
usage['OwnershipLeverage'] = usage['OwnershipLeverage'].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.9, 1.0)

entry_team_counts = log.dropna(subset=['TeamAbbrev']).groupby(['EntryId', 'TeamAbbrev']).size().reset_index(name='TeamCount')
stack_base = log.dropna(subset=['TeamAbbrev']).merge(entry_team_counts, on=['EntryId', 'TeamAbbrev'], how='left')
stack_lookup = (stack_base['TeamCount'] > 1).groupby(stack_base['Player']).mean()

variance_data = log.groupby(['Position', 'Player'])['FPTS'].agg(['std', 'count']).reset_index()
variance_data = variance_data[variance_data['count'] >= 5]
low_var_players = variance_data[variance_data['std'] <= 2]
low_var_bonus_lookup = {}
for _, row in low_var_players.iterrows():
    if row['Position'] in {'RB', 'TE'}:
        low_var_bonus_lookup[row['Player']] = max(0.05, (2 - row['std']) * 0.05)


def qb_stack(group: pd.DataFrame) -> bool:
    qb_teams = group.loc[group['Position'] == 'QB', 'TeamAbbrev'].dropna().unique()
    if qb_teams.size == 0:
        return False
    catchers = group[group['Position'].isin(['WR', 'TE'])]
    return catchers['TeamAbbrev'].isin(qb_teams).any()


def bring_back(group: pd.DataFrame) -> bool:
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


def rb_dst(group: pd.DataFrame) -> bool:
    rbs = set(group.loc[group['Position'] == 'RB', 'TeamAbbrev'].dropna())
    dsts = set(group.loc[group['Position'] == 'DST', 'TeamAbbrev'].dropna())
    return bool(rbs & dsts)


def max_team_stack(group: pd.DataFrame) -> float:
    counts = group['TeamAbbrev'].value_counts()
    return counts.max() if not counts.empty else 0.0


flag_records = []
for entry_id, group in log.groupby('EntryId'):
    qb_flag = qb_stack(group)
    bring_flag = bring_back(group)
    rb_flag = rb_dst(group)
    max_stack = max_team_stack(group)
    players = group['Player']
    flag_records.append(pd.DataFrame({
        'Player': players,
        'EntryId': entry_id,
        'QBStackFlag': qb_flag,
        'BringBackFlag': bring_flag,
        'RBDSTFlag': rb_flag,
        'MaxTeamStack': max_stack,
    }))

flag_df = pd.concat(flag_records, ignore_index=True)
flag_stats = flag_df.groupby('Player').agg(
    QBStackRate=('QBStackFlag', 'mean'),
    BringBackRate=('BringBackFlag', 'mean'),
    RBDSTRate=('RBDSTFlag', 'mean'),
    AvgMaxTeamStack=('MaxTeamStack', 'mean'),
)

# Position scarcity
pos_field = log.groupby('Position')['EntryId'].nunique()
pos_top = log[log['IsTop1']].groupby('Position')['EntryId'].nunique()
pos_share = pos_field.to_frame('Field').join(pos_top.to_frame('Top'), how='left').fillna(0)
pos_share['FieldShare'] = pos_share['Field'] / pos_share['Field'].sum()
pos_share['TopShare'] = pos_share['Top'] / pos_share['Top'].sum()
pos_share['Scarcity'] = (pos_share['TopShare'] / pos_share['FieldShare']).replace([np.inf, -np.inf], np.nan).fillna(1.0)

# Summary table
summary = log.drop_duplicates(subset=['Player', 'Slate']).copy()
summary = summary.merge(usage[['FieldRate', 'TopRate', 'OwnershipLeverage']], left_on='Player', right_index=True, how='left')
summary['StackRate'] = summary['Player'].map(stack_lookup).fillna(0)
summary = summary.merge(flag_stats, left_on='Player', right_index=True, how='left')
summary['LowVarBonus'] = summary['Player'].map(low_var_bonus_lookup).fillna(0)
stack_component = (
    summary['StackRate'] * 0.30
    + summary['QBStackRate'].fillna(0) * 0.40
    + summary['BringBackRate'].fillna(0) * 1.00
    + summary['RBDSTRate'].fillna(0) * 0.35
    + np.maximum(summary['AvgMaxTeamStack'].fillna(0) - 2, 0) * 0.07
)
summary['StackBonus'] = 1 + stack_component + summary['LowVarBonus']
summary['StackBonus'] = summary['StackBonus'].clip(lower=1.0)
summary = summary.merge(pos_share['Scarcity'], left_on='Position', right_index=True, how='left')
summary['BaseValue'] = (summary['ProjPoints'] / summary['Salary'].replace(0, np.nan) * 1000).fillna(0)
summary['MaddenEdge'] = summary['BaseValue'] * (1 + summary['OwnershipLeverage'].fillna(0)) * summary['StackBonus'] * summary['Scarcity'].fillna(1)

cols = ['Player', 'Slate', 'TeamAbbrev', 'Position', 'Rating', 'Salary', 'ProjPoints', 'FPTS', 'TopRate', 'FieldRate', 'OwnershipLeverage', 'StackRate', 'QBStackRate', 'BringBackRate', 'RBDSTRate', 'LowVarBonus', 'StackBonus', 'Scarcity', 'MaddenEdge']
result = summary[cols].sort_values('MaddenEdge', ascending=False).round(3)
print(result.head(15).to_string(index=False))
