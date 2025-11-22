import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / 'data' / 'madden_game_log.parquet'
RATINGS_PATH = ROOT / 'data' / 'madden_ratings.csv'


def load_history():
    if not LOG_PATH.exists():
        raise FileNotFoundError("Run scripts/step4_build_game_log.py to create the Madden game log.")
    log = pd.read_parquet(LOG_PATH).rename(columns={'Slate_x': 'Slate'})
    if 'Slate_y' in log.columns:
        log = log.drop(columns=['Slate_y'])
    for col in ['FPTS', 'Salary', 'AvgPointsPerGame']:
        log[col] = pd.to_numeric(log[col], errors='coerce')
    return log


def compute_usage(log):
    entries = log[['EntryId', 'Slate', 'LineupScore']].drop_duplicates()
    entries['Rank'] = entries.groupby('Slate')['LineupScore'].rank(ascending=False, method='first')
    entries['Cut'] = (entries.groupby('Slate')['EntryId'].transform('count') * 0.01).clip(lower=1)
    entries['IsTop1'] = entries['Rank'] <= entries['Cut']
    log = log.merge(entries[['EntryId', 'IsTop1']], on='EntryId', how='left')

    field_counts = log.groupby('Player')['EntryId'].nunique()
    top_counts = log[log['IsTop1']].groupby('Player')['EntryId'].nunique()
    usage = field_counts.to_frame('FieldCount').join(top_counts.to_frame('TopCount'), how='left').fillna(0)

    entries_total = entries['EntryId'].nunique() or 1
    entries_top = entries[entries['IsTop1']]['EntryId'].nunique() or 1
    usage['FieldRate'] = usage['FieldCount'] / entries_total
    usage['TopRate'] = usage['TopCount'] / entries_top
    usage['OwnershipLeverage'] = ((usage['TopRate'] - usage['FieldRate']) / usage['FieldRate'].replace(0, np.nan))
    usage['OwnershipLeverage'] = usage['OwnershipLeverage'].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-0.9, 1.0)

    return log, usage, entries


def compute_stack_flags(log):
    entry_team_counts = log.dropna(subset=['TeamAbbrev']).groupby(['EntryId', 'TeamAbbrev']).size().reset_index(name='TeamCount')
    stack_base = log.dropna(subset=['TeamAbbrev']).merge(entry_team_counts, on=['EntryId', 'TeamAbbrev'], how='left')
    stack_lookup = (stack_base['TeamCount'] > 1).groupby(stack_base['Player']).mean()

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
        rbs = set(group.loc[group['Position'] == 'RB', 'TeamAbbrev'].dropna())
        dsts = set(group.loc[group['Position'] == 'DST', 'TeamAbbrev'].dropna())
        return bool(rbs & dsts)

    def max_team_stack(group):
        counts = group['TeamAbbrev'].value_counts()
        return counts.max() if not counts.empty else 0

    records = []
    for entry_id, group in log.groupby('EntryId'):
        qb_flag = qb_stack(group)
        bring_flag = bring_back(group)
        rb_flag = rb_dst(group)
        max_stack = max_team_stack(group)
        records.append(pd.DataFrame({
            'Player': group['Player'],
            'QBStackRate': qb_flag,
            'BringBackRate': bring_flag,
            'RBDSTRate': rb_flag,
            'AvgMaxTeamStack': max_stack,
        }))
    flag_df = pd.concat(records, ignore_index=True)
    flag_stats = flag_df.groupby('Player').agg(
        QBStackRate=('QBStackRate', 'mean'),
        BringBackRate=('BringBackRate', 'mean'),
        RBDSTRate=('RBDSTRate', 'mean'),
        AvgMaxTeamStack=('AvgMaxTeamStack', 'mean'),
    )
    return stack_lookup, flag_stats


def compute_low_variance_bonus(log):
    variance_data = log.groupby(['Position', 'Player'])['FPTS'].agg(['std', 'count']).reset_index()
    variance_data = variance_data[variance_data['count'] >= 5]
    low_var_players = variance_data[variance_data['std'] <= 2]
    bonus = {}
    for _, row in low_var_players.iterrows():
        if row['Position'] in {'RB', 'TE'}:
            bonus[row['Player']] = max(0.05, (2 - row['std']) * 0.05)
    return bonus


def compute_pos_scarcity(log, entries):
    pos_field = log.groupby('Position')['EntryId'].nunique()
    pos_top = log[log['IsTop1']].groupby('Position')['EntryId'].nunique()
    pos_share = pos_field.to_frame('Field').join(pos_top.to_frame('Top'), how='left').fillna(0)
    pos_share['FieldShare'] = pos_share['Field'] / pos_share['Field'].sum()
    pos_share['TopShare'] = pos_share['Top'] / pos_share['Top'].sum()
    pos_share['Scarcity'] = (pos_share['TopShare'] / pos_share['FieldShare']).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return pos_share


def build_feature_matrix(log):
    ratings_map = {}
    if RATINGS_PATH.exists():
        ratings_df = pd.read_csv(RATINGS_PATH)
        ratings_df['Player'] = ratings_df['Player'].astype(str).str.strip()
        ratings_map = ratings_df.set_index('Player')['Rating'].to_dict()

    match_mean = log.dropna(subset=['Opponent']).groupby(['Player', 'Opponent'])['FPTS'].mean()
    player_mean = log.groupby('Player')['FPTS'].mean()
    team_pos_mean = log.dropna(subset=['TeamAbbrev']).groupby(['TeamAbbrev', 'Position'])['FPTS'].mean()
    team_pos_opp_mean = log.dropna(subset=['TeamAbbrev', 'Opponent']).groupby(['TeamAbbrev', 'Opponent', 'Position'])['FPTS'].mean()
    team_total_mean = log.groupby(['TeamAbbrev', 'Slate'])['FPTS'].sum().groupby('TeamAbbrev').mean()

    unique_cols = ['Player', 'Slate', 'TeamAbbrev', 'Position', 'Opponent', 'IsHome', 'Salary', 'AvgPointsPerGame', 'FPTS']
    unique = log[unique_cols].drop_duplicates()
    recent_mean = log.sort_values('Slate').groupby('Player')['FPTS'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    log = log.assign(RecentAvg=recent_mean)
    recent_lookup = log.groupby('Player')['RecentAvg'].last()

    records = []
    for row in unique.itertuples(index=False):
        records.append({
            'Player': row.Player,
            'Slate': row.Slate,
            'TeamAbbrev': row.TeamAbbrev,
            'Position': row.Position,
            'Opponent': row.Opponent,
            'IsHome': float(row.IsHome) if pd.notna(row.IsHome) else np.nan,
            'Salary': row.Salary,
            'AvgPointsPerGame': row.AvgPointsPerGame,
            'FPTS': row.FPTS,
            'PlayerMean': player_mean.get(row.Player, np.nan),
            'PlayerOpponentMean': match_mean.get((row.Player, row.Opponent), np.nan),
            'TeamMean': team_pos_mean.get((row.TeamAbbrev, row.Position), np.nan),
            'TeamOpponentMean': team_pos_opp_mean.get((row.TeamAbbrev, row.Opponent, row.Position), np.nan),
            'TeamTotalMean': team_total_mean.get(row.TeamAbbrev, np.nan),
            'RecentAvg': recent_lookup.get(row.Player, np.nan),
            'Rating': ratings_map.get(row.Player, np.nan),
        })
    return pd.DataFrame(records)


def fit_models(train_df):
    feature_cols = [
        'Salary',
        'AvgPointsPerGame',
        'IsHome',
        'PlayerMean',
        'PlayerOpponentMean',
        'TeamMean',
        'TeamOpponentMean',
        'TeamTotalMean',
        'RecentAvg',
        'Rating',
    ]
    models = {}
    feature_means = {}
    for position, group in train_df.groupby('Position'):
        if len(group) < 20:
            continue
        X = group[feature_cols].copy()
        pos_means = X.mean().fillna(0)
        X = X.fillna(pos_means)
        y = group['FPTS']
        model = Ridge(alpha=5.0)
        model.fit(X, y)
        models[position] = model
        feature_means[position] = pos_means
    return models, feature_means


def parse_matchup(game_info, team):
    if not isinstance(game_info, str) or '@' not in game_info:
        return np.nan, np.nan
    part = game_info.split()[0]
    away, home = part.split('@')[:2]
    if team == home:
        return 1.0, away
    if team == away:
        return 0.0, home
    return np.nan, np.nan


def score_salary(salary_path, models, feature_means, usage, stack_lookup, flag_stats, low_var_bonus, pos_share, history_features):
    salary = pd.read_csv(salary_path)
    salary['Name'] = salary['Name'].astype(str).str.strip()
    salary['Game Info'] = salary['Game Info'].astype(str)
    matchup = salary.apply(lambda row: parse_matchup(row['Game Info'], row['TeamAbbrev']), axis=1, result_type='expand')
    salary['IsHome'] = matchup[0]
    salary['Opponent'] = matchup[1]

    feature_cols = list(feature_means[next(iter(feature_means))].index)
    records = []
    for row in salary.itertuples():
        position = row.Position.split('/')[0]
        model = models.get(position)
        if model is None:
            continue
        pos_means = feature_means[position]
        features = {
            'Salary': float(row.Salary),
            'AvgPointsPerGame': float(row.AvgPointsPerGame),
            'IsHome': row.IsHome,
            'PlayerMean': history_features['PlayerMean'].get(row.Name, np.nan),
            'PlayerOpponentMean': history_features['PlayerOpponentMean'].get((row.Name, row.Opponent), np.nan),
            'TeamMean': history_features['TeamMean'].get((row.TeamAbbrev, position), np.nan),
            'TeamOpponentMean': history_features['TeamOpponentMean'].get((row.TeamAbbrev, row.Opponent, position), np.nan),
            'TeamTotalMean': history_features['TeamTotalMean'].get(row.TeamAbbrev, np.nan),
            'RecentAvg': history_features['RecentAvg'].get(row.Name, np.nan),
            'Rating': history_features['Rating'].get(row.Name, np.nan),
        }
        fv = pd.Series(features).reindex(feature_cols)
        fv = fv.fillna(pos_means).fillna(0)
        pred = float(model.predict(fv.values.reshape(1, -1))[0])

        usage_row = usage.reindex([row.Name]).fillna(0)
        stack_rate = stack_lookup.get(row.Name, 0.0)
        flag_row = flag_stats.reindex([row.Name]).fillna(0)
        low_var = low_var_bonus.get(row.Name, 0.0)
        scarcity = pos_share['Scarcity'].get(position, 1.0)

        stack_component = (
            stack_rate * 0.30
            + flag_row['QBStackRate'].iloc[0] * 0.40
            + flag_row['BringBackRate'].iloc[0] * 1.00
            + flag_row['RBDSTRate'].iloc[0] * 0.35
            + max(flag_row['AvgMaxTeamStack'].iloc[0] - 2, 0) * 0.07
        )
        stack_bonus = 1 + stack_component + low_var
        stack_bonus = max(stack_bonus, 1.0)

        base_value = (pred / float(row.Salary)) * 1000 if row.Salary else 0
        edge = base_value * (1 + usage_row['OwnershipLeverage'].iloc[0]) * stack_bonus * scarcity

        records.append({
            'Player': row.Name,
            'TeamAbbrev': row.TeamAbbrev,
            'Position': position,
            'Salary': row.Salary,
            'Opponent': row.Opponent,
            'IsHome': row.IsHome,
            'PredFPTS': pred,
            'BaseValue': base_value,
            'StackBonus': stack_bonus,
            'Scarcity': scarcity,
            'OwnershipLeverage': usage_row['OwnershipLeverage'].iloc[0],
            'StackRate': stack_rate,
            'QBStackRate': flag_row['QBStackRate'].iloc[0],
            'BringBackRate': flag_row['BringBackRate'].iloc[0],
            'RBDSTRate': flag_row['RBDSTRate'].iloc[0],
            'LowVarBonus': low_var,
            'MaddenEdge': edge,
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description='Score future Madden slate using historical metric.')
    parser.add_argument('--salary-csv', required=True, help='DraftKings salary CSV for future slate.')
    parser.add_argument('--output-csv', help='Optional path to save ranked results.')
    args = parser.parse_args()

    log = load_history()
    log, usage, entries = compute_usage(log)
    stack_lookup, flag_stats = compute_stack_flags(log)
    low_var_bonus = compute_low_variance_bonus(log)
    pos_share = compute_pos_scarcity(log, entries)
    history_features_df = build_feature_matrix(log)

    history_features = {
        'PlayerMean': history_features_df.set_index('Player')['PlayerMean'].to_dict(),
        'PlayerOpponentMean': history_features_df.set_index(['Player', 'Opponent'])['PlayerOpponentMean'].dropna().to_dict(),
        'TeamMean': history_features_df.set_index(['TeamAbbrev', 'Position'])['TeamMean'].dropna().to_dict(),
        'TeamOpponentMean': history_features_df.set_index(['TeamAbbrev', 'Opponent', 'Position'])['TeamOpponentMean'].dropna().to_dict(),
        'TeamTotalMean': history_features_df.groupby('TeamAbbrev')['TeamTotalMean'].mean().to_dict(),
        'RecentAvg': history_features_df.set_index('Player')['RecentAvg'].to_dict(),
        'Rating': history_features_df.set_index('Player')['Rating'].to_dict(),
    }

    models, feature_means = fit_models(history_features_df)

    salary_path = Path(args.salary_csv)
    result_df = score_salary(salary_path, models, feature_means, usage, stack_lookup, flag_stats, low_var_bonus, pos_share, history_features)

    if result_df.empty:
        raise RuntimeError('No players scored; check salary file and history coverage.')

    result_df = result_df.sort_values('MaddenEdge', ascending=False)

    print("Top plays by position (MaddenEdge):")
    for position, group in result_df.groupby('Position'):
        print(f"\n{position}:")
        print(group.head(10)[['Player', 'TeamAbbrev', 'Salary', 'PredFPTS', 'MaddenEdge', 'StackBonus', 'OwnershipLeverage']].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    output_path = Path(args.output_csv) if args.output_csv else ROOT / 'data' / f"madden_future_rank_{salary_path.stem}.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved full rankings to {output_path}")


if __name__ == '__main__':
    main()


