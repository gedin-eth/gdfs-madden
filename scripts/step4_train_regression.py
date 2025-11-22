import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / 'data' / 'madden_game_log.parquet'
RATINGS_PATH = ROOT / 'data' / 'madden_ratings.csv'

parser = argparse.ArgumentParser(description="Train per-position ridge models and score a salary slate.")
parser.add_argument('--salary-csv', required=True, help='Path to a DraftKings salary CSV to score.')
parser.add_argument('--output-csv', help='Optional path to write predictions (default data/madden_predictions_<slate>.csv).')
args = parser.parse_args()

SALARY_PATH = Path(args.salary_csv)

if not LOG_PATH.exists():
    raise FileNotFoundError("Run scripts/step4_build_game_log.py to create the Madden game log.")

log = pd.read_parquet(LOG_PATH).rename(columns={'Slate_x': 'Slate'})
if 'Slate_y' in log.columns:
    log = log.drop(columns=['Slate_y'])

log['FPTS'] = log['FPTS'].astype(float)
log['Salary'] = pd.to_numeric(log['Salary'], errors='coerce')
log['AvgPointsPerGame'] = pd.to_numeric(log['AvgPointsPerGame'], errors='coerce')

# Stack rate derived from entry composition
entry_team_counts = log.dropna(subset=['TeamAbbrev']).groupby(['EntryId', 'TeamAbbrev']).size().reset_index(name='TeamCount')
stack_base = log.dropna(subset=['TeamAbbrev']).merge(entry_team_counts, on=['EntryId', 'TeamAbbrev'], how='left')
stack_lookup = (stack_base['TeamCount'] > 1).groupby(stack_base['Player']).mean()

# Deduplicate player/slate rows (FPTS repeated across entries)
unique_cols = ['Player', 'Slate', 'TeamAbbrev', 'Position', 'Opponent', 'IsHome', 'Salary', 'AvgPointsPerGame', 'FPTS']
unique = log[unique_cols].drop_duplicates()
slates_sorted = sorted(unique['Slate'].unique())
holdout_slate = slates_sorted[-1] if slates_sorted else None

# Rating map
rating_map = {}
if RATINGS_PATH.exists():
    ratings_df = pd.read_csv(RATINGS_PATH)
    ratings_df['Player'] = ratings_df['Player'].astype(str).str.strip()
    rating_map = ratings_df.set_index('Player')['Rating'].to_dict()

# Historical aggregates
player_mean = unique.groupby('Player')['FPTS'].mean().to_dict()
player_opp_mean = unique.dropna(subset=['Opponent']).groupby(['Player', 'Opponent'])['FPTS'].mean().to_dict()
team_pos_mean = unique.dropna(subset=['TeamAbbrev']).groupby(['TeamAbbrev', 'Position'])['FPTS'].mean().to_dict()
team_pos_opp_mean = unique.dropna(subset=['TeamAbbrev', 'Opponent']).groupby(['TeamAbbrev', 'Opponent', 'Position'])['FPTS'].mean().to_dict()
team_total_mean = unique.groupby(['TeamAbbrev', 'Slate'])['FPTS'].sum().groupby('TeamAbbrev').mean().to_dict()

# Feature assembly
records = []
for row in unique.itertuples(index=False):
    slate = row.Slate
    records.append({
        'Player': row.Player,
        'Slate': slate,
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
        'Rating': rating_map.get(row.Player, np.nan),
        'IsHoldout': slate == holdout_slate,
    })

train_df = pd.DataFrame(records)

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

models = {}
feature_means = {}
def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, mae

metrics = []
holdout_metrics = []
holdout_residuals = []
cv_records = []

for position, group in train_df.groupby('Position'):
    if len(group) < 20:
        continue
    train_mask = ~group['IsHoldout']
    holdout_mask = group['IsHoldout']
    if train_mask.sum() < 10:
        continue
    y = group['FPTS']
    X_train = group.loc[train_mask, feature_cols].copy()
    pos_means = X_train.mean().fillna(0)
    X_train = X_train.fillna(pos_means)
    y_train = group.loc[train_mask, 'FPTS']
    model = Ridge(alpha=5.0)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    rmse, mae = compute_metrics(y_train, train_preds)
    metrics.append({'Position': position, 'Samples': len(y_train), 'RMSE': rmse, 'MAE': mae, 'TargetMean': y_train.mean()})
    models[position] = model
    feature_means[position] = pos_means
    if holdout_mask.any():
        X_holdout = group.loc[holdout_mask, feature_cols].copy()
        y_holdout = group.loc[holdout_mask, 'FPTS']
        X_holdout = X_holdout.fillna(pos_means)
        holdout_preds = model.predict(X_holdout)
        holdout_rmse, holdout_mae = compute_metrics(y_holdout, holdout_preds)
        holdout_metrics.append({'Position': position, 'Samples': len(y_holdout), 'RMSE': holdout_rmse, 'MAE': holdout_mae, 'TargetMean': y_holdout.mean()})
        holdout_section = group.loc[holdout_mask, ['Player', 'Slate']].copy()
        holdout_section['Position'] = position
        holdout_section['Actual'] = y_holdout.values
        holdout_section['Predicted'] = holdout_preds
        holdout_section['Residual'] = holdout_section['Actual'] - holdout_section['Predicted']
        holdout_residuals.append(holdout_section)

    # Cross-validation across remaining slates
    slates_in_group = sorted(group['Slate'].unique())
    for cv_slate in slates_in_group:
        cv_mask = group['Slate'] == cv_slate
        train_cv = group.loc[~cv_mask]
        test_cv = group.loc[cv_mask]
        if len(train_cv) < 10 or len(test_cv) == 0:
            continue
        X_cv_train = train_cv[feature_cols].copy()
        cv_means = X_cv_train.mean().fillna(0)
        X_cv_train = X_cv_train.fillna(cv_means)
        y_cv_train = train_cv['FPTS']
        cv_model = Ridge(alpha=5.0)
        cv_model.fit(X_cv_train, y_cv_train)
        X_cv_test = test_cv[feature_cols].copy().fillna(cv_means)
        y_cv_test = test_cv['FPTS']
        preds_cv = cv_model.predict(X_cv_test)
        rmse_cv, mae_cv = compute_metrics(y_cv_test, preds_cv)
        cv_records.append({
            'Position': position,
            'HoldoutSlate': cv_slate,
            'Samples': len(y_cv_test),
            'RMSE': rmse_cv,
            'MAE': mae_cv,
            'TargetMean': y_cv_test.mean(),
        })

metrics_df = pd.DataFrame(metrics).sort_values('Position')
print("Training metrics (in-sample):")
print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

if holdout_metrics:
    holdout_df = pd.DataFrame(holdout_metrics).sort_values('Position')
    print(f"\nHoldout metrics (slate {holdout_slate}):")
    print(holdout_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

if holdout_residuals:
    residual_df = pd.concat(holdout_residuals, ignore_index=True)
    top_misses = residual_df.reindex(residual_df['Residual'].abs().sort_values(ascending=False).index).head(10)
    print("\nTop holdout residuals:")
    print(top_misses.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

if cv_records:
    cv_df = pd.DataFrame(cv_records)
    cv_summary = cv_df.groupby('Position')[['RMSE', 'MAE']].mean().reset_index()
    print("\nTime-series CV (average per position):")
    print(cv_summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

if not SALARY_PATH.exists():
    raise FileNotFoundError(f"Salary file not found: {SALARY_PATH}")

salary = pd.read_csv(SALARY_PATH)
salary['Name'] = salary['Name'].astype(str).str.strip()
salary['Game Info'] = salary['Game Info'].astype(str)

def parse_matchup(row):
    info = row['Game Info']
    if not info or info == 'nan':
        return np.nan, np.nan
    part = info.split()[0]
    if '@' not in part:
        return np.nan, np.nan
    away, home = part.split('@')[:2]
    team = row['TeamAbbrev']
    if team == home:
        return 1.0, away
    if team == away:
        return 0.0, home
    return np.nan, np.nan

matchup = salary.apply(parse_matchup, axis=1, result_type='expand')
salary['IsHome'] = matchup[0]
salary['Opponent'] = matchup[1]

pred_records = []
for row in salary.itertuples():
    pos = row.Position.split('/')[0]
    model = models.get(pos)
    if model is None:
        continue
    pos_means = feature_means[pos]
    features = {
        'Salary': float(row.Salary),
        'AvgPointsPerGame': float(row.AvgPointsPerGame),
        'IsHome': row.IsHome,
        'StackRate': stack_lookup.get(row.Name, 0.0),
        'PlayerMean': player_mean.get(row.Name, np.nan),
        'PlayerOpponentMean': player_opp_mean.get((row.Name, row.Opponent), np.nan),
        'TeamMean': team_pos_mean.get((row.TeamAbbrev, pos), np.nan),
        'TeamOpponentMean': team_pos_opp_mean.get((row.TeamAbbrev, row.Opponent, pos), np.nan),
        'TeamTotalMean': team_total_mean.get(row.TeamAbbrev, np.nan),
        'Rating': rating_map.get(row.Name, np.nan),
    }
    feature_vector = pd.Series(features)
    feature_vector = feature_vector.reindex(feature_cols)
    feature_vector = feature_vector.fillna(pos_means).fillna(0)
    pred_input = pd.DataFrame([feature_vector], columns=feature_cols)
    pred = float(model.predict(pred_input)[0])
    pred_records.append({
        'Player': row.Name,
        'Position': pos,
        'TeamAbbrev': row.TeamAbbrev,
        'Salary': row.Salary,
        'Opponent': row.Opponent,
        'IsHome': row.IsHome,
        'PredictedFPTS': pred,
    })

pred_df = pd.DataFrame(pred_records)
pred_df = pred_df.sort_values('PredictedFPTS', ascending=False)

output_pred = Path(args.output_csv) if args.output_csv else ROOT / 'data' / f"madden_predictions_{SALARY_PATH.stem}.csv"
print(f"\nPredictions for {SALARY_PATH.stem} (top 20):")
print(pred_df.head(20).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

pred_df.to_csv(output_pred, index=False)
print(f"Saved predictions to {output_pred}")

