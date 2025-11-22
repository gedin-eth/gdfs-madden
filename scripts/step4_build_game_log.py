import pandas as pd
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pattern = re.compile(r"(DST|FLEX|QB|RB|WR|TE)\s+")

entries, rows = [], []
for contest_path in sorted((root / 'contest').glob('DKcontest_*.csv')):
    slate = contest_path.stem.split('DKcontest_')[1]
    contest = pd.read_csv(contest_path)
    contest = contest[['EntryId', 'Points', 'Lineup']].dropna(subset=['Lineup'])
    for _, row in contest.iterrows():
        entry_id = row['EntryId']
        entries.append({'EntryId': entry_id, 'Slate': slate, 'LineupScore': row['Points']})
        lineup = str(row['Lineup'])
        matches = list(pattern.finditer(lineup))
        for i, match in enumerate(matches):
            slot = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(lineup)
            name = lineup[start:end].strip()
            if name and name.upper() != 'LOCKED':
                rows.append({'EntryId': entry_id, 'Slate': slate, 'Slot': slot, 'Player': name})

entries_df = pd.DataFrame(entries).drop_duplicates('EntryId')
players_df = pd.DataFrame(rows)

salary = pd.concat(
    pd.read_csv(f).assign(Slate=f.stem.split('DKSalaries_')[1])
    for f in sorted((root / 'salary').glob('DKSalaries_*.csv'))
)
salary['Name'] = salary['Name'].astype(str).str.strip()
salary = salary.rename(columns={'Game Info': 'GameInfo'})
players_df = players_df.merge(
    salary[['Name', 'Position', 'TeamAbbrev', 'Salary', 'AvgPointsPerGame', 'GameInfo', 'Slate']],
    left_on=['Player', 'Slate'],
    right_on=['Name', 'Slate'],
    how='left'
)

contest_scores = []
for contest_path in sorted((root / 'contest').glob('DKcontest_*.csv')):
    slate = contest_path.stem.split('DKcontest_')[1]
    df = pd.read_csv(contest_path)[['Player', 'FPTS']].dropna()
    df['Player'] = df['Player'].astype(str).str.strip()
    df['Slate'] = slate
    contest_scores.append(df)
actual = pd.concat(contest_scores, ignore_index=True)
players_df = players_df.merge(actual.groupby(['Player', 'Slate'])['FPTS'].mean().reset_index(), on=['Player', 'Slate'], how='left')

players_df['Salary'] = players_df['Salary'].astype(float)
players_df['AvgPointsPerGame'] = players_df['AvgPointsPerGame'].astype(float)
players_df['FPTS'] = players_df['FPTS'].astype(float)

players_df['TeamAbbrev'] = players_df['TeamAbbrev'].replace('UNK', pd.NA)
split = players_df['GameInfo'].fillna('').str.split().str[0].str.split('@')
players_df['GameAway'] = split.str[0]
players_df['GameHome'] = split.str[1]
players_df['Opponent'] = players_df.apply(
    lambda r: r['GameHome'] if r['TeamAbbrev'] == r['GameAway'] else r['GameAway'] if r['GameHome'] and r['GameAway'] else pd.NA,
    axis=1
)

players_df['IsHome'] = players_df.apply(
    lambda r: 1 if pd.notna(r['TeamAbbrev']) and r['TeamAbbrev'] == r['GameHome'] else 0 if pd.notna(r['TeamAbbrev']) and r['TeamAbbrev'] == r['GameAway'] else pd.NA,
    axis=1
)

log = players_df[['EntryId', 'Slate', 'Player', 'Slot', 'Position', 'TeamAbbrev', 'Opponent', 'IsHome', 'Salary', 'AvgPointsPerGame', 'FPTS']]
log = log.merge(entries_df, on='EntryId', how='left')
log = log[log['Player'].notna()].reset_index(drop=True)

output = root / 'data' / 'madden_game_log.parquet'
log.to_parquet(output, index=False)
print(f"Saved game log to {output} ({len(log)} rows)")
