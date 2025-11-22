import pandas as pd
import numpy as np
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pattern = re.compile(r"(DST|FLEX|QB|RB|WR|TE)\s+")

entry_rows = []
player_rows = []
for contest_path in sorted((root / "contest").glob("DKcontest_*.csv")):
    slate = contest_path.stem.split("DKcontest_")[1]
    contest = pd.read_csv(contest_path)
    contest = contest[['EntryId', 'Points', 'Lineup']].dropna()
    for _, row in contest.iterrows():
        entry_rows.append({'EntryId': row['EntryId'], 'Points': row['Points'], 'Slate': slate})
        lineup = str(row['Lineup'])
        matches = list(pattern.finditer(lineup))
        for i, match in enumerate(matches):
            slot = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(lineup)
            name = lineup[start:end].strip()
            if name:
                player_rows.append({'EntryId': row['EntryId'], 'Slate': slate, 'Slot': slot, 'Player': name})

entries = pd.DataFrame(entry_rows).drop_duplicates('EntryId')
players = pd.DataFrame(player_rows)
players['Player'] = players['Player'].str.strip()
players = players[~players['Player'].str.upper().fillna('').eq('LOCKED')]

salary = pd.concat(
    pd.read_csv(f).assign(Slate=f.stem.split("DKSalaries_")[1])
    for f in sorted((root / "salary").glob("DKSalaries_*.csv"))
)
salary['Name'] = salary['Name'].astype(str).str.strip()
players = players.merge(
    salary[['Name', 'Position', 'TeamAbbrev', 'Salary', 'Game Info', 'Slate']],
    left_on=['Player', 'Slate'],
    right_on=['Name', 'Slate'],
    how='left'
)
players['TeamAbbrev'] = players['TeamAbbrev'].replace('UNK', pd.NA)
players[['Salary']] = players[['Salary']].astype(float)

entries = entries.dropna(subset=['Points'])
entries['Rank'] = entries.groupby('Slate')['Points'].rank(ascending=False, method='first')
entries['Cut'] = (entries.groupby('Slate')['EntryId'].transform('count') * 0.01).clip(lower=1)
top_ids = set(entries[entries['Rank'] <= entries['Cut']]['EntryId'])

info = players.dropna(subset=['Game Info', 'TeamAbbrev']).copy()
split = info['Game Info'].str.split().str[0].str.split('@')
info['Away'] = split.str[0]
info['Home'] = split.str[1]
info['OppTeam'] = info.apply(lambda r: r['Home'] if r['TeamAbbrev'] == r['Away'] else r['Away'], axis=1)
players = players.merge(info[['EntryId', 'Player', 'OppTeam']], on=['EntryId', 'Player'], how='left')

field_groups = list(players.groupby('EntryId'))
top_groups = [(eid, grp) for eid, grp in field_groups if eid in top_ids]


def qb_stack(grp):
    qb = grp.loc[(grp['Position'] == 'QB') & grp['TeamAbbrev'].notna(), 'TeamAbbrev']
    if qb.empty:
        return None
    catch = grp.loc[(grp['Position'].isin(['WR', 'TE'])) & grp['TeamAbbrev'].notna(), 'TeamAbbrev']
    if catch.empty:
        return None
    return catch.isin(qb.unique()).any()


def bring_back(grp):
    qb_rows = grp[(grp['Position'] == 'QB') & grp['TeamAbbrev'].notna()]
    catch = grp[(grp['Position'].isin(['WR', 'TE'])) & grp['TeamAbbrev'].notna()]
    if qb_rows.empty or catch.empty:
        return None
    for _, qb in qb_rows.iterrows():
        team = qb.get('TeamAbbrev')
        opp = qb.get('OppTeam')
        if pd.isna(team) or pd.isna(opp):
            continue
        if catch[catch['TeamAbbrev'] == team].empty:
            continue
        if not catch[catch['TeamAbbrev'] == opp].empty:
            return True
    return False


def rb_dst(grp):
    rb_teams = set(grp.loc[(grp['Position'] == 'RB') & grp['TeamAbbrev'].notna(), 'TeamAbbrev'])
    dst_teams = set(grp.loc[(grp['Position'] == 'DST') & grp['TeamAbbrev'].notna(), 'TeamAbbrev'])
    if not rb_teams or not dst_teams:
        return None
    return bool(rb_teams & dst_teams)


def metric(groups, func):
    values = [func(grp) for _, grp in groups]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float('nan')

print("Stack rates (Top1 / Field):")
print("  QB+pass-catcher:", round(metric(top_groups, qb_stack), 3), '/', round(metric(field_groups, qb_stack), 3))
print("  RB+DST:", round(metric(top_groups, rb_dst), 3), '/', round(metric(field_groups, rb_dst), 3))
stacked_top = [(eid, grp) for eid, grp in top_groups if qb_stack(grp)]
print("  Bring-back on stack:", round(metric(stacked_top, bring_back), 3))

salary_view = players[players['Salary'].notna()]
print("\nTop1 salary profile by roster slot:")
print(salary_view[salary_view['EntryId'].isin(top_ids)].groupby('Slot')['Salary'].agg(['mean', 'min', 'max']).round(0))

std_top = salary_view[salary_view['EntryId'].isin(top_ids)].groupby('EntryId')['Salary'].std().dropna()
std_field = salary_view.groupby('EntryId')['Salary'].std().dropna()
print("\nSalary spread (higher = more stars/scrubs):", round(std_top.mean(), 1), "(Top1)", '/', round(std_field.mean(), 1), "(Field)")

total_entries = entries['EntryId'].nunique()
top_total = len(top_ids)
field_counts = players.groupby('Player')['EntryId'].nunique()
top_counts = players[players['EntryId'].isin(top_ids)].groupby('Player')['EntryId'].nunique()
usage = field_counts.to_frame('FieldCount').join(top_counts.to_frame('TopCount'), how='left').fillna(0)
usage['FieldRate'] = usage['FieldCount'] / total_entries
usage['TopRate'] = usage['TopCount'] / (top_total if top_total else 1)
usage['Lift'] = (usage['TopRate'] / usage['FieldRate'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

total_field_exposures = usage['FieldCount'].sum()
total_top_exposures = usage['TopCount'].sum()

bins = [0, 0.05, 0.1, 0.2, 0.3, 1]
labels = ['<5%', '5-10%', '10-20%', '20-30%', '30%+']
usage['Bucket'] = pd.cut(usage['FieldRate'], bins=bins, labels=labels, include_lowest=True, right=False)
bucket = usage.groupby('Bucket')[['FieldCount', 'TopCount']].sum()
bucket['FieldShare'] = bucket['FieldCount'] / (total_field_exposures if total_field_exposures else 1)
bucket['TopShare'] = bucket['TopCount'] / (total_top_exposures if total_top_exposures else 1)
bucket['Lift'] = (bucket['TopShare'] / bucket['FieldShare']).replace([np.inf, -np.inf], np.nan).round(2)
print("\nOwnership bucket share (Top1 vs Field):")
print(bucket[['TopShare', 'FieldShare', 'Lift']].round(3))

traps = usage[(usage['FieldRate'] >= 0.1) & usage['Lift'].notna()]
traps = traps[traps['Lift'] < 1].sort_values('Lift').head(5)
print("\nPotential trap plays (highest field rate, lowest Top1 lift):")
print(traps[['FieldRate', 'TopRate', 'Lift']].round(3))

