import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
salary_dir = root / "salary"
salary_files = sorted(salary_dir.glob("DKSalaries_*.csv"))

player_team = {}
for path in salary_files:
    df = pd.read_csv(path)
    known = df[df["TeamAbbrev"].notna() & (df["TeamAbbrev"] != "UNK")]
    for name, team in zip(known["Name"], known["TeamAbbrev"]):
        player_team.setdefault(name.strip(), team.strip())

valid_teams = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
}

for path in salary_files:
    df = pd.read_csv(path)
    if not ((df["TeamAbbrev"] == "UNK") | df["TeamAbbrev"].isna()).any():
        continue
    fill = df["Name"].map(lambda x: player_team.get(str(x).strip()))
    df.loc[df["TeamAbbrev"].isin(["UNK", None]) | df["TeamAbbrev"].isna(), "TeamAbbrev"] = fill
    df.loc[~df["TeamAbbrev"].isin(valid_teams), "TeamAbbrev"] = pd.NA
    df.to_csv(path, index=False)

