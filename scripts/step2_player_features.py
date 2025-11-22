import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
contest = pd.concat(pd.read_csv(f).assign(Slate=f.stem.split("DKcontest_")[1]) for f in sorted((root / "contest").glob("DKcontest_*.csv")))
contest = contest.rename(columns={"Roster Position": "RosterPosition"}).drop(columns=[c for c in contest.columns if c.startswith("Unnamed")])
contest["Ownership"] = contest["%Drafted"].str.rstrip("%").astype(float) / 100
salary = pd.concat(pd.read_csv(f).assign(Slate=f.stem.split("DKSalaries_")[1]) for f in sorted((root / "salary").glob("DKSalaries_*.csv")))
dataset = contest.merge(salary[["Name", "TeamAbbrev", "Game Info", "Position", "AvgPointsPerGame", "Salary", "Slate"]], left_on=["Player", "Slate"], right_on=["Name", "Slate"], how="left")
dataset = dataset.dropna(subset=["FPTS", "Salary"])
dataset["FPTS"] = dataset["FPTS"].astype(float)
dataset["Salary"] = dataset["Salary"].astype(float)

lineups = contest[["EntryId", "Points", "Slate"]].drop_duplicates()
lineups["Rank"] = lineups.groupby("Slate")["Points"].rank(ascending=False, method="first")
lineups["Cut"] = (lineups.groupby("Slate")["EntryId"].transform("count") * 0.01).clip(lower=1)
top_ids = set(lineups[lineups["Rank"] <= lineups["Cut"]]["EntryId"])
dataset["IsTop1"] = dataset["EntryId"].isin(top_ids)

dataset["SalaryEff"] = dataset["FPTS"].astype(float) / dataset["Salary"].astype(float)
dataset = dataset.sort_values(["Player", "Slate"])
N = 5
eff = dataset.groupby("Player").tail(N).groupby("Player")["SalaryEff"].mean().reset_index(name="SalaryEfficiency")
vol = dataset.groupby("Player")["FPTS"].std().fillna(0).reset_index(name="Volatility")

own = dataset.groupby("Player")["Ownership"].apply(lambda s: s.replace(0, pd.NA).mean()).rename("MeanOwnership")
top = dataset.groupby("Player")["IsTop1"].mean().rename("Top1Rate")
top_rate = top.to_frame().join(own)
top_rate["MeanOwnership"] = top_rate["MeanOwnership"].apply(lambda v: v if pd.notna(v) and v > 0 else 1)
top_rate["Top1AdjRate"] = top_rate["Top1Rate"] / top_rate["MeanOwnership"]
top_rate = top_rate[["Top1AdjRate"]].reset_index()

game_info = dataset.dropna(subset=["Game Info", "TeamAbbrev"]).copy()
split = game_info["Game Info"].str.split().str[0].str.split("@")
game_info.loc[:, "Away"], game_info.loc[:, "Home"] = split.str[0], split.str[1]
game_info.loc[:, "OppTeam"] = game_info.apply(lambda r: r["Home"] if r["TeamAbbrev"] == r["Away"] else r["Away"], axis=1)
opp_rating = game_info.groupby(["OppTeam", "RosterPosition"])['FPTS'].mean().reset_index(name="OppDefenseRating")
game_info = game_info.merge(opp_rating, on=["OppTeam", "RosterPosition"], how="left")

team_avg = dataset.dropna(subset=["TeamAbbrev"]).groupby(["Slate", "TeamAbbrev"])['FPTS'].mean().reset_index(name="TeamAvgFPTS")
team_env = game_info[["Slate", "TeamAbbrev", "OppTeam"]].drop_duplicates()
team_env = team_env.merge(team_avg, on=["Slate", "TeamAbbrev"], how="left")
opp_avg = team_avg.rename(columns={"TeamAbbrev": "OppTeam", "TeamAvgFPTS": "OppAvgFPTS"})
team_env = team_env.merge(opp_avg, on=["Slate", "OppTeam"], how="left")
env_cols = team_env[["Slate", "TeamAbbrev", "TeamAvgFPTS"]]

player_base = dataset[["Player", "Slate", "TeamAbbrev"]].drop_duplicates()
opp_view = game_info.groupby("Player")["OppDefenseRating"].mean().reset_index()
features = (player_base
            .merge(eff, on="Player", how="left")
            .merge(vol, on="Player", how="left")
            .merge(top_rate, on="Player", how="left")
            .merge(opp_view, on="Player", how="left")
            .merge(env_cols, on=["Slate", "TeamAbbrev"], how="left"))
features = features[["Player", "Slate", "SalaryEfficiency", "Volatility", "Top1AdjRate", "OppDefenseRating", "TeamAvgFPTS"]].sort_values(["Slate", "Player"])
print(features.head())

