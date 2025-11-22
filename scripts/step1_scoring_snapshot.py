import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
contest = pd.concat(
    pd.read_csv(f).assign(Slate=f.stem.split("DKcontest_")[1])
    for f in sorted((root / "contest").glob("DKcontest_*.csv"))
)
contest = contest.rename(columns={"Roster Position": "RosterPosition"}).drop(columns=[c for c in contest.columns if c.startswith("Unnamed")])
contest = contest.dropna(subset=["FPTS"])
salary = pd.concat(
    pd.read_csv(f).assign(Slate=f.stem.split("DKSalaries_")[1])
    for f in sorted((root / "salary").glob("DKSalaries_*.csv"))
)
dataset = contest.merge(
    salary[["Name", "Salary", "TeamAbbrev", "Game Info", "Position", "AvgPointsPerGame", "Slate"]],
    left_on=["Player", "Slate"],
    right_on=["Name", "Slate"],
    how="left",
)
dataset["FPTS"] = dataset["FPTS"].astype(float)
dataset["Salary"] = dataset["Salary"].astype(float)
dataset["PositionGap"] = dataset["FPTS"] / dataset["Salary"] * 1000
pos_view = dataset.groupby("RosterPosition")[["FPTS", "Salary", "PositionGap"]].mean().round(2)
home_flag = dataset.dropna(subset=["Game Info"]).assign(
    HomeTeam=lambda x: x["Game Info"].str.split("@").str[1].str.split().str[0],
)
home_flag["IsHome"] = (home_flag["TeamAbbrev"] == home_flag["HomeTeam"]).astype(int)
home_view = home_flag.groupby("IsHome")["FPTS"].mean().round(2).rename({0: "Away", 1: "Home"})
print("Position scoring snapshot:\n", pos_view)
print("\nHome vs away FPTS:\n", home_view)

