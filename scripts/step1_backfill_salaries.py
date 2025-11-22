import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
contest_dir = root / "contest"
salary_dir = root / "salary"
salary_frames = [pd.read_csv(p) for p in salary_dir.glob("DKSalaries_*.csv")]
salary_all = pd.concat(salary_frames, ignore_index=True) if salary_frames else pd.DataFrame(columns=["Salary", "AvgPointsPerGame"])
ratio = salary_all["Salary"].div(salary_all["AvgPointsPerGame"].replace(0, pd.NA)).dropna().median()
ratio = ratio if pd.notna(ratio) else 300

for contest_path in sorted(contest_dir.glob("DKcontest_*.csv")):
    slate = contest_path.stem.split("DKcontest_")[1]
    target = salary_dir / f"DKSalaries_{slate}.csv"
    if target.exists():
        continue
    contest = pd.read_csv(contest_path).dropna(subset=["Player", "FPTS"])
    pos = contest.groupby("Player")["Roster Position"].agg(lambda x: x[x != "FLEX"].mode().iat[0] if (x != "FLEX").any() else "FLEX")
    avg = contest.groupby("Player")["FPTS"].mean().round(2)
    pos = pos.reindex(avg.index)
    frame = pd.DataFrame({"Name": avg.index, "Position": pos.values})
    frame["Name"] = frame["Name"].str.strip()
    frame["ID"] = frame["Name"].map(lambda x: abs(hash((slate, x))) % 10**8)
    frame["AvgPointsPerGame"] = avg.values
    frame["Salary"] = (frame["AvgPointsPerGame"] * ratio).round(-2).clip(lower=2000).astype(int)
    frame["Name + ID"] = frame.apply(lambda r: f"{r['Name']} ({int(r['ID'])})", axis=1)
    roster_map = {"QB": "QB", "RB": "RB/FLEX", "WR": "WR/FLEX", "TE": "TE/FLEX", "DST": "DST"}
    frame["Roster Position"] = frame["Position"].map(roster_map).fillna(frame["Position"])
    frame["Game Info"] = "TBD"
    frame["TeamAbbrev"] = "UNK"
    frame = frame[
        [
            "Position",
            "Name + ID",
            "Name",
            "ID",
            "Roster Position",
            "Salary",
            "Game Info",
            "TeamAbbrev",
            "AvgPointsPerGame",
        ]
    ]
    frame.to_csv(target, index=False)

