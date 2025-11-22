import pandas as pd
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://dknetwork.draftkings.com/2025/09/03/draftkings-madden-26-stream-{}-2026-depth-chart/"
TEAM_SLUGS = {
    'ARI': 'arizona-cardinals',
    'ATL': 'atlanta-falcons',
    'BAL': 'baltimore-ravens',
    'BUF': 'buffalo-bills',
    'CAR': 'carolina-panthers',
    'CHI': 'chicago-bears',
    'CIN': 'cincinnati-bengals',
    'CLE': 'cleveland-browns',
    'DAL': 'dallas-cowboys',
    'DEN': 'denver-broncos',
    'DET': 'detroit-lions',
    'GB': 'green-bay-packers',
    'HOU': 'houston-texans',
    'IND': 'indianapolis-colts',
    'JAX': 'jacksonville-jaguars',
    'KC': 'kansas-city-chiefs',
    'LV': 'las-vegas-raiders',
    'LAC': 'los-angeles-chargers',
    'LAR': 'los-angeles-rams',
    'MIA': 'miami-dolphins',
    'MIN': 'minnesota-vikings',
    'NE': 'new-england-patriots',
    'NO': 'new-orleans-saints',
    'NYG': 'new-york-giants',
    'NYJ': 'new-york-jets',
    'PHI': 'philadelphia-eagles',
    'PIT': 'pittsburgh-steelers',
    'SF': 'san-francisco-49ers',
    'SEA': 'seattle-seahawks',
    'TB': 'tampa-bay-buccaneers',
    'TEN': 'tennessee-titans',
    'WAS': 'washington-commanders',
}

records = []
for abbrev, slug in TEAM_SLUGS.items():
    url = BASE_URL.format(slug)
    print(f"Fetching {abbrev} from {url}")
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        print(f"  Warning: 404 Not Found for {abbrev}, skipping...")
        continue
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    depth_table = None
    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(['th', 'td'])]
            if cells:
                rows.append(cells)
        if not rows:
            continue
        header = [c.lower() for c in rows[0]]
        has_header = {'position', 'player', 'rating'} <= set(header) or {'team', 'player', 'rating'} <= set(header)
        if has_header:
            cols = [c.title() for c in rows[0]]
            df = pd.DataFrame(rows[1:], columns=cols)
            if 'Team' in df.columns and 'Position' not in df.columns:
                df = df.rename(columns={'Team': 'Position'})
        else:
            if len(rows[0]) != 3:
                continue
            df = pd.DataFrame(rows, columns=['Position', 'Player', 'Rating'])
        if {'Position', 'Player', 'Rating'} <= set(df.columns):
            depth_table = df[['Position', 'Player', 'Rating']]
            break
    if depth_table is None:
        print(f"No depth table found for {abbrev}")
        continue
    depth_table['Team'] = abbrev
    records.append(depth_table)

if not records:
    raise RuntimeError("No ratings captured; check the source URLs.")

ratings = pd.concat(records, ignore_index=True)
ratings['Rating'] = pd.to_numeric(ratings['Rating'], errors='coerce')
ratings = ratings.dropna(subset=['Rating'])
output = Path(__file__).resolve().parents[1] / 'data' / 'madden_ratings.csv'
ratings.to_csv(output, index=False)
print(f"Saved ratings for {ratings['Team'].nunique()} teams to {output}")
