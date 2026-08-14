from bs4 import BeautifulSoup
import argparse
import re
import requests
import time
import unicodedata

import sqlite_helpers
from config import ADP_SLEEP, RAW_DIR, adp_seasons, season_str

URL = "https://www.fantasypros.com/nba/adp/overall.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# "Russell Westbrook III(FA - PG,SG)RET" -> "Russell Westbrook III"
NAME_RE = re.compile(r"^(.*?)\s*\(")


def clean_name(text):
    match = NAME_RE.match(text)
    return (match.group(1) if match else text).strip()


# spread across the sites backing a row, None when too few to compute
def stdev(values):
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return round(var**0.5, 3)


def fetch_year(year):
    want = season_str(year)
    resp = requests.get(URL, params={"year": year}, headers=HEADERS, timeout=30)

    if resp.status_code != 200:
        print(f"  {want}: HTTP {resp.status_code}, skipping")
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"adp_{year}.html").write_text(resp.text, encoding="utf-8")
    soup = BeautifulSoup(resp.text, "lxml")

    # DO NOT REMOVE. FantasyPros fails soft: year=2012 returns HTTP 200 carrying
    # the CURRENT season's board, which would silently corrupt a backtest with
    # no error anywhere. The season lives in the page title, so check it.
    title = soup.title.get_text(strip=True) if soup.title else ""
    if want not in title:
        print(f"  {want}: REJECTED, page is for another season ({title[:60]})")
        return []

    table = soup.select_one("table#data")
    if not table:
        print(f"  {want}: no ADP table found")
        return []

    # column count varies by season (some years carry CBS, some do not), so read
    # by header name and never by index
    headers = [th.get_text(strip=True) for th in table.select("thead th")]
    try:
        name_idx = headers.index("Player")
        avg_idx = headers.index("AVG")
    except ValueError:
        print(f"  {want}: unexpected columns {headers}")
        return []

    # every column between Player and AVG is one site's ADP
    source_idx = list(range(name_idx + 1, avg_idx))
    pulled_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    rows = []
    for tr in table.select("tbody tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) <= avg_idx:
            continue
        try:
            adp = float(cells[avg_idx])
        except ValueError:
            continue

        per_site = []
        for i in source_idx:
            try:
                per_site.append(float(cells[i]))
            except (ValueError, IndexError):
                pass

        # team is never stored: historical rows are joined to TODAY's team, so
        # 2015-16 lists Harden as CLE. only name and ADP are trustworthy.
        rows.append(
            {
                "player_id": None,  # filled in by link_players
                "adp_name": clean_name(cells[name_idx]),
                "season": want,
                "source": "fantasypros_avg",
                "adp": adp,
                "adp_sd": stdev(per_site),
                "n_observations": len(per_site),
                "pulled_at": pulled_at,
            }
        )

    print(f"  {want}: {len(rows)} players ({len(source_idx)} sources)")
    return rows


def normalize(name):
    # strip accents: nba_api writes "Nikola Jokić", FantasyPros writes "Jokic",
    # and without this the best players on the board fail to link
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().replace(".", "").replace("'", "").replace("-", " ")
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", name.strip())
    return re.sub(r"\s+", " ", name)


# ADP sources publish names, not ids, so match them to nba_api player_id
def link_players(conn):
    lookup = {}
    for pid, name in conn.execute("SELECT player_id, name FROM players"):
        lookup.setdefault(normalize(name), pid)

    unmatched = []
    for (adp_name,) in conn.execute(
        "SELECT DISTINCT adp_name FROM adp WHERE player_id IS NULL"
    ).fetchall():
        pid = lookup.get(normalize(adp_name))
        if pid:
            conn.execute(
                "UPDATE adp SET player_id = ? WHERE adp_name = ?", (pid, adp_name)
            )
        else:
            # leave it NULL rather than guessing, adp_name keeps the original
            unmatched.append(adp_name)

    conn.commit()

    total = sqlite_helpers.count(conn, "adp")
    linked = conn.execute(
        "SELECT COUNT(*) FROM adp WHERE player_id IS NOT NULL"
    ).fetchone()[0]
    print(f"linked {linked}/{total} adp rows")
    if unmatched:
        print(f"  unmatched names ({len(unmatched)}): {', '.join(unmatched[:10])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, help="start years, e.g. 2024 2025")
    args = ap.parse_args()

    years = args.years or adp_seasons()
    conn = sqlite_helpers.connect()
    sqlite_helpers.init(conn)

    print(f"fetching ADP for {len(years)} seasons")
    for i, year in enumerate(years):
        # robots.txt asks for Crawl-delay: 5
        if i:
            time.sleep(ADP_SLEEP)
        sqlite_helpers.upsert(conn, "adp", fetch_year(year))

    link_players(conn)

    seasons = conn.execute("SELECT COUNT(DISTINCT season) FROM adp").fetchone()[0]
    print(f"adp: {sqlite_helpers.count(conn, 'adp')} rows across {seasons} seasons")
    conn.close()


if __name__ == "__main__":
    main()
