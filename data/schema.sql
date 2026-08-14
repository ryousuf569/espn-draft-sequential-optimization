CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    position    TEXT,
    birth_date  TEXT
);

CREATE TABLE IF NOT EXISTS season_stats (
    player_id  INTEGER NOT NULL,
    season     TEXT NOT NULL,
    gp         INTEGER,
    mpg        REAL,
    pts        INTEGER,
    reb        INTEGER,
    ast        INTEGER,
    stl        INTEGER,
    blk        INTEGER,
    tov        INTEGER,
    fg3m       INTEGER,
    fgm        INTEGER,
    fga        INTEGER,
    ftm        INTEGER,
    fta        INTEGER,
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS game_logs (
    player_id  INTEGER NOT NULL,
    game_id    TEXT NOT NULL,
    season     TEXT NOT NULL,
    date       TEXT NOT NULL,
    min        REAL,
    pts        INTEGER,
    reb        INTEGER,
    ast        INTEGER,
    stl        INTEGER,
    blk        INTEGER,
    tov        INTEGER,
    fg3m       INTEGER,
    fgm        INTEGER,
    fga        INTEGER,
    ftm        INTEGER,
    fta        INTEGER,
    PRIMARY KEY (player_id, game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS adp (
    player_id      INTEGER,
    adp_name       TEXT NOT NULL,
    season         TEXT NOT NULL,
    source         TEXT NOT NULL,
    adp            REAL NOT NULL,
    adp_sd         REAL,
    n_observations INTEGER,
    pulled_at      TEXT NOT NULL,
    PRIMARY KEY (season, source, adp_name)
);

CREATE TABLE IF NOT EXISTS draft_results (
    draft_id    TEXT NOT NULL,
    season      TEXT NOT NULL,
    pick_number INTEGER NOT NULL,
    player_id   INTEGER,
    team_slot   INTEGER NOT NULL,
    PRIMARY KEY (draft_id, pick_number),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS player_status (
    player_id  INTEGER PRIMARY KEY,
    team       TEXT,
    status     TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE INDEX IF NOT EXISTS idx_season_stats_season ON season_stats(season);
CREATE INDEX IF NOT EXISTS idx_game_logs_season    ON game_logs(season);
CREATE INDEX IF NOT EXISTS idx_game_logs_date      ON game_logs(date);
CREATE INDEX IF NOT EXISTS idx_adp_season          ON adp(season);
CREATE INDEX IF NOT EXISTS idx_adp_player          ON adp(player_id);
CREATE INDEX IF NOT EXISTS idx_draft_results_player ON draft_results(player_id, season);
