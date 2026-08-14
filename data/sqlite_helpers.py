import sqlite3

from config import DB_PATH, SCHEMA_PATH


def connect(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(conn):
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


# insert rows, overwriting any that collide on the primary key
def upsert(conn, table, rows):
    if not rows:
        return 0

    # every row must have the same keys
    cols = list(rows[0])
    sql = (
        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
        f"VALUES ({','.join('?' for _ in cols)})"
    )

    conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
    conn.commit()
    return len(rows)


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
