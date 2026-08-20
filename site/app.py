# Live draft assistant. One modeled number: the chance a player is still on the
# board at your next turn. Everything else is a manual pick tracker.
#
# That number is S(k)/S(j) out of board.json -- the conditional, never raw S(k).
# See availability() for why the ratio is the only honest form of the question.

import json
import os
from pathlib import Path

# Streamlit reads its theme from config at import time, and the deploy runs the
# bare start command with no config.toml -- so the dark base is set here, before
# the import, rather than by chasing each widget's testid in CSS.
os.environ.setdefault("STREAMLIT_THEME_BASE", "dark")
os.environ.setdefault("STREAMLIT_THEME_BACKGROUND_COLOR", "#141311")
os.environ.setdefault("STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR", "#211f1d")
os.environ.setdefault("STREAMLIT_THEME_PRIMARY_COLOR", "#ffb68e")
os.environ.setdefault("STREAMLIT_THEME_TEXT_COLOR", "#e6e2de")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st

BOARD_PATH = Path(__file__).resolve().parent / "board.json"

ROWS = 12          # what fits above the fold; a draft needs the top of the board
WONT_LAST = 5

POSITION_SLOTS = ("PG", "SG", "SF", "PF", "C")
DEFAULT_ROSTER = (("PG", 1), ("SG", 1), ("SF", 1), ("PF", 1), ("C", 1),
                  ("Flex", 2), ("Bench", 5))

# Slot -> the exported positions that may fill it. The database carries G/F/C and
# hyphenates, not PG/SG/SF/PF, so a guard is eligible at either guard seat.
SLOT_ELIGIBLE = {
    "PG": ("G", "G-F", "F-G"),
    "SG": ("G", "G-F", "F-G"),
    "SF": ("F", "F-G", "G-F", "F-C"),
    "PF": ("F", "F-C", "C-F", "F-G"),
    "C": ("C", "C-F", "F-C"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Inter:wght@400&display=swap');

#MainMenu, footer, header {visibility: hidden;}
.stAppDeployButton {display: none;}

.stApp {background: #141311;}
.block-container {padding: 12px 12px 48px 12px; max-width: 1100px;}

[data-testid="stSidebar"] {background: #1c1b19; border-right: 1px solid #554339;}

.panel {background: #211f1d; border: 1px solid #554339; border-radius: 4px; padding: 10px 12px;}

.pickno {font-family: 'Barlow Condensed'; font-weight: 700; font-size: 48px; line-height: 48px; letter-spacing: -0.02em; color: #ffb68e; font-feature-settings: 'tnum'; margin: 0;}
.next {font-family: 'Barlow Condensed'; font-weight: 600; font-size: 20px; line-height: 24px; color: #e6e2de; font-feature-settings: 'tnum'; margin: 0;}
.caps {font-family: 'Barlow Condensed'; font-weight: 700; font-size: 12px; line-height: 16px; letter-spacing: 0.05em; text-transform: uppercase; color: #cac6bd; margin: 0;}
.note {font-family: 'Inter'; font-weight: 400; font-size: 14px; line-height: 20px; color: #dbc1b5; margin: 0;}

.bar {height: 2px; background: #554339; border-radius: 0; margin-top: 10px; width: 100%; display: block;}
.fill {height: 2px; background: #ffb68e; border-radius: 0; display: block;}

.name {font-family: 'Barlow Condensed'; font-weight: 600; font-size: 20px; line-height: 24px; color: #e6e2de; margin: 0;}
.team {font-family: 'Barlow Condensed'; font-weight: 600; color: #cac6bd; font-feature-settings: 'tnum';}
.cell {font-family: 'Barlow Condensed'; font-weight: 600; font-size: 20px; line-height: 24px; color: #cac6bd; font-feature-settings: 'tnum'; margin: 0;}
.gone {font-family: 'Barlow Condensed'; font-weight: 700; font-size: 20px; line-height: 24px; color: #e6e2de; font-feature-settings: 'tnum'; margin: 0;}
.lasts {font-family: 'Inter'; font-weight: 400; font-size: 16px; line-height: 24px; color: #dbc1b5; font-feature-settings: 'tnum'; margin: 0;}

/* Header line: a flex row whose cells match the row button's cells below. */
.head {display: flex; align-items: baseline; gap: 12px; padding: 8px 10px; border-bottom: 1px solid #554339;}
.col-name {flex: 1 1 auto; min-width: 0; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.col-pos {flex: 0 0 46px; text-align: left;}
.col-adp {flex: 0 0 38px; text-align: right;}
.col-avail {flex: 0 0 58px; text-align: right;}

.slot {font-family: 'Barlow Condensed'; font-weight: 600; font-size: 16px; line-height: 22px; padding: 3px 0; border-bottom: 1px solid #554339; font-feature-settings: 'tnum';}
.open {color: #554339;}
.full {color: #e6e2de;}

.stButton > button {background: #2b2a28; color: #e6e2de; border: 1px solid #554339; border-radius: 2px; font-family: 'Barlow Condensed'; font-weight: 700; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; padding: 6px 8px; width: 100%; box-shadow: none; white-space: nowrap;}
.stButton > button:hover {background: #363432; border-color: #a38c80; color: #e6e2de;}
.stButton > button:active {background: #1c1b19;}
.stButton > button:disabled {color: #554339; border-color: #554339; background: #1c1b19;}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {background: #ffb68e !important; color: #532200 !important; border: 1px solid #ffb68e;}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {background: #d8773a !important; color: #491d00 !important; border-color: #d8773a;}

/* The action bar stays horizontal on a phone: st.columns stacks below ~640px. */
/* A column narrower than this wraps to its own line instead of being squeezed:
   the board/panel split becomes one column on a phone. */
[data-testid="stColumn"] {min-width: 260px;}
[data-testid="stNumberInput"] {min-width: 56px;}
[data-testid="stNumberInput"] input {min-width: 32px; text-align: center;}
[data-testid="stWidgetLabel"] {white-space: nowrap;}

/* A board row is a flex line plus a transparent button pulled up on top of it:
   the whole row is one tap target while the cells stay a real grid. Streamlit
   stamps st-key-<key> on the container, the only stable per-widget hook, and
   every row key is p<player_id>. */
.rowline {display: flex; align-items: baseline; gap: 12px; white-space: nowrap; padding: 9px 10px 9px 10px; border-bottom: 1px solid #554339; position: relative; z-index: 0;}
.rowline.sel {border-left: 4px solid #6796b4; padding-left: 6px;}

/* the right-hand panel: same ticker line, no name column to compete with */
.side {padding: 10px 12px;}
.rowline.side {padding: 7px 0; gap: 6px;}
.rowline.side .col-adp {flex: 0 0 26px;}
.rowline.side .col-avail {flex: 0 0 46px;}
.rowline.side:last-of-type {border-bottom: 0;}
.note.foot {margin-top: 10px; padding-top: 8px; border-top: 1px solid #554339;}

[class*="st-key-p"] {margin: -47px 0 0 0 !important; height: 47px; width: 100%;}
[class*="st-key-p"] .stButton {width: 100%;}
[class*="st-key-p"] .stButton > button {background: transparent; border: 0; border-radius: 0; height: 47px; width: 100%; padding: 0; color: transparent; position: relative; z-index: 1;}
[class*="st-key-p"] .stButton > button:hover {background: rgba(230, 226, 222, 0.05); border: 0; color: transparent;}
[class*="st-key-p"] .stButton > button p {visibility: hidden;}

/* Inputs: the testids differ across Streamlit versions, so both the old
   baseweb wrapper and the current stNumberInputContainer are covered. The
   base palette itself comes from [theme] in config.toml. */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {background: #141311; color: #e6e2de; font-family: 'Inter'; font-size: 14px;}
[data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"], div[data-baseweb="input"] {background: #141311; border: 1px solid #554339; border-radius: 2px;}
[data-testid="stNumberInputContainer"]:focus-within, [data-testid="stTextInputRootElement"]:focus-within {border-color: #ffb68e;}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {background: #211f1d; color: #cac6bd; border-left: 1px solid #554339;}
[data-testid="stNumberInputStepUp"]:hover, [data-testid="stNumberInputStepDown"]:hover {background: #2b2a28; color: #ffb68e;}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stSelectbox"] > div > div {background: #141311; border: 1px solid #554339; border-radius: 2px; color: #e6e2de; font-family: 'Inter'; font-size: 14px;}
[data-testid="stSelectbox"] * {color: #e6e2de !important;}
[data-testid="stSelectbox"] svg {fill: #cac6bd !important;}
[data-baseweb="popover"] li {background: #211f1d; color: #e6e2de; font-family: 'Inter';}
[data-baseweb="popover"] li:hover {background: #2b2a28; color: #ffb68e;}

[data-testid="stExpander"] {border: 1px solid #554339; border-radius: 4px; background: #211f1d;}
[data-testid="stExpander"] summary {font-family: 'Barlow Condensed'; font-weight: 700; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; color: #cac6bd; background: #211f1d; border-radius: 4px;}
[data-testid="stExpander"] summary:hover {background: #2b2a28; color: #ffb68e;}
[data-testid="stExpander"] details, [data-testid="stExpander"] details > div {
  background: #211f1d; border: 0;}
[data-testid="stExpander"] svg {fill: #cac6bd;}

[data-testid="stWidgetLabel"] p {font-family: 'Inter'; font-size: 14px; color: #cac6bd;}
hr {border-color: #554339; margin: 8px 0;}
</style>
"""


# board.json is the only load, and it happens once per server process
@st.cache_data
def load_board():
    data = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    return data["players"], data["survival"], data["last_pick"]


# P(on the board at k | on the board at j). Two int16 lookups and a division.
#
# Raw S(k) answers "does he last to k from the start of the draft", which at pick
# 30 is a question nobody asked -- the picks before j already happened, and him
# still being here is information. Conditioning on it is the whole model.
def availability(curve, j, k, last_pick):
    if k <= j:
        return 1.0

    # Past the fitted horizon the curves are all 0, so the ratio is 0/0. A league
    # deep enough to draft there gets no number rather than a confident 0%.
    if j > last_pick:
        return None

    s_j = curve[j - 1]
    s_k = curve[min(k, last_pick) - 1]

    # gone by j on the model's reckoning, so the ratio would be noise over noise
    if s_j <= 0:
        return 0.0

    return max(0.0, min(1.0, s_k / s_j))


# Which overall picks belong to a slot. Snake reverses every even round, so the
# gap between your turns alternates long-short instead of staying constant.
def picks_for_slot(slot, teams, rounds, snake):
    out = []
    for rnd in range(rounds):
        offset = slot if (not snake or rnd % 2 == 0) else (teams + 1 - slot)
        out.append(rnd * teams + offset)
    return out


def next_pick_for(slot, current, teams, rounds, snake):
    return next((p for p in picks_for_slot(slot, teams, rounds, snake)
                 if p >= current), None)


# The slot is never asked for -- it is read off the first pick logged as your own.
def infer_slot(pick, teams, snake):
    rnd, index = divmod(pick - 1, teams)
    return index + 1 if (not snake or rnd % 2 == 0) else teams - index


def roster_seats(counts):
    seats = []
    for pos in POSITION_SLOTS:
        seats += [pos] * int(counts.get(pos, 0))
    seats += ["Flex"] * int(counts.get("Flex", 0))
    seats += ["Bench"] * int(counts.get("Bench", 0))
    return seats


def seat_labels(counts):
    seats = roster_seats(counts)
    labels, seen = [], {}
    for seat in seats:
        seen[seat] = seen.get(seat, 0) + 1
        labels.append(f"{seat}{seen[seat]}" if seats.count(seat) > 1 else seat)
    return labels


def seat_kind(label):
    return label.rstrip("0123456789") or label


# Starters fill before the bench, so a positional run lands in bench seats instead
# of blocking the roster: a third centre is bench depth, not a wasted pick.
def assign_seat(position, filled, counts):
    for label in seat_labels(counts):
        if filled.get(label) is not None:
            continue
        kind = seat_kind(label)
        if kind in SLOT_ELIGIBLE and position not in SLOT_ELIGIBLE[kind]:
            continue
        return label
    return None


def setup_screen():
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<p class="caps">Draft setup</p>', unsafe_allow_html=True)

        with st.container(border=True):
            a, b = st.columns(2)
            teams = a.number_input("Teams", 4, 20, 12, 1)
            rounds = b.number_input("Rounds", 5, 20, 13, 1)
            order = st.selectbox("Draft order", ("Snake", "Linear"))

            st.markdown('<p class="caps">Roster slots</p>', unsafe_allow_html=True)
            counts, cols = {}, st.columns(4)
            for i, (slot, default) in enumerate(DEFAULT_ROSTER):
                counts[slot] = cols[i % 4].number_input(
                    slot, 0, 10, default, 1, key=f"slot_{slot}")

            if st.button("Start draft", type="primary"):
                st.session_state.settings = {
                    "teams": int(teams), "rounds": int(rounds),
                    "snake": order == "Snake", "counts": counts}
                st.session_state.roster = {label: None
                                           for label in seat_labels(counts)}
                st.session_state.stage = "board"
                st.rerun()


def header(settings):
    pick = st.session_state.pick
    teams, rounds = settings["teams"], settings["rounds"]
    rnd, index = divmod(pick - 1, teams)
    total = teams * rounds

    left, right = st.columns([1, 2])
    left.markdown(f'<div class="panel"><p class="caps">Pick</p>'
                  f'<p class="pickno">{rnd + 1:02d}.{index + 1:02d}</p></div>',
                  unsafe_allow_html=True)

    slot = st.session_state.my_slot
    if slot is None:
        body = ('<p class="note">Log your first pick with My pick. '
                'Your slot follows from it.</p>')
    else:
        nxt = next_pick_for(slot, pick, teams, rounds, settings["snake"])
        if nxt is None:
            body = '<p class="next">No picks left.</p>'
        else:
            away = nxt - pick
            gap = "on the clock" if away == 0 else f"{away} away"
            body = f'<p class="next">Your next pick: {nxt} and {gap}</p>'

    width = 100.0 * min(pick - 1, total) / max(total, 1)
    right.markdown(f'<div class="panel"><p class="caps">Draft</p>{body}'
                   f'<div class="bar"><div class="fill" '
                   f'style="width:{width:.1f}%"></div></div></div>',
                   unsafe_allow_html=True)


def log_pick(pid, owner, settings):
    st.session_state.taken[pid] = owner
    entry = {"pid": pid, "owner": owner, "pick": st.session_state.pick,
             "slot_set": False, "seat": None}

    if owner == "me":
        if st.session_state.my_slot is None:
            st.session_state.my_slot = infer_slot(
                st.session_state.pick, settings["teams"], settings["snake"])
            entry["slot_set"] = True

        seat = assign_seat(st.session_state.by_id[pid]["position"],
                           st.session_state.roster, settings["counts"])
        if seat:
            st.session_state.roster[seat] = pid
            entry["seat"] = seat

    st.session_state.history.append(entry)
    st.session_state.pick += 1
    st.session_state.selected = None


# Reverts the last entry whoever made it, including the slot if that pick set it.
def undo():
    if not st.session_state.history:
        return

    entry = st.session_state.history.pop()
    st.session_state.taken.pop(entry["pid"], None)
    st.session_state.pick = entry["pick"]

    if entry["seat"]:
        st.session_state.roster[entry["seat"]] = None
    if entry["slot_set"]:
        st.session_state.my_slot = None

    st.session_state.selected = None


# One st.rerun() per logged pick, never more.
def actions(settings, roster_full):
    pid = st.session_state.selected
    left, mid, right = st.columns(3, wrap=False)

    if left.button("My pick", disabled=pid is None or roster_full):
        log_pick(pid, "me", settings)
        st.rerun()

    if mid.button("Opponent pick", disabled=pid is None):
        log_pick(pid, "opp", settings)
        st.rerun()

    if right.button("Undo", disabled=not st.session_state.history):
        undo()
        st.rerun()


# Weight encodes urgency: likely gone is heavier and brighter, likely to last is
# 400-weight. No colour pills -- urgency is what you scan for, so it is type.
def availability_cell(chance):
    if chance is None:
        return '<span class="cell">&mdash;</span>'
    css = "gone" if chance < 0.5 else "lasts"
    return f'<span class="{css}">{round(chance * 100):d}%</span>'


def board_rows(players, survival, last_pick, settings):
    query = st.text_input("Search players", placeholder="Search players",
                          label_visibility="collapsed").strip().lower()

    available = [p for p in players if p["player_id"] not in st.session_state.taken]
    if query:
        available = [p for p in available
                     if query in f"{p['name']} {p['team']}".lower()]

    slot = st.session_state.my_slot
    nxt = (next_pick_for(slot, st.session_state.pick, settings["teams"],
                         settings["rounds"], settings["snake"])
           if slot is not None else None)

    st.markdown('<div class="head"><span class="caps col-name">Player</span>'
                '<span class="caps col-pos">Pos</span>'
                '<span class="caps col-adp">ADP</span>'
                '<span class="caps col-avail">Available</span></div>',
                unsafe_allow_html=True)

    # Two elements per row: a narrow select button, then the data as one flex
    # line. A button label cannot be styled per cell, and st.columns stacks
    # below ~640px -- this keeps real columns on a phone.
    #
    # 12 visible rows means 12 divisions. Nothing here is cached or precomputed.
    for player in available[:ROWS]:
        pid = player["player_id"]
        chance = (availability(survival[str(pid)], st.session_state.pick, nxt,
                               last_pick) if nxt is not None else None)
        chosen = st.session_state.selected == pid

        # the key carries the selected state, so the CSS needs no DOM adjacency
        st.markdown(f'<div class="rowline{" sel" if chosen else ""}">'
                    f'<span class="col-name"><span class="name">'
                    f'{player["name"]}</span>'
                    f'<span class="team"> {player["team"]}</span></span>'
                    f'<span class="cell col-pos">{player["position"]}</span>'
                    f'<span class="cell col-adp">{player["adp_rank"]}</span>'
                    f'<span class="col-avail">{availability_cell(chance)}</span>'
                    f'</div>', unsafe_allow_html=True)

        if st.button("Select", key=f"p{pid}"):
            st.session_state.selected = None if chosen else pid
            st.rerun()

    return available, nxt


# The right-hand panel: the top of the ADP board with the chance each man lasts
# to your next turn. Ordered by ADP, not by that chance -- the disclosure line
# below says why, and re-sorting by the model would contradict it.
# The right-hand panel: the top of the ADP board with the chance each man lasts
# to your next turn. Ordered by ADP, not by that chance -- the disclosure line
# says why, and re-sorting by the model would contradict it.
#
# Built as ONE markdown string: Streamlit sanitises each block on its own, so a
# wrapper div opened in one call would not enclose the next.
def wont_last(available, survival, last_pick, nxt):
    if nxt is None:
        body = '<p class="note">Log your first pick to see this.</p>'
    elif not available:
        body = '<p class="note">No players left on the board.</p>'
    else:
        body = ""
        for player in available[:WONT_LAST]:
            chance = availability(survival[str(player["player_id"])],
                                  st.session_state.pick, nxt, last_pick)
            body += (f'<div class="rowline side">'
                     f'<span class="col-name"><span class="name">'
                     f'{player["name"]}</span></span>'
                     f'<span class="cell col-adp">{player["adp_rank"]}</span>'
                     f'<span class="col-avail">{availability_cell(chance)}</span>'
                     f'</div>')

    st.markdown(
        f'<div class="panel side"><p class="caps">Will last till next pick</p>{body}'
        f'<p class="note foot">Ranked by ADP (Average Draft Position - FantasyPros)</p></div>', unsafe_allow_html=True)


def history_panel():
    with st.expander("Draft history", expanded=False):
        if not st.session_state.history:
            st.markdown('<p class="note">No picks yet.</p>', unsafe_allow_html=True)
            return

        for entry in reversed(st.session_state.history):
            player = st.session_state.by_id[entry["pid"]]
            who = "You" if entry["owner"] == "me" else "Opponent"
            cols = st.columns([1, 5, 2])
            cols[0].markdown(f'<div class="row"><p class="cell">{entry["pick"]}'
                             f'</p></div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div class="row"><p class="name">{player["name"]}'
                             f'<span class="team"> {player["team"]} '
                             f'{player["position"]}</span></p></div>',
                             unsafe_allow_html=True)
            cols[2].markdown(f'<div class="row"><p class="cell">{who}</p></div>',
                             unsafe_allow_html=True)


def sidebar_roster():
    with st.sidebar:
        st.markdown('<p class="caps">Roster</p>', unsafe_allow_html=True)
        seats = st.session_state.roster

        for label, pid in seats.items():
            if pid is None:
                st.markdown(f'<div class="slot open">{label}</div>',
                            unsafe_allow_html=True)
            else:
                player = st.session_state.by_id[pid]
                st.markdown(f'<div class="slot full">{label} &nbsp;{player["name"]}'
                            f'<span class="team"> {player["team"]}</span></div>',
                            unsafe_allow_html=True)

        open_seats = sum(1 for v in seats.values() if v is None)
        st.markdown(f'<p class="note">{open_seats} open of {len(seats)}</p>',
                    unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Draft assistant", page_icon="🏀",
                       layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    state = st.session_state
    state.setdefault("stage", "setup")
    state.setdefault("pick", 1)
    state.setdefault("taken", {})
    state.setdefault("my_slot", None)
    state.setdefault("roster", {})
    state.setdefault("settings", {})
    state.setdefault("selected", None)
    state.setdefault("history", [])

    players, survival, last_pick = load_board()
    state.by_id = {p["player_id"]: p for p in players}

    if state.stage == "setup":
        setup_screen()
        return

    settings = state.settings
    roster_full = all(v is not None for v in state.roster.values())

    sidebar_roster()
    header(settings)
    actions(settings, roster_full)

    board, side = st.columns([2, 1], gap="medium")
    with board:
        available, nxt = board_rows(players, survival, last_pick, settings)
        history_panel()
    with side:
        wont_last(available, survival, last_pick, nxt)


main()
