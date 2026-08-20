# Roster legality. §8.1 enforces this inside the rollout, so this file is the
# constraint itself and there is no separate optimizer.

import numpy as np

# config first so the repo root lands on sys.path before any data.* import
import config
from config import BENCH_SLOTS, POSITIONS, ROSTER_SLOTS, UNKNOWN_POSITION

# positions are small ints so the rollout can index arrays instead of dicts
POSITION_INDEX = {pos: i for i, pos in enumerate(POSITIONS)}
UNKNOWN_INDEX = len(POSITIONS)
N_POSITION_SLOTS = len(POSITIONS) + 1


# Model A emits UNK when a position never resolved. He is bench-only rather than
# dropped -- removing him would shrink the pool opponents draw from.
def position_index(position):
    return POSITION_INDEX.get(position, UNKNOWN_INDEX)


# starting slots as a count per position index, the shape the rollout adds to
def slot_capacity(slots=None, bench=BENCH_SLOTS):
    slots = slots or ROSTER_SLOTS
    capacity = np.zeros(N_POSITION_SLOTS, dtype=np.int32)

    for pos, n in slots.items():
        capacity[position_index(pos)] = n

    return capacity, int(bench)


# how many seats a full roster has, which is also how many rounds it takes
def roster_size(slots=None, bench=BENCH_SLOTS):
    capacity, bench_seats = slot_capacity(slots, bench)
    return int(capacity.sum()) + bench_seats


# Whether one more at this position fits: his slot free OR bench left, which is
# what makes the bench the slack rather than a separate pool.
def can_add(counts, bench_used, position_idx, capacity, bench_seats):
    if counts[position_idx] < capacity[position_idx]:
        return True

    return bench_used < bench_seats


# Starting slots fill first, so a guard taken while a guard slot is open does not
# burn the bench seat that absorbs a later position run.
def add_player(counts, bench_used, position_idx, capacity, bench_seats):
    if counts[position_idx] < capacity[position_idx]:
        counts = counts.copy()
        counts[position_idx] += 1
        return counts, bench_used

    if bench_used < bench_seats:
        return counts, bench_used + 1

    raise ValueError("roster is full at this position and the bench is full")


# Tracked as counts, not players: the rollout only asks whether a player fits.
# backtest.py keeps the ids separately for realized scoring.
class RosterState:
    # A plain list with an incrementally tracked count. can_add runs ~650k times a
    # recommendation, where an ndarray .sum() over four elements dominated the profile.
    def __init__(self, slots=None, bench=BENCH_SLOTS):
        capacity, self.bench_seats = slot_capacity(slots, bench)
        self.capacity = [int(c) for c in capacity]
        self.counts = [0] * N_POSITION_SLOTS
        self.bench_used = 0
        self.size = roster_size(slots, bench)
        self._filled = 0

    def copy(self):
        other = RosterState.__new__(RosterState)
        other.capacity = self.capacity
        other.bench_seats = self.bench_seats
        other.counts = self.counts[:]
        other.bench_used = self.bench_used
        other.size = self.size
        other._filled = self._filled
        return other

    def filled(self):
        return self._filled

    def is_full(self):
        return self._filled >= self.size

    def can_add(self, position_idx):
        if self._filled >= self.size:
            return False
        if self.counts[position_idx] < self.capacity[position_idx]:
            return True
        return self.bench_used < self.bench_seats

    def add(self, position_idx):
        if self.counts[position_idx] < self.capacity[position_idx]:
            self.counts[position_idx] += 1
        elif self.bench_used < self.bench_seats:
            self.bench_used += 1
        else:
            raise ValueError("roster is full at this position and the bench is full")

        self._filled += 1
        return self

    # positions this roster can still take, which is what an opponent samples within
    def open_positions(self):
        return [i for i in range(N_POSITION_SLOTS) if self.can_add(i)]

    # How badly a position is still needed, for §8.1's opponent model: a team with no
    # centers wants one more than a team with two.
    def need(self, position_idx):
        if not self.can_add(position_idx):
            return 0.0

        open_slots = max(0, self.capacity[position_idx] - self.counts[position_idx])
        if open_slots > 0:
            return float(open_slots)

        # only bench left, so he is wanted but not needed
        return 0.25


if __name__ == "__main__":
    print(f"slots {ROSTER_SLOTS}, bench {BENCH_SLOTS}, "
          f"roster size {roster_size()}\n")
