"""Spatial Memory & Cognitive Map — hippocampus-like place cells.

Builds a navigational map of the environment with place-cell-like
nodes recording what was found at each location.  The agent can
then recall crystal locations, danger zones, and plan routes to
remembered resource-rich areas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlaceCell:
    """A single place cell — a remembered location in the environment."""
    x: int
    y: int
    crystal_score: float = 0.0     # how often crystals were found here
    danger_score: float = 0.0      # pain / collision events here
    visit_count: int = 0
    last_visit_tick: int = 0
    agent_encounters: int = 0      # other agents seen at this location


class CognitiveMap:
    """Grid-based spatial memory that builds over time.

    Stores associations at discretised grid positions:
      - crystal abundance  (food memory)
      - danger level       (avoidance memory)
      - recency of visit   (temporal flag)

    Provides goal-directed navigation signals by pointing toward the
    most promising unvisited or resource-rich locations.
    """

    def __init__(self, width: int = 320, height: int = 240,
                 cell_size: int = 4) -> None:
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.cells: dict[tuple[int, int], PlaceCell] = {}
        self.total_updates: int = 0
        # Cached navigation result (invalidated when cells change)
        self._nav_cache_key: tuple | None = None
        self._nav_cache_result: np.ndarray = np.zeros(2, dtype=np.float32)
        self._nav_dirty: bool = True

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(x) // self.cell_size, int(y) // self.cell_size)

    def update(self, position: tuple[float, float], tick: int, *,
               found_crystal: bool = False,
               pain: float = 0.0,
               saw_agent: bool = False) -> None:
        """Record what happened at the current position."""
        key = self._key(position[0], position[1])
        if key not in self.cells:
            self.cells[key] = PlaceCell(x=key[0], y=key[1])
        cell = self.cells[key]
        cell.visit_count += 1
        cell.last_visit_tick = tick
        self.total_updates += 1
        self._nav_dirty = True

        decay = 0.98
        if found_crystal:
            cell.crystal_score = cell.crystal_score * decay + 1.0
        else:
            cell.crystal_score *= decay
        if pain > 0.01:
            cell.danger_score = cell.danger_score * decay + pain
        else:
            cell.danger_score *= decay
        if saw_agent:
            cell.agent_encounters += 1

    def get_navigation_signal(self, position: tuple[float, float],
                              tick: int) -> np.ndarray:
        """Return a direction vector pointing toward the best goal.

        Best goal = highest (crystal_score - danger_score) weighted by
        staleness (prefer cells not visited recently).

        Returns a 2-d direction vector (unit or zero).
        Uses cache to avoid O(cells) scan every tick.
        """
        key = self._key(position[0], position[1])
        cache_key = (key, tick // 5)  # recalculate every 5 ticks
        if not self._nav_dirty and self._nav_cache_key == cache_key:
            return self._nav_cache_result

        best_score = -1e9
        best_dir = np.zeros(2, dtype=np.float32)

        for ckey, cell in self.cells.items():
            if ckey == key:
                continue
            staleness = min(1.0, (tick - cell.last_visit_tick) / 500.0)
            score = (cell.crystal_score * 1.5
                     - cell.danger_score * 2.0
                     + staleness * 0.5)
            if score > best_score:
                best_score = score
                dx = (ckey[0] - key[0]) * self.cell_size
                dy = (ckey[1] - key[1]) * self.cell_size
                mag = max(0.01, np.sqrt(dx * dx + dy * dy))
                best_dir = np.array([dx / mag, dy / mag], dtype=np.float32)

        self._nav_cache_key = cache_key
        self._nav_cache_result = best_dir
        self._nav_dirty = False
        return best_dir

    def get_encoding(self, position: tuple[float, float],
                     tick: int) -> np.ndarray:
        """Encode the cognitive map state for a workspace packet."""
        enc = np.zeros(12, dtype=np.float32)
        key = self._key(position[0], position[1])

        # Current cell info
        if key in self.cells:
            cell = self.cells[key]
            enc[0] = min(1.0, cell.crystal_score)
            enc[1] = min(1.0, cell.danger_score)
            enc[2] = min(1.0, cell.visit_count / 50.0)
        else:
            enc[2] = 0.0  # never visited

        # Navigation signal
        nav = self.get_navigation_signal(position, tick)
        enc[3] = nav[0]
        enc[4] = nav[1]

        # Map coverage
        enc[5] = min(1.0, len(self.cells) / (self.cols * self.rows))

        # Nearby cell summary (3×3 neighbourhood)
        crystal_near = 0.0
        danger_near = 0.0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nk = (key[0] + dx, key[1] + dy)
                if nk in self.cells:
                    crystal_near += self.cells[nk].crystal_score
                    danger_near += self.cells[nk].danger_score
        enc[6] = min(1.0, crystal_near / 9.0)
        enc[7] = min(1.0, danger_near / 9.0)

        return enc

    def get_summary(self) -> dict:
        explored = len(self.cells)
        total = self.cols * self.rows
        best_crystal = max((c.crystal_score for c in self.cells.values()),
                           default=0.0)
        worst_danger = max((c.danger_score for c in self.cells.values()),
                           default=0.0)
        return {
            "explored_cells": explored,
            "total_cells": total,
            "coverage": explored / max(1, total),
            "best_crystal_score": best_crystal,
            "worst_danger_score": worst_danger,
            "total_updates": self.total_updates,
        }
