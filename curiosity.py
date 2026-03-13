"""Curiosity-Driven Exploration — intrinsic motivation from prediction error.

Implements Schmidhuber's *compression progress* idea: the agent is
rewarded not by absolute prediction error, but by the *decrease* in
error — learning progress itself becomes rewarding.  High curiosity
boosts exploration noise and biases workspace relevance toward novel
stimuli.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class CuriosityEngine:
    """Generates an intrinsic motivation signal from prediction-error dynamics.

    The curiosity drive tracks a running window of prediction errors and
    computes the *learning progress* (negative slope of the error curve).
    When the agent is actively learning (errors dropping), curiosity is
    satisfied.  When facing something truly unpredictable, curiosity
    surges — pushing the agent toward novel regions of the environment.

    Temporal novelty: cells visited long ago become partially novel again,
    preventing curiosity from permanently collapsing.
    """

    CURIOSITY_FLOOR = 0.12          # minimum curiosity (never fully habituated)
    TEMPORAL_NOVELTY_DECAY = 500    # ticks before a visited cell becomes novel again
    CHANGE_SENSITIVITY = 0.4        # weight of environmental change signal

    def __init__(self, window: int = 60, exploration_boost: float = 0.3) -> None:
        self.error_history: deque[float] = deque(maxlen=window)
        self.window = window
        self.exploration_boost = exploration_boost

        # State
        self.curiosity_level: float = 0.2  # start with some curiosity
        self.learning_progress: float = 0.0
        self.visited_cells: dict[tuple[int, int], int] = {}  # cell → last visit tick
        self.novelty_bonus: float = 0.0
        self.total_novel_visits: int = 0
        self._tick: int = 0
        self._env_change_signal: float = 0.0  # set externally by environment events

    def set_environment_change(self, change: float) -> None:
        """Notify curiosity of an environmental change (weather, resource pulse, etc.)."""
        self._env_change_signal = max(self._env_change_signal, change)

    def update(self, prediction_error: float,
               position: tuple[float, float]) -> float:
        """Update curiosity from current prediction error and position.

        Returns the current curiosity drive (0–1+).
        """
        self._tick += 1
        self.error_history.append(prediction_error)

        # Learning progress: negative slope of error over window
        if len(self.error_history) >= 4:
            recent = list(self.error_history)
            half = len(recent) // 2
            old_mean = float(np.mean(recent[:half]))
            new_mean = float(np.mean(recent[half:]))
            self.learning_progress = old_mean - new_mean  # positive = improving
        else:
            self.learning_progress = 0.0

        # Temporal novelty: cells visited long ago become partially novel again
        cell = (int(position[0]), int(position[1]))
        if cell not in self.visited_cells:
            self.visited_cells[cell] = self._tick
            self.novelty_bonus = 1.0
            self.total_novel_visits += 1
        else:
            ticks_since = self._tick - self.visited_cells[cell]
            if ticks_since > self.TEMPORAL_NOVELTY_DECAY:
                # Re-visiting after a long time — partial novelty
                temporal_novelty = min(1.0, ticks_since / (self.TEMPORAL_NOVELTY_DECAY * 3))
                self.novelty_bonus = temporal_novelty * 0.6
                self.total_novel_visits += 1
            else:
                self.novelty_bonus *= 0.92  # slightly slower decay
            self.visited_cells[cell] = self._tick

        # Environmental change boosts curiosity
        env_boost = self._env_change_signal * self.CHANGE_SENSITIVITY
        self._env_change_signal *= 0.85  # decay the change signal

        # Curiosity level: prediction error + novelty + env change - learning progress
        raw = (prediction_error * 0.5
               + self.novelty_bonus * 0.3
               + env_boost * 0.3
               - self.learning_progress * 0.3)  # reduced penalty for learning

        self.curiosity_level = float(np.clip(
            0.8 * self.curiosity_level + 0.2 * max(raw, self.CURIOSITY_FLOOR),
            self.CURIOSITY_FLOOR, 1.0
        ))
        return self.curiosity_level

    def get_exploration_noise(self) -> float:
        """Extra exploration noise to add to action selection."""
        return self.curiosity_level * self.exploration_boost

    def get_workspace_relevance(self) -> float:
        """Relevance boost for the curiosity workspace packet."""
        return 0.1 + self.curiosity_level * 0.4

    def get_encoding(self) -> np.ndarray:
        """Encode curiosity state for a workspace packet."""
        enc = np.zeros(8, dtype=np.float32)
        enc[0] = self.curiosity_level
        enc[1] = self.learning_progress
        enc[2] = self.novelty_bonus
        enc[3] = min(1.0, len(self.visited_cells) / 500.0)
        enc[4] = float(np.mean(self.error_history)) if self.error_history else 0.0
        return enc

    def get_summary(self) -> dict:
        return {
            "curiosity_level": self.curiosity_level,
            "learning_progress": self.learning_progress,
            "novel_cells_visited": len(self.visited_cells),
            "total_novel_visits": self.total_novel_visits,
            "env_change_signal": self._env_change_signal,
        }
