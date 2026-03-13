"""Attention Schema — the agent's model of its own attention.

Based on Michael Graziano's Attention Schema Theory (AST), the agent
maintains a simplified internal model of *what it is currently attending
to* and *why*.  This meta-cognitive representation feeds back into the
Global Workspace competition, enabling the agent to:
  • Predict where its own attention will be drawn next
  • Detect when its attention is being "hijacked" (e.g. by pain)
  • Report (via proto-language) what it is "aware of"

The schema is a compact vector that tracks:
  - current focus source (which module won the workspace)
  - focus duration (how long the same module has held attention)
  - attention inertia (tendency to stay on the same focus)
  - predicted next focus
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# Module indices used for one-hot encoding of attention target
ATTENTION_SOURCES = [
    "vision", "audio", "proprioception", "prediction",
    "self_model", "memory_recall", "homeostasis_alarm",
    "theory_of_mind", "emotion", "goals", "inner_speech",
    "curiosity", "cognitive_map", "binding", "narrative",
    "empowerment", "abstraction",
]
NUM_SOURCES = len(ATTENTION_SOURCES)


@dataclass
class AttentionSchema:
    """The agent's internal model of its own attentional state."""

    # Current state
    current_focus: str = "none"
    focus_duration: int = 0          # consecutive ticks on same focus
    max_focus_seen: int = 0          # longest single focus streak

    # History (for prediction)
    focus_history: list[int] = field(default_factory=list)
    history_limit: int = 100

    # Transition model: P(next_focus | current_focus)
    transition_counts: np.ndarray = field(
        default_factory=lambda: np.full((NUM_SOURCES, NUM_SOURCES),
                                        0.01, dtype=np.float32))

    # Schema accuracy (how well it predicts its own attention)
    predicted_focus: str = "none"
    prediction_hits: int = 0
    prediction_total: int = 0
    recent_predictions: deque = field(
        default_factory=lambda: deque(maxlen=500))

    @property
    def schema_accuracy(self) -> float:
        if len(self.recent_predictions) < 10:
            return 0.0
        return sum(self.recent_predictions) / len(self.recent_predictions)

    def update(self, broadcast_source: str) -> dict:
        """Update the schema after each Global Workspace broadcast.

        Returns a summary dict.
        """
        src_idx = self._source_index(broadcast_source)

        # Check prediction accuracy (top-2 credit: full for exact, partial for runner-up)
        self.prediction_total += 1
        hit = broadcast_source == self.predicted_focus
        if hit:
            self.prediction_hits += 1
            self.recent_predictions.append(1.0)
        else:
            # Partial credit if the actual source was the 2nd-best prediction
            idx_curr = self._source_index(self.current_focus)
            if idx_curr >= 0:
                row = self.transition_counts[idx_curr]
                sorted_idx = np.argsort(row)[::-1]
                if len(sorted_idx) >= 2 and ATTENTION_SOURCES[sorted_idx[1]] == broadcast_source:
                    self.recent_predictions.append(0.4)
                else:
                    self.recent_predictions.append(0.0)
            else:
                self.recent_predictions.append(0.0)

        # Update focus tracking
        if broadcast_source == self.current_focus:
            self.focus_duration += 1
            # Reinforce self-transitions for sustained attention
            if src_idx >= 0:
                self.transition_counts[src_idx, src_idx] += 1.0
        else:
            # Transition happened
            prev_idx = self._source_index(self.current_focus)
            if prev_idx >= 0 and src_idx >= 0:
                self.transition_counts[prev_idx, src_idx] += 2.0
            # Gentle recency decay to preserve learned patterns
            self.transition_counts *= 0.995
            self.transition_counts = np.maximum(self.transition_counts, 0.001)
            self.max_focus_seen = max(self.max_focus_seen, self.focus_duration)
            self.focus_duration = 1
            self.current_focus = broadcast_source

        # Record history
        if src_idx >= 0:
            self.focus_history.append(src_idx)
            if len(self.focus_history) > self.history_limit:
                self.focus_history.pop(0)

        # Predict next focus
        self.predicted_focus = self._predict_next()

        return {
            "focus": self.current_focus,
            "duration": self.focus_duration,
            "predicted_next": self.predicted_focus,
            "schema_accuracy": self.schema_accuracy,
        }

    def get_encoding(self) -> np.ndarray:
        """Encode the attention schema as a vector for workspace injection."""
        enc = np.zeros(NUM_SOURCES + 4, dtype=np.float32)
        # One-hot for current focus
        idx = self._source_index(self.current_focus)
        if idx >= 0:
            enc[idx] = 1.0
        # Duration (normalised)
        enc[NUM_SOURCES] = min(1.0, self.focus_duration / 50.0)
        # Schema accuracy
        enc[NUM_SOURCES + 1] = self.schema_accuracy
        # Attention inertia (how "sticky" is focus)
        enc[NUM_SOURCES + 2] = min(1.0, self.max_focus_seen / 100.0)
        # Predicted next (argmax of transition row)
        pred_idx = self._source_index(self.predicted_focus)
        if pred_idx >= 0:
            enc[NUM_SOURCES + 3] = (pred_idx + 1) / NUM_SOURCES
        return enc

    def get_summary(self) -> dict:
        return {
            "current_focus": self.current_focus,
            "focus_duration": self.focus_duration,
            "max_focus_streak": self.max_focus_seen,
            "schema_accuracy": self.schema_accuracy,
            "predicted_next": self.predicted_focus,
            "prediction_hits": self.prediction_hits,
            "prediction_total": self.prediction_total,
        }

    # ── internals ───────────────────────────────────────────────

    def _source_index(self, name: str) -> int:
        try:
            return ATTENTION_SOURCES.index(name)
        except ValueError:
            return -1

    def _predict_next(self) -> str:
        """Predict which module will win next based on transition history.

        Uses trigram context (last 3 sources) for better prediction,
        blending the current, previous, and two-back source transition rows.
        """
        idx = self._source_index(self.current_focus)
        if idx < 0:
            return "prediction"  # default

        row = self.transition_counts[idx].copy()

        # Blend in previous source's transition row for bigram context
        if len(self.focus_history) >= 2:
            prev_idx = self.focus_history[-2]
            if 0 <= prev_idx < NUM_SOURCES:
                row = 0.6 * row + 0.3 * self.transition_counts[prev_idx]
                # Trigram: also blend two-back for richer sequential context
                if len(self.focus_history) >= 3:
                    prev2_idx = self.focus_history[-3]
                    if 0 <= prev2_idx < NUM_SOURCES:
                        row += 0.1 * self.transition_counts[prev2_idx]

        pred_idx = int(np.argmax(row))
        return ATTENTION_SOURCES[pred_idx]

    def get_action_bias(self, num_actions: int) -> np.ndarray:
        """Return an action bias based on sustained attention patterns.

        Long sustained focus on vision → trust motor output more.
        Sustained focus on proprioception → prioritise survival actions.
        Sustained focus on audio → prioritise social responses.
        """
        bias = np.zeros(num_actions, dtype=np.float32)
        if self.focus_duration < 5:
            return bias  # too brief to matter

        sustain_strength = min(0.15, self.focus_duration * 0.01)

        if self.current_focus == "proprioception" or self.current_focus == "homeostasis_alarm":
            # Survival focus — boost movement to escape, or rest if sustained
            n_move = min(4, num_actions)
            bias[:n_move] += sustain_strength
            if self.focus_duration > 15 and num_actions > 10:
                bias[10] += sustain_strength * 0.5  # REST when stuck in pain
        elif self.current_focus == "audio":
            # Social attention — boost communication
            if num_actions > 6:
                bias[6] += sustain_strength
        elif self.current_focus == "vision":
            # Visual focus — boost collection if crystal detected, or examine
            if num_actions > 5:
                bias[5] += sustain_strength * 0.5
            if num_actions > 8:
                bias[8] += sustain_strength * 0.3  # EXAMINE when visually focused
        elif self.current_focus == "prediction":
            # Prediction focus — boost study (learning) and examine (discovery)
            if num_actions > 11:
                bias[11] += sustain_strength * 0.4  # STUDY
            if num_actions > 8:
                bias[8] += sustain_strength * 0.2  # EXAMINE

        return bias

    def get_workspace_relevance_boost(self) -> dict[str, float]:
        """Return relevance boosts for workspace sources.

        If the schema predicted wrong, boost the unexpected source
        (attention was hijacked — make it more salient).
        If prediction was right than maintain current focus.
        """
        boosts: dict[str, float] = {}
        if self.prediction_total > 0 and self.schema_accuracy < 0.3:
            # Poor prediction → give a boost to actual focus (surprising = important)
            boosts[self.current_focus] = 0.15
        elif self.focus_duration > 20:
            # Very long focus → slightly boost other sources to break fixation
            for src in ATTENTION_SOURCES:
                if src != self.current_focus:
                    boosts[src] = 0.05
        return boosts
