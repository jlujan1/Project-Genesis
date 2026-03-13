"""Empowerment — intrinsic motivation from maximising future options.

The agent is driven to reach states where it has the *most possible
future actions* available.  This is information-theoretic empowerment:
the mutual information between actions and resulting states.

Unlike curiosity (which rewards prediction-error reduction), empowerment
rewards *control* — the agent prefers states where its actions have
clear, distinct consequences.

Inspired by Klyubin, Polani & Nehaniv's empowerment formalism and
Karl Friston's active inference framework.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class EmpowermentEngine:
    """Computes an empowerment signal — how much control the agent has.

    Each tick, the engine estimates how distinguishable the outcomes of
    different actions are from the current state.  High empowerment
    means the agent can strongly influence its future; low empowerment
    means it is 'stuck' or in a state where actions don't matter.

    Uses a lightweight approximation: variance of predicted outcomes
    across available actions.
    """

    def __init__(self, num_actions: int = 8, state_size: int = 16,
                 history_len: int = 100) -> None:
        self.num_actions = num_actions
        self.state_size = state_size
        self.empowerment: float = 0.0
        self.empowerment_history: deque[float] = deque(maxlen=history_len)
        self.agency_score: float = 0.0  # longer-term sense of agency
        self.peak_empowerment: float = 0.0

    def compute(self, prediction_engine, current_state: np.ndarray,
                self_model) -> float:
        """Compute empowerment for the current state.

        Uses the prediction engine to simulate each action and measures
        how distinct the resulting states are (high variance = high
        empowerment = agent has real choices).
        """
        predicted_states = []
        for a in range(self.num_actions):
            pred = prediction_engine.simulate_action(
                current_state, a, self_model, steps=1
            )
            predicted_states.append(pred)

        # Empowerment ≈ variance of predicted outcome values
        # (high variance = actions lead to distinct outcomes = control)
        values = np.array(predicted_states, dtype=np.float32)
        if len(values) > 1:
            outcome_variance = float(np.var(values))
            # Normalize to 0–1 range
            self.empowerment = float(np.clip(outcome_variance * 5.0, 0.0, 1.0))
        else:
            self.empowerment = 0.0

        self.empowerment_history.append(self.empowerment)
        self.peak_empowerment = max(self.peak_empowerment, self.empowerment)

        # Agency score: smoothed long-term sense of control
        self.agency_score = 0.95 * self.agency_score + 0.05 * self.empowerment

        return self.empowerment

    def get_exploration_bias(self, num_actions: int) -> np.ndarray:
        """When empowerment is low, bias toward actions that increase it.

        Low empowerment → explore (movement) to reach more empowering states.
        High empowerment → no bias (agent already has good control).
        """
        bias = np.zeros(num_actions, dtype=np.float32)
        if self.empowerment > 0.4:
            return bias  # already empowered, no override

        # Low empowerment: bias toward movement to change state
        boost = (0.4 - self.empowerment) * 0.3
        n_move = min(4, num_actions)
        bias[:n_move] = boost
        return bias

    def get_workspace_relevance(self) -> float:
        """Empowerment is most relevant when it's very low (agent feels stuck)
        or very high (agent feels maximally in control)."""
        if self.empowerment < 0.15:
            return 0.35  # agent is stuck — needs attention
        if self.empowerment > 0.7:
            return 0.25  # agent is thriving — note it
        return 0.1

    def get_encoding(self) -> np.ndarray:
        """Encode empowerment state for workspace packet."""
        enc = np.zeros(8, dtype=np.float32)
        enc[0] = self.empowerment
        enc[1] = self.agency_score
        enc[2] = self.peak_empowerment
        if self.empowerment_history:
            enc[3] = float(np.mean(self.empowerment_history))
            # Trend: is empowerment increasing or decreasing?
            if len(self.empowerment_history) >= 10:
                recent = list(self.empowerment_history)
                half = len(recent) // 2
                old_mean = float(np.mean(recent[:half]))
                new_mean = float(np.mean(recent[half:]))
                enc[4] = float(np.clip(new_mean - old_mean, -1.0, 1.0))
        return enc

    def get_summary(self) -> dict:
        avg = (float(np.mean(self.empowerment_history))
               if self.empowerment_history else 0.0)
        return {
            "empowerment": self.empowerment,
            "agency_score": self.agency_score,
            "peak_empowerment": self.peak_empowerment,
            "average_empowerment": avg,
        }
