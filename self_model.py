"""The Self-Model — the digital ego.

The agent constructs a compressed representation of itself: its body,
its capabilities, its internal states. When the Global Workspace
broadcasts data about this self-model, the system is theoretically
'aware that it exists.'
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelfModel:
    """The agent's internal representation of itself.

    Built incrementally as the prediction engine discovers that the
    most predictable thing in the environment is the agent's own body.
    """
    # Physical model
    position_estimate: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity_estimate: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32))

    # Internal state estimates
    energy_estimate: float = 1.0
    integrity_estimate: float = 1.0

    # Capability model (learned over time)
    action_success_rates: np.ndarray = field(
        default_factory=lambda: np.ones(14, dtype=np.float32) * 0.5)

    # Body boundary — which cells 'belong to me'
    body_radius_estimate: float = 1.0

    # Predictions about own future state
    predicted_energy: float = 1.0
    predicted_integrity: float = 1.0

    # Tracked energy delta (EMA of actual per-tick change)
    _energy_delta_ema: float = -0.02
    _integrity_delta_ema: float = 0.0

    # Self-model confidence
    model_accuracy: float = 0.0
    update_count: int = 0

    def update(self, actual_pos: np.ndarray, actual_vel: np.ndarray,
               actual_energy: float, actual_integrity: float,
               last_action: int, action_succeeded: bool) -> float:
        """Update the self-model with actual observations.

        Returns prediction error (how wrong the model was).
        """
        # Position/velocity prediction error — normalized to [0, 1] scale
        pos_error = float(np.linalg.norm(actual_pos - self.position_estimate)) / 100.0
        vel_error = float(np.linalg.norm(actual_vel - self.velocity_estimate)) / 3.0

        # Energy/integrity prediction error
        energy_error = abs(actual_energy - self.predicted_energy)
        integrity_error = abs(actual_integrity - self.predicted_integrity)

        total_error = (pos_error + vel_error + energy_error + integrity_error) / 4.0

        # Track actual energy and integrity deltas BEFORE overwriting
        actual_e_delta = actual_energy - self.energy_estimate
        self._energy_delta_ema = 0.8 * self._energy_delta_ema + 0.2 * actual_e_delta
        actual_i_delta = actual_integrity - self.integrity_estimate
        self._integrity_delta_ema = 0.8 * self._integrity_delta_ema + 0.2 * actual_i_delta

        # Update estimates — faster convergence for position
        lr = 0.3
        self.position_estimate = (1 - lr) * self.position_estimate + lr * actual_pos
        self.velocity_estimate = (1 - lr) * self.velocity_estimate + lr * actual_vel
        self.energy_estimate = actual_energy
        self.integrity_estimate = actual_integrity

        # Update action success tracking
        if 0 <= last_action < len(self.action_success_rates):
            current = self.action_success_rates[last_action]
            self.action_success_rates[last_action] = \
                0.95 * current + 0.05 * (1.0 if action_succeeded else 0.0)

        # Track model accuracy (faster EMA for quicker recovery)
        self.model_accuracy = 0.8 * self.model_accuracy + 0.2 * (1.0 - min(1.0, total_error))
        self.update_count += 1

        # Predict next tick's energy and integrity from tracked deltas
        self.predicted_energy = max(0.0, actual_energy + self._energy_delta_ema)
        self.predicted_integrity = max(0.0, min(1.0,
            actual_integrity + self._integrity_delta_ema))

        return total_error

    def predict_action_outcome(self, action: int) -> float:
        """Predict probability of success for a given action (internal simulation)."""
        if 0 <= action < len(self.action_success_rates):
            return float(self.action_success_rates[action])
        return 0.5

    def get_encoding(self) -> np.ndarray:
        """Encode the self-model as a vector for the workspace."""
        encoding = np.zeros(16, dtype=np.float32)
        encoding[0:2] = self.position_estimate
        encoding[2:4] = self.velocity_estimate
        encoding[4] = self.energy_estimate
        encoding[5] = self.integrity_estimate
        encoding[6] = self.model_accuracy
        encoding[7] = self.predicted_energy
        encoding[8] = self.predicted_integrity
        encoding[9] = self.body_radius_estimate
        # Action confidence summary
        encoding[10] = float(np.mean(self.action_success_rates))
        encoding[11] = float(np.std(self.action_success_rates))
        return encoding

    @property
    def has_ego(self) -> bool:
        """Whether the self-model is developed enough to constitute an 'ego'.

        The ego emerges when the model has high accuracy (it successfully
        predicts its own state) and has been updated many times.
        """
        return self.model_accuracy > 0.6 and self.update_count > 200
