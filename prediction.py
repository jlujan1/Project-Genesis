"""The Prediction Engine — constantly predicting the next moment.

Runs internal simulations ('mental sandbox') to evaluate potential
actions before committing to them. When prediction error is high,
it forces attention in the Global Workspace.
"""

from __future__ import annotations

import numpy as np

from genesis.cognition.self_model import SelfModel


class PredictionEngine:
    """Predicts the next state of the world and the agent.

    Uses a simple learned transition model to:
    1. Predict what will happen next (and generate error when wrong)
    2. Mentally simulate future actions before executing them
    """

    def __init__(self, state_size: int = 16, num_actions: int = 8) -> None:
        self.state_size = state_size
        self.num_actions = num_actions

        # Transition model: for each action, how does state change?
        self.transition_models = [
            np.random.randn(state_size, state_size).astype(np.float32) * 0.05
            for _ in range(num_actions)
        ]
        self.learning_rate = 0.01
        self.prediction_error_history: list[float] = []

    def predict_next_state(self, current_state: np.ndarray,
                           action: int) -> np.ndarray:
        """Predict the next state given current state and action."""
        state = current_state[:self.state_size]
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)))

        if 0 <= action < self.num_actions:
            predicted = self.transition_models[action] @ state
        else:
            predicted = state.copy()
        return predicted

    def learn_transition(self, prev_state: np.ndarray, action: int,
                         actual_next: np.ndarray) -> float:
        """Update the transition model from experience. Returns prediction error."""
        prev = prev_state[:self.state_size]
        actual = actual_next[:self.state_size]
        if len(prev) < self.state_size:
            prev = np.pad(prev, (0, self.state_size - len(prev)))
        if len(actual) < self.state_size:
            actual = np.pad(actual, (0, self.state_size - len(actual)))

        if 0 <= action < self.num_actions:
            predicted = self.transition_models[action] @ prev
            error = actual - predicted
            error_mag = float(np.sqrt(np.mean(error ** 2)))

            # Gradient update
            gradient = np.outer(prev, error)
            self.transition_models[action] += self.learning_rate * gradient.T

            # Regularize
            norm = np.linalg.norm(self.transition_models[action])
            if norm > 10.0:
                self.transition_models[action] *= 10.0 / norm

            self.prediction_error_history.append(error_mag)
            if len(self.prediction_error_history) > 200:
                self.prediction_error_history.pop(0)

            return error_mag
        return 0.0

    def simulate_action(self, current_state: np.ndarray, action: int,
                        self_model: SelfModel, steps: int = 3) -> float:
        """Run an internal simulation (mental sandbox) for a proposed action.

        Returns estimated value of the outcome (positive = good, negative = bad).
        Values both safety (energy/integrity) AND state change (exploration).
        """
        state = current_state[:self.state_size].copy()
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)))

        total_value = 0.0
        prev_sim = state.copy()
        for step in range(steps):
            state = self.predict_next_state(state, action)
            # Evaluate predicted state using self-model
            energy_idx = 4  # where energy is in state vector
            integrity_idx = 5
            if energy_idx < len(state):
                total_value += state[energy_idx] * 0.4  # energy is good
            if integrity_idx < len(state):
                total_value += state[integrity_idx] * 0.2  # integrity is good

            # Value state CHANGE — actions that change the world are interesting
            state_change = float(np.sqrt(np.mean((state - prev_sim) ** 2)))
            total_value += state_change * 0.3  # novelty/exploration value
            prev_sim = state.copy()

        # Penalize inaction — NONE produces no state change and stalls learning
        if action == 0:
            total_value -= 0.1

        # Factor in action success probability
        success_prob = self_model.predict_action_outcome(action)
        total_value *= success_prob

        return total_value

    @property
    def average_error(self) -> float:
        if not self.prediction_error_history:
            return 0.0
        return sum(self.prediction_error_history) / len(self.prediction_error_history)
