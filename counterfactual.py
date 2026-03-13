"""Counterfactual Reasoning — 'what if I had done something else?'

Replays past episodic memories through the prediction engine with
alternative actions, comparing actual outcomes to simulated ones.
When a counterfactual would have produced a better outcome, the
agent strengthens the alternative pathway in its SNN.

This runs during idle moments or dream cycles, enabling offline
learning from regret and relief.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


@dataclass
class CounterfactualResult:
    """Result of a single what-if simulation."""
    original_action: int
    alternative_action: int
    original_value: float
    alternative_value: float
    regret: float              # positive = should have done alternative
    tick_of_episode: int


class CounterfactualEngine:
    """Simulates alternative pasts using the prediction engine.

    Periodically re-examines episodic memories, replaying them with
    different actions.  If the predicted outcome is better, a *regret*
    signal is generated, which can be used to adjust SNN weights.
    """

    def __init__(self, max_history: int = 100,
                 replay_interval: int = 50) -> None:
        self.results: deque[CounterfactualResult] = deque(maxlen=max_history)
        self.replay_interval = replay_interval
        self.total_replays: int = 0
        self.total_regrets: int = 0       # cases where alternative was better
        self.total_reliefs: int = 0       # cases where actual was better
        self.cumulative_regret: float = 0.0
        self.adjustments_made: int = 0    # total SNN weight adjustments

    def maybe_replay(self, tick: int, episodic_memory, prediction_engine,
                     self_model, num_actions: int) -> list[CounterfactualResult]:
        """Replay recent episodes with alternative actions.

        Fires reliably every replay_interval ticks (not conditional).
        Returns list of counterfactual results generated this tick.
        """
        if tick % self.replay_interval != 0:
            return []

        recent = episodic_memory.recall_recent(n=5)
        if not recent:
            return []

        results: list[CounterfactualResult] = []
        for episode in recent:
            original_action = episode.action_taken

            # Simulate the original action too (apples-to-apples comparison)
            original_value = prediction_engine.simulate_action(
                episode.state, original_action, self_model, steps=2
            )

            # Try each alternative action
            best_alt_action = original_action
            best_alt_value = -1e9

            for a in range(num_actions):
                if a == original_action:
                    continue
                simulated_value = prediction_engine.simulate_action(
                    episode.state, a, self_model, steps=2
                )
                if simulated_value > best_alt_value:
                    best_alt_value = simulated_value
                    best_alt_action = a

            regret = best_alt_value - original_value
            result = CounterfactualResult(
                original_action=original_action,
                alternative_action=best_alt_action,
                original_value=original_value,
                alternative_value=best_alt_value,
                regret=regret,
                tick_of_episode=episode.tick,
            )
            results.append(result)
            self.results.append(result)
            self.total_replays += 1

            if regret > 0.1:
                self.total_regrets += 1
                self.cumulative_regret += regret
            elif regret < -0.1:
                self.total_reliefs += 1

        return results

    def apply_learning(self, results: list[CounterfactualResult],
                       snn, sensory_input: np.ndarray,
                       learning_rate: float = 0.003) -> int:
        """Strengthen alternative pathways when regret is high.

        Returns the number of weight adjustments made.
        """
        adjustments = 0
        motor_start = snn.motor_start
        n_sens = min(len(sensory_input), snn.config.sensory_neurons)

        for r in results:
            if r.regret <= 0.05:
                continue  # lower threshold: learn from smaller regrets too

            # Strengthen the better alternative
            alt_motor = motor_start + r.alternative_action
            orig_motor = motor_start + r.original_action
            scale = min(1.0, r.regret) * learning_rate

            if alt_motor < snn.num_neurons:
                active = sensory_input[:n_sens] > 0.1
                sens_indices = np.where(active)[0]
                snn.weights[sens_indices, alt_motor] += scale
                adjustments += 1

            # Weaken the worse original
            if orig_motor < snn.num_neurons:
                active = sensory_input[:n_sens] > 0.1
                sens_indices = np.where(active)[0]
                snn.weights[sens_indices, orig_motor] -= scale * 0.5
                adjustments += 1

        if adjustments > 0:
            np.clip(snn.weights, snn.config.min_weight, snn.config.max_weight,
                    out=snn.weights)
            self.adjustments_made += adjustments
        return adjustments

    def get_summary(self) -> dict:
        avg_regret = (self.cumulative_regret / max(1, self.total_regrets))
        return {
            "total_replays": self.total_replays,
            "total_regrets": self.total_regrets,
            "total_reliefs": self.total_reliefs,
            "average_regret": avg_regret,
            "adjustments_made": self.adjustments_made,
        }

    def get_regret_bias(self, num_actions: int) -> np.ndarray:
        """Return action bias based on accumulated regrets.

        Recent counterfactual replays that found better alternatives
        bias future action selection toward those alternatives and
        away from the regretted actions.
        """
        bias = np.zeros(num_actions, dtype=np.float32)
        if not self.results:
            return bias

        recent = list(self.results)[-10:]
        for r in recent:
            if r.regret > 0.05:
                scale = min(0.15, r.regret * 0.1)
                if 0 <= r.alternative_action < num_actions:
                    bias[r.alternative_action] += scale
                if 0 <= r.original_action < num_actions:
                    bias[r.original_action] -= scale * 0.5
        return bias
