"""Social Learning — imitation and observational learning between agents.

When one agent observes another performing an action and receiving a
reward (e.g. collecting a crystal), it can strengthen the same
sensory→motor pathways in its own SNN — learning vicariously rather
than through direct trial-and-error.

This mirrors mirror-neuron-like mechanisms in biological organisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObservationRecord:
    """A single observed event from watching another agent."""
    tick: int
    other_id: int
    other_pos: np.ndarray
    other_action: int
    observed_reward: bool  # did the other agent gain energy?
    observed_pain: bool    # did the other agent take damage?


class SocialLearning:
    """Enables agents to learn from observing others.

    Maintains a short buffer of observed (agent, action, outcome)
    tuples.  When a positive outcome is observed, the observer's
    SNN weights are gently nudged to favour the same action in a
    similar sensory context.
    """

    def __init__(self, learning_rate: float = 0.005,
                 observation_buffer_size: int = 20) -> None:
        self.learning_rate = learning_rate
        self.buffer: list[ObservationRecord] = []
        self.buffer_limit = observation_buffer_size
        self.total_imitations: int = 0
        self.total_observations: int = 0

    # ── public API ──────────────────────────────────────────────

    def observe(self, tick: int, other_id: int,
                other_pos: np.ndarray, other_action: int,
                other_gained_energy: bool,
                other_took_damage: bool) -> None:
        """Record an observation of another agent's action and outcome."""
        rec = ObservationRecord(
            tick=tick,
            other_id=other_id,
            other_pos=other_pos.copy(),
            other_action=other_action,
            observed_reward=other_gained_energy,
            observed_pain=other_took_damage,
        )
        self.buffer.append(rec)
        self.total_observations += 1
        if len(self.buffer) > self.buffer_limit:
            self.buffer.pop(0)

    def imitate(self, snn, sensory_input: np.ndarray) -> None:
        """Nudge the SNN toward actions that were observed to be rewarding.

        For each recently observed *rewarding* action, strengthen the
        pathway from the current sensory context to the corresponding
        motor neuron.  For observed *painful* actions, weaken it.
        """
        if not self.buffer:
            return

        motor_start = snn.motor_start
        n_sens = min(len(sensory_input), snn.config.sensory_neurons)

        for rec in self.buffer:
            if rec.observed_reward:
                # Strengthen sensory→motor for the rewarded action
                motor_idx = motor_start + rec.other_action
                if motor_idx < snn.num_neurons:
                    active = sensory_input[:n_sens] > 0.1
                    sens_indices = np.where(active)[0]
                    snn.weights[sens_indices, motor_idx] += self.learning_rate
                    self.total_imitations += 1

            elif rec.observed_pain:
                # Weakly inhibit the action that caused pain
                motor_idx = motor_start + rec.other_action
                if motor_idx < snn.num_neurons:
                    active = sensory_input[:n_sens] > 0.1
                    sens_indices = np.where(active)[0]
                    snn.weights[sens_indices, motor_idx] -= self.learning_rate * 0.5

        # Clip weights
        np.clip(snn.weights, snn.config.min_weight, snn.config.max_weight,
                out=snn.weights)

        # Keep unconsumed observations, discard acted-upon ones
        self.buffer = [
            r for r in self.buffer
            if not r.observed_reward and not r.observed_pain
        ]

    def get_summary(self) -> dict:
        return {
            "total_observations": self.total_observations,
            "total_imitations": self.total_imitations,
            "pending_observations": len(self.buffer),
        }
