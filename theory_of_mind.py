"""Theory of Mind — modelling other agents' internal states.

Each agent projects its own self-model onto observed agents to predict
their behaviour.  This is the 'mirror mechanism': "what would *I* do
if I were in *their* position?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis.cognition.self_model import SelfModel


@dataclass
class OtherModel:
    """A simplified internal model of another agent."""
    agent_id: int
    position_estimate: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity_estimate: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_action: int = 0
    prediction_errors: list[float] = field(default_factory=list)
    observations: int = 0

    @property
    def average_error(self) -> float:
        if not self.prediction_errors:
            return 1.0
        return float(np.mean(self.prediction_errors[-20:]))


class TheoryOfMind:
    """Allows an agent to reason about other agents' goals and actions.

    Uses the agent's own self-model as the starting template
    (mirror mechanism) and refines over time via observation.
    """

    def __init__(self, own_self_model: SelfModel) -> None:
        self.own_model = own_self_model
        self.other_models: dict[int, OtherModel] = {}

    def observe(self, other_id: int,
                other_pos: np.ndarray,
                other_vel: np.ndarray,
                other_action: int | None = None) -> dict:
        """Observe another agent's visible state and update model.

        Returns a summary containing the predicted action and error.
        """
        if other_id not in self.other_models:
            self.other_models[other_id] = OtherModel(agent_id=other_id)

        model = self.other_models[other_id]

        # Compute prediction error (how well did we predict their movement?)
        pos_err = float(np.linalg.norm(other_pos - model.position_estimate))
        vel_err = float(np.linalg.norm(other_vel - model.velocity_estimate))
        error = (pos_err + vel_err) / 2.0

        model.prediction_errors.append(error)
        if len(model.prediction_errors) > 50:
            model.prediction_errors.pop(0)

        # Update estimates
        lr = 0.4
        model.position_estimate = (1 - lr) * model.position_estimate + lr * other_pos
        model.velocity_estimate = (1 - lr) * model.velocity_estimate + lr * other_vel
        if other_action is not None:
            model.last_action = other_action
        model.observations += 1

        # Mirror-predict: what would *I* do in their situation?
        predicted_action = self._mirror_predict(model)

        return {
            "other_id": other_id,
            "prediction_error": error,
            "predicted_action": predicted_action,
            "model_quality": 1.0 - min(1.0, model.average_error),
        }

    def _mirror_predict(self, other: OtherModel) -> int:
        """Project own behaviour patterns onto the other agent."""
        vel = other.velocity_estimate

        # If the other agent has clear velocity, assume they're pursuing
        # in that direction — just as *I* would.
        speed = float(np.linalg.norm(vel))
        if speed < 0.05:
            return 0  # probably idle

        # Map velocity to directional action (same mapping I use)
        if abs(vel[0]) > abs(vel[1]):
            return 4 if vel[0] > 0 else 3  # RIGHT / LEFT
        else:
            return 2 if vel[1] > 0 else 1  # DOWN / UP

    def predict_others_actions(self) -> dict[int, int]:
        """Predict what each observed agent will do next.

        Returns mapping of agent_id → predicted action.
        """
        predictions: dict[int, int] = {}
        for aid, model in self.other_models.items():
            if model.observations >= 3:  # need minimum observations
                predictions[aid] = self._mirror_predict(model)
        return predictions

    def get_social_motor_bias(self, own_pos: np.ndarray,
                              num_actions: int) -> np.ndarray:
        """Return a motor bias based on predictions of other agents' behaviour.

        - If a high-confidence model predicts an agent is moving toward
          the same resource, bias toward getting there first or away.
        - If a predicted-threatening agent is nearby, bias toward fleeing.
        - If a cooperator is nearby, bias toward approaching/communicating.
        """
        bias = np.zeros(num_actions, dtype=np.float32)
        if not self.other_models:
            return bias

        for aid, model in self.other_models.items():
            quality = 1.0 - min(1.0, model.average_error)
            if quality < 0.2 or model.observations < 5:
                continue  # unreliable model, skip

            # Relative position of the other agent
            rel = model.position_estimate - own_pos
            dist = float(np.linalg.norm(rel))
            if dist < 0.5:
                continue  # too close / overlapping

            predicted_action = self._mirror_predict(model)

            # If the other agent is pursuing in our direction (competing)
            # bias toward intercepting (get there first) or avoiding
            other_heading = model.velocity_estimate
            heading_toward_us = float(np.dot(other_heading, -rel)) / (dist + 1e-6)

            if heading_toward_us > 0.3:
                # They're coming toward us — bias away (flee bias)
                # bias movement away from them
                if dist < 5:
                    flee_scale = quality * 0.15 * (1.0 - dist / 5.0)
                    if rel[0] > 0 and num_actions > 3:
                        bias[3] += flee_scale  # move left (away from right)
                    elif num_actions > 4:
                        bias[4] += flee_scale  # move right (away from left)
                    if rel[1] > 0 and num_actions > 1:
                        bias[1] += flee_scale  # move up (away from below)
                    elif num_actions > 2:
                        bias[2] += flee_scale  # move down (away from above)

            # If they're moving away — safe to approach if we want to socialize
            if heading_toward_us < -0.2 and dist < 8:
                approach_scale = quality * 0.08
                if num_actions > 6:
                    bias[6] += approach_scale  # emit sound to attract attention

        return bias

    def get_threat_level(self) -> float:
        """Overall threat level from observed agents (0–1)."""
        if not self.other_models:
            return 0.0
        max_threat = 0.0
        for model in self.other_models.values():
            quality = 1.0 - min(1.0, model.average_error)
            speed = float(np.linalg.norm(model.velocity_estimate))
            threat = quality * speed * 0.3
            max_threat = max(max_threat, threat)
        return min(1.0, max_threat)

    def get_encoding(self, other_id: int) -> np.ndarray:
        """Encode the model of another agent as a vector for workspace submission."""
        enc = np.zeros(8, dtype=np.float32)
        if other_id not in self.other_models:
            return enc
        m = self.other_models[other_id]
        enc[0:2] = m.position_estimate
        enc[2:4] = m.velocity_estimate
        enc[4] = float(m.last_action) / 8.0
        enc[5] = 1.0 - min(1.0, m.average_error)  # model confidence
        enc[6] = float(m.observations) / 100.0
        enc[7] = self.get_threat_level()
        return enc

    def get_goal_influence(self) -> dict:
        """Return influence on goal priorities based on ToM predictions.

        If other agents are predicted to be competing (heading toward
        the same resources), boost foraging urgency.  If agents seem
        cooperative, boost socialization.
        """
        influence = {
            "forage_boost": 0.0,
            "socialize_boost": 0.0,
            "survive_boost": 0.0,
        }
        if not self.other_models:
            return influence

        for model in self.other_models.values():
            quality = 1.0 - min(1.0, model.average_error)
            if quality < 0.2 or model.observations < 5:
                continue

            speed = float(np.linalg.norm(model.velocity_estimate))
            # Fast-moving agent nearby = competitive threat
            if speed > 0.5:
                influence["forage_boost"] += quality * 0.15
            # Slow/stationary agent nearby = potential cooperator
            if speed < 0.2:
                influence["socialize_boost"] += quality * 0.1

            # High threat = survival priority
            threat = quality * speed * 0.3
            if threat > 0.3:
                influence["survive_boost"] += threat * 0.2

        # Cap values
        for k in influence:
            influence[k] = min(0.5, influence[k])
        return influence
