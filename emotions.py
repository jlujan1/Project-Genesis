"""Emotion System — continuous affective states that bias cognition.

Unlike homeostasis (which provides raw pain/pleasure signals), emotions
are *persistent states* that colour perception, memory encoding, and
decision-making over many ticks.  Inspired by the Circumplex Model
(valence × arousal) and Damasio's somatic-marker hypothesis.

Six core emotions form the basis:
  fear, curiosity, contentment, frustration, loneliness, surprise
Each has a value in [0, 1] that decays naturally toward a baseline and
is driven by environmental/internal triggers.

Extended with:
  - Mood: a slow-moving affective background (valence + arousal averaged
    over ~200 ticks) that tints all emotions.
  - Personality: per-agent baseline modifiers making each agent
    temperamentally unique (e.g. bold vs timid).
  - Emotional bonds: attachment/trust scores toward specific other agents
    built through proximity and shared experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


EMOTION_NAMES = ["fear", "curiosity", "contentment",
                 "frustration", "loneliness", "surprise"]
NUM_EMOTIONS = len(EMOTION_NAMES)

# Personality trait names → which emotion baseline they shift
PERSONALITY_TRAITS = ["boldness", "curiosity_trait", "sociability",
                      "patience", "sensitivity"]


@dataclass
class EmotionState:
    """Current affect vector."""
    values: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_EMOTIONS, dtype=np.float32))
    baseline: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.3, 0.2, 0.05, 0.1, 0.1],
                                         dtype=np.float32))
    decay_rate: float = 0.02  # per-tick exponential decay toward baseline

    def __getitem__(self, name: str) -> float:
        idx = EMOTION_NAMES.index(name)
        return float(self.values[idx])

    def __setitem__(self, name: str, val: float) -> None:
        idx = EMOTION_NAMES.index(name)
        self.values[idx] = np.clip(val, 0.0, 1.0)


@dataclass
class Personality:
    """Per-agent personality traits that modify emotional baselines.

    Each trait is in [0, 1] and shifts a specific emotion's baseline:
      boldness       → lowers fear baseline
      curiosity_trait→ raises curiosity baseline
      sociability    → raises loneliness baseline (more social = feels alone faster)
      patience       → lowers frustration baseline
      sensitivity    → raises surprise baseline (more reactive to novelty)
    """
    traits: np.ndarray = field(
        default_factory=lambda: np.clip(
            np.random.normal(0.5, 0.15, len(PERSONALITY_TRAITS)).astype(np.float32),
            0.1, 0.9))

    def apply_to_baseline(self, base: np.ndarray) -> np.ndarray:
        """Return a personality-modified baseline."""
        modified = base.copy()
        t = self.traits
        modified[0] = max(0.01, base[0] - (t[0] - 0.5) * 0.06)  # boldness ↓ fear
        modified[1] = min(0.6, base[1] + (t[1] - 0.5) * 0.08)   # curiosity_trait ↑ curiosity
        modified[4] = min(0.3, base[4] + (t[2] - 0.5) * 0.06)   # sociability ↑ loneliness sensitivity
        modified[3] = max(0.01, base[3] - (t[3] - 0.5) * 0.04)  # patience ↓ frustration
        modified[5] = min(0.3, base[5] + (t[4] - 0.5) * 0.04)   # sensitivity ↑ surprise
        return np.clip(modified, 0.0, 1.0)


@dataclass
class EmotionalBond:
    """Attachment / trust toward a specific other agent."""
    other_id: int
    proximity_ticks: int = 0
    positive_shared: float = 0.0
    negative_shared: float = 0.0

    @property
    def attachment(self) -> float:
        """Attachment strength [0, 1]."""
        proximity_factor = min(1.0, self.proximity_ticks / 300.0)
        experience = self.positive_shared - self.negative_shared * 0.5
        return float(np.clip(proximity_factor * 0.5 + min(0.5, experience / 20.0), 0.0, 1.0))


class EmotionEngine:
    """Computes and maintains the agent's emotional state each tick.

    Inputs that drive emotions
    --------------------------
    * pain / pleasure signals          → fear, contentment
    * prediction error                → surprise, curiosity
    * low energy / integrity          → fear
    * failed actions                  → frustration
    * absence of other agents nearby  → loneliness
    * successful foraging             → contentment

    Outputs
    -------
    * A 6-d vector accessible by other modules (workspace relevance
      boost, memory-salience modulation, exploration/exploitation bias).
    """

    def __init__(self, personality: Personality | None = None) -> None:
        self.personality = personality or Personality()
        self.state = EmotionState()
        # Apply personality to baseline
        self.state.baseline = self.personality.apply_to_baseline(self.state.baseline)
        self.prev_energy: float = 1.0
        self.prev_integrity: float = 1.0
        self.ticks_alone: int = 0

        # Mood — slow-moving affective background
        self.mood_valence: float = 0.0   # −1 .. +1
        self.mood_arousal: float = 0.3   # 0 .. 1
        self._mood_decay: float = 0.005  # very slow

        # Emotional bonds (other_agent_id → EmotionalBond)
        self.bonds: dict[int, EmotionalBond] = {}

    # ── public API ──────────────────────────────────────────────

    def update(self, *, pain: float, pleasure: float,
               prediction_error: float,
               energy_ratio: float, integrity_ratio: float,
               action_succeeded: bool,
               nearby_agents: int) -> np.ndarray:
        """Advance emotions by one tick. Returns the new emotion vector."""

        v = self.state.values

        # ── Triggers ────────────────────────────────────────────

        # Fear: driven by pain, low resources, sudden drops
        energy_drop = max(0.0, self.prev_energy - energy_ratio)
        integrity_drop = max(0.0, self.prev_integrity - integrity_ratio)
        fear_input = (pain * 0.4
                      + max(0.0, 0.3 - energy_ratio) * 1.0
                      + max(0.0, 0.3 - integrity_ratio) * 1.0
                      + energy_drop * 2.0
                      + integrity_drop * 2.0)
        v[0] += fear_input * 0.15  # fear

        # Curiosity: driven by moderate prediction error, novelty
        curiosity_input = min(prediction_error, 0.5) * 1.0
        v[1] += curiosity_input * 0.1  # curiosity

        # Contentment: driven by pleasure, high resources
        content_input = (pleasure * 0.5
                         + max(0.0, energy_ratio - 0.7) * 0.5
                         + max(0.0, integrity_ratio - 0.7) * 0.5)
        v[2] += content_input * 0.1  # contentment

        # Frustration: repeated failed actions
        if not action_succeeded:
            v[3] += 0.08  # frustration
        else:
            v[3] *= 0.9   # relief

        # Loneliness: no nearby agents (capped accumulation)
        if nearby_agents == 0:
            self.ticks_alone += 1
            if self.ticks_alone > 50:
                v[4] += 0.008  # loneliness (reduced from 0.02)
        else:
            self.ticks_alone = 0
            v[4] *= 0.85  # faster decay when social

        # Surprise: large prediction error
        if prediction_error > 0.4:
            v[5] += prediction_error * 0.2  # surprise

        # ── Decay toward baseline ───────────────────────────────
        v += (self.state.baseline - v) * self.state.decay_rate

        # Clamp
        np.clip(v, 0.0, 1.0, out=v)

        # Track for next tick
        self.prev_energy = energy_ratio
        self.prev_integrity = integrity_ratio

        # Update mood (slow-moving average of valence/arousal)
        instant_valence = self.get_valence()
        instant_arousal = self.get_arousal()
        self.mood_valence += (instant_valence - self.mood_valence) * self._mood_decay
        self.mood_arousal += (instant_arousal - self.mood_arousal) * self._mood_decay

        # Mood tints emotions slightly
        if self.mood_valence > 0.2:
            v[2] += 0.005  # contentment boost in good mood
        elif self.mood_valence < -0.2:
            v[0] += 0.003  # fear tint in bad mood
            v[3] += 0.003  # frustration tint

        np.clip(v, 0.0, 1.0, out=v)

        return v.copy()

    def update_bonds(self, nearby_agent_ids: list[int],
                     shared_pleasure: float = 0.0,
                     shared_pain: float = 0.0) -> None:
        """Update emotional bonds based on proximity and shared experience."""
        for aid in nearby_agent_ids:
            bond = self.bonds.setdefault(aid, EmotionalBond(other_id=aid))
            bond.proximity_ticks += 1
            bond.positive_shared += shared_pleasure
            bond.negative_shared += shared_pain
        # Loneliness reduction from bonded agents nearby
        bonded_nearby = sum(
            1 for aid in nearby_agent_ids
            if aid in self.bonds and self.bonds[aid].attachment > 0.3
        )
        if bonded_nearby > 0:
            self.state.values[4] *= max(0.7, 1.0 - bonded_nearby * 0.1)

    def get_strongest_bond(self) -> int | None:
        """Return agent_id of strongest emotional bond, or None."""
        if not self.bonds:
            return None
        best = max(self.bonds.values(), key=lambda b: b.attachment)
        return best.other_id if best.attachment > 0.2 else None

    def get_valence(self) -> float:
        """Overall positive vs negative affect (−1 .. +1)."""
        v = self.state.values
        positive = v[1] + v[2]                # curiosity + contentment
        negative = v[0] + v[3] + v[4]         # fear + frustration + loneliness
        return float(np.clip(positive - negative, -1.0, 1.0))

    def get_arousal(self) -> float:
        """Overall arousal level (0 .. 1)."""
        v = self.state.values
        return float(np.clip(v[0] + v[1] + v[5], 0.0, 1.0))  # fear+curiosity+surprise

    def get_encoding(self) -> np.ndarray:
        """Return emotion vector (length 6) for workspace injection."""
        return self.state.values.copy()

    def get_dominant(self) -> str:
        """Name of the currently strongest emotion."""
        return EMOTION_NAMES[int(np.argmax(self.state.values))]

    def get_summary(self) -> dict:
        """Summary dict for analytics."""
        return {
            "dominant": self.get_dominant(),
            "valence": self.get_valence(),
            "arousal": self.get_arousal(),
            "mood_valence": self.mood_valence,
            "mood_arousal": self.mood_arousal,
            "bonds": len(self.bonds),
            "strongest_bond": self.get_strongest_bond(),
            "personality": {PERSONALITY_TRAITS[i]: float(self.personality.traits[i])
                            for i in range(len(PERSONALITY_TRAITS))},
            **{name: float(self.state.values[i])
               for i, name in enumerate(EMOTION_NAMES)},
        }
