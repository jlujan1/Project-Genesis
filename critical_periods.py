"""Critical Periods — time-windowed plasticity regimes.

During early life the SNN has high plasticity (rapid learning), which
gradually consolidates into lower plasticity as the agent matures.
Specific subsystems have their own critical windows:

  - Sensory wiring:   ticks 0–500   (high plasticity)
  - Social bonding:   ticks 200–800
  - Language:          ticks 300–1000
  - Self-model:       ticks 100–600

After a critical period closes, the learning rate for that domain is
permanently reduced, mimicking biological development.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CriticalWindow:
    """Defines a critical period for one cognitive domain."""
    name: str
    open_tick: int
    close_tick: int
    peak_multiplier: float = 3.0    # LR multiplier at peak openness
    closed_multiplier: float = 0.3  # LR multiplier after window closes

    def openness(self, tick: int) -> float:
        """Return 0–1 openness of this window at the given tick."""
        if tick < self.open_tick:
            return 0.0
        if tick > self.close_tick:
            return 0.0
        mid = (self.open_tick + self.close_tick) / 2
        half = (self.close_tick - self.open_tick) / 2
        return max(0.0, 1.0 - abs(tick - mid) / max(1, half))

    def multiplier(self, tick: int) -> float:
        """Return the learning-rate multiplier at this tick."""
        o = self.openness(tick)
        if o > 0:
            return 1.0 + (self.peak_multiplier - 1.0) * o
        if tick > self.close_tick:
            return self.closed_multiplier
        return 1.0  # before window opens


class CriticalPeriods:
    """Manages developmental critical periods for the agent.

    Provides learning-rate multipliers for different cognitive domains
    that vary over the agent's lifetime, implementing a biological-like
    development trajectory.
    """

    def __init__(self) -> None:
        self.windows = {
            "sensory":    CriticalWindow("sensory",    0, 500, 3.0, 0.3),
            "social":     CriticalWindow("social",     0, 1000, 3.0, 0.5),
            "language":   CriticalWindow("language",   0, 1200, 2.5, 0.5),
            "self_model": CriticalWindow("self_model", 50, 600, 2.5, 0.4),
            "motor":      CriticalWindow("motor",      0, 400, 3.0, 0.3),
        }

    def get_multiplier(self, domain: str, tick: int) -> float:
        """Get the learning-rate multiplier for a domain at the given tick."""
        if domain in self.windows:
            return self.windows[domain].multiplier(tick)
        return 1.0

    def modulate_snn_learning_rate(self, base_lr: float, tick: int) -> float:
        """Compute the effective SNN learning rate at this tick.

        Uses the 'sensory' and 'motor' windows averaged.
        """
        sensory_m = self.windows["sensory"].multiplier(tick)
        motor_m = self.windows["motor"].multiplier(tick)
        return base_lr * (sensory_m + motor_m) / 2.0

    def is_domain_plastic(self, domain: str, tick: int) -> bool:
        """Whether a cognitive domain is currently in its plastic phase.

        During a critical period, the domain can learn rapidly.
        After closure, learning is still possible but at reduced rate.
        Before opening, the domain has not yet begun development.
        """
        if domain not in self.windows:
            return True  # unknown domains are always plastic
        w = self.windows[domain]
        return w.openness(tick) > 0 or tick <= w.close_tick

    def gate_learning(self, domain: str, base_strength: float,
                      tick: int) -> float:
        """Gate a learning signal by the critical period for that domain.

        Returns the gated learning strength:
        - During critical period: amplified by peak_multiplier
        - After closure: reduced by closed_multiplier
        - Before opening: baseline (1.0)
        """
        if domain not in self.windows:
            return base_strength
        return base_strength * self.windows[domain].multiplier(tick)

    def get_developmental_stage(self, tick: int) -> str:
        """Return a human-readable developmental stage label."""
        open_count = sum(1 for w in self.windows.values() if w.openness(tick) > 0)
        closed_count = sum(1 for w in self.windows.values() if tick > w.close_tick)
        total = len(self.windows)

        if closed_count == total:
            return "mature"
        if open_count >= 3:
            return "early_development"
        if open_count >= 1:
            return "mid_development"
        if closed_count >= 2:
            return "consolidating"
        return "nascent"

    def get_encoding(self, tick: int) -> np.ndarray:
        """Encode critical period state for analytics."""
        enc = np.zeros(8, dtype=np.float32)
        for i, (name, w) in enumerate(self.windows.items()):
            if i < 5:
                enc[i] = w.openness(tick)
        enc[5] = self.modulate_snn_learning_rate(1.0, tick)
        return enc

    def get_summary(self, tick: int) -> dict:
        result: dict = {"tick": tick, "stage": self.get_developmental_stage(tick)}
        for name, w in self.windows.items():
            o = w.openness(tick)
            m = w.multiplier(tick)
            status = "open" if o > 0 else ("closed" if tick > w.close_tick else "pre-open")
            result[name] = {
                "status": status,
                "openness": o,
                "multiplier": m,
            }
        result["effective_lr_multiplier"] = self.modulate_snn_learning_rate(1.0, tick)
        return result
