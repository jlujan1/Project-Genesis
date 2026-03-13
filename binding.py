"""Multi-Modal Binding — fusing vision, audio, and proprioception.

Combines signals from different sensory modalities into unified
*percepts* (bound representations).  When vision, audio, and
proprioception all point to the same spatial location, the binding
strength is high — producing a coherent, integrated experience.

Inspired by Treisman's Feature Integration Theory and the binding
problem in neuroscience.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class MultiModalBinding:
    """Combines multi-modal signals into unified bound percepts.

    Each tick, the module receives encoded vectors from vision, audio,
    and proprioception and computes:
      - Cross-modal coherence (do modalities agree?)
      - A bound representation (weighted fusion)
      - A binding strength metric

    The bound percept can be submitted to the workspace as a higher-
    level, integrated representation.
    """

    def __init__(self, output_size: int = 12) -> None:
        self.output_size = output_size

        # Learnable binding weights (cross-modal attention)
        self.vision_weight: float = 0.5
        self.audio_weight: float = 0.25
        self.proprio_weight: float = 0.25

        # State
        self.binding_strength: float = 0.0
        self.coherence: float = 0.0
        self.bound_percept: np.ndarray = np.zeros(output_size, dtype=np.float32)
        self.binding_history: deque[float] = deque(maxlen=100)

        # Conflict resolution state
        self.conflict_detected: bool = False
        self.conflict_count: int = 0
        self.dominant_modality: str = "v"
        self.pairwise_coherence: dict[str, float] = {}

        # Temporal coherence tracking (do modalities change together?)
        self._prev_v_energy: float = 0.0
        self._prev_a_energy: float = 0.0
        self._prev_p_energy: float = 0.0
        self._temporal_coherence: float = 0.0

    def bind(self, vision_data: np.ndarray,
             audio_data: np.ndarray,
             proprio_data: np.ndarray) -> np.ndarray:
        """Fuse multi-modal inputs into a single bound percept.

        Resolves cross-modal conflicts: when modalities disagree (low
        coherence), the modality with highest energy dominates.
        Returns the bound representation vector.
        """
        # Normalise inputs to same scale
        v = self._norm(vision_data[:self.output_size])
        a = self._norm(audio_data[:min(len(audio_data), self.output_size)])
        p = self._norm(proprio_data[:min(len(proprio_data), self.output_size)])

        # Pad shorter modalities
        a = self._pad(a, self.output_size)
        p = self._pad(p, self.output_size)
        v = self._pad(v, self.output_size)

        # Compute modality energies (needed for temporal coherence and weighting)
        v_energy = float(np.sum(v ** 2))
        a_energy = float(np.sum(a ** 2))
        p_energy = float(np.sum(p ** 2))
        total_energy = v_energy + a_energy + p_energy + 1e-9

        # Cross-modal coherence: average pairwise cosine similarity
        va_sim = self._cosine(v, a)
        vp_sim = self._cosine(v, p)
        ap_sim = self._cosine(a, p)
        spatial_coherence = (va_sim + vp_sim + ap_sim) / 3.0

        # Temporal coherence: do modalities change energy together?
        dv = v_energy - self._prev_v_energy
        da = a_energy - self._prev_a_energy
        dp = p_energy - self._prev_p_energy
        self._prev_v_energy = v_energy
        self._prev_a_energy = a_energy
        self._prev_p_energy = p_energy

        deltas = np.array([dv, da, dp])
        delta_norm = np.linalg.norm(deltas)
        if delta_norm > 1e-6:
            # All changing in same direction → high temporal coherence
            same_sign = float(np.sum(np.sign(deltas) == np.sign(deltas[0]))) / 3.0
            self._temporal_coherence = 0.7 * self._temporal_coherence + 0.3 * same_sign
        # else: keep previous value (no change = no evidence either way)

        # Blend spatial and temporal coherence
        self.coherence = 0.4 * spatial_coherence + 0.6 * self._temporal_coherence

        # Track pairwise coherences for conflict detection
        self.pairwise_coherence = {
            "vision_audio": va_sim,
            "vision_proprio": vp_sim,
            "audio_proprio": ap_sim,
        }

        # Adaptive weight adjustment: use energy-weighted fusion
        w_v = v_energy / total_energy
        w_a = a_energy / total_energy
        w_p = p_energy / total_energy

        # ── Conflict resolution ──
        # When coherence is low, modalities disagree. The dominant
        # modality (highest energy) takes over; conflicting signals
        # are suppressed.
        if self.coherence < 0.3:
            self.conflict_detected = True
            self.conflict_count += 1
            # Amplify the dominant modality, suppress others
            dominant = max((v_energy, 'v'), (a_energy, 'a'), (p_energy, 'p'),
                           key=lambda x: x[0])
            suppress_factor = 0.3
            if dominant[1] == 'v':
                w_v = 0.7
                w_a *= suppress_factor
                w_p *= suppress_factor
            elif dominant[1] == 'a':
                w_a = 0.7
                w_v *= suppress_factor
                w_p *= suppress_factor
            else:
                w_p = 0.7
                w_v *= suppress_factor
                w_a *= suppress_factor
            # Renormalise
            w_total = w_v + w_a + w_p + 1e-9
            w_v /= w_total
            w_a /= w_total
            w_p /= w_total
            self.dominant_modality = dominant[1]
        else:
            self.conflict_detected = False
            energies = {'v': v_energy, 'a': a_energy, 'p': p_energy}
            self.dominant_modality = max(energies, key=energies.get)

        self.bound_percept = (w_v * v + w_a * a + w_p * p).astype(np.float32)

        # Binding strength: coherence × total signal energy
        self.binding_strength = float(np.clip(
            self.coherence * min(1.0, total_energy), 0.0, 1.0
        ))
        self.binding_history.append(self.binding_strength)

        # Update learnable weights (slow adaptation)
        lr = 0.01
        self.vision_weight += lr * (w_v - self.vision_weight)
        self.audio_weight += lr * (w_a - self.audio_weight)
        self.proprio_weight += lr * (w_p - self.proprio_weight)

        return self.bound_percept

    def get_workspace_relevance(self) -> float:
        """High binding strength → high relevance workspace packet."""
        return 0.1 + self.binding_strength * 0.4

    def get_encoding(self) -> np.ndarray:
        """Return the bound percept with binding metadata appended."""
        enc = np.zeros(self.output_size, dtype=np.float32)
        n = min(len(self.bound_percept), self.output_size - 2)
        enc[:n] = self.bound_percept[:n]
        enc[-2] = self.binding_strength
        enc[-1] = self.coherence
        return enc

    def get_summary(self) -> dict:
        avg_binding = (float(np.mean(self.binding_history))
                       if self.binding_history else 0.0)
        return {
            "binding_strength": self.binding_strength,
            "coherence": self.coherence,
            "average_binding": avg_binding,
            "vision_weight": self.vision_weight,
            "audio_weight": self.audio_weight,
            "proprio_weight": self.proprio_weight,
            "conflict_detected": self.conflict_detected,
            "conflict_count": self.conflict_count,
            "dominant_modality": self.dominant_modality,
        }

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        mag = np.linalg.norm(x)
        if mag < 1e-9:
            return x
        return x / mag

    @staticmethod
    def _pad(x: np.ndarray, size: int) -> np.ndarray:
        if len(x) >= size:
            return x[:size]
        return np.pad(x, (0, size - len(x)))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
