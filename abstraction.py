"""Symbolic Abstraction — forming concepts from raw experience.

The agent learns to group raw sensory states into *categories*
(abstract concepts) based on their outcomes and co-occurrence
patterns.  For example:
  - 'things that hurt' (high-pain states cluster together)
  - 'places with food' (crystal-associated states)
  - 'safe zones'       (low-pain, high-energy states)

These symbols emerge unsupervised via online clustering of the
agent's state-valence history.  Once formed, concepts can be
reasoned over symbolically and submitted to the workspace.

Inspired by Harnad's Symbol Grounding Problem, Barsalou's
perceptual symbol systems, and Lakoff & Johnson's embodied cognition.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Concept:
    """A learned abstract category grounded in sensory experience."""
    name: str
    centroid: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32))
    exemplar_count: int = 0
    average_valence: float = 0.0
    confidence: float = 0.0          # how well-defined this concept is
    last_activated_tick: int = 0


# Predefined concept slots that emerge from experience
CONCEPT_SLOTS = [
    "danger",       # high-pain states
    "food_source",  # crystal-associated states
    "safe_zone",    # low-pain, high-energy states
    "social",       # near-agent states
    "novelty",      # high-prediction-error states
    "home",         # frequently visited, positive-valence states
    "obstacle",     # collision-associated states
    "opportunity",  # high-empowerment states
]
NUM_CONCEPTS = len(CONCEPT_SLOTS)


class SymbolicAbstraction:
    """Learns and maintains abstract concepts from raw experience.

    Uses lightweight online k-means-like clustering: each concept has a
    centroid in state space.  When a new experience arrives, it is
    matched to the best-fitting concept and used to update the centroid.
    New concepts form when experience doesn't match any existing cluster.

    Concepts can then be 'activated' — checked against the current
    state to see which abstract categories apply right now.
    """

    def __init__(self, state_size: int = 8,
                 learning_rate: float = 0.05) -> None:
        self.state_size = state_size
        self.learning_rate = learning_rate

        # Concept library
        self.concepts: list[Concept] = [
            Concept(name=name) for name in CONCEPT_SLOTS
        ]

        # Active concepts (which concepts match the current state)
        self.active_concepts: list[str] = []
        self.activation_scores: np.ndarray = np.zeros(
            NUM_CONCEPTS, dtype=np.float32)

        # History for concept quality tracking
        self.total_categorisations: int = 0
        self.concept_formation_tick: int = 0

    def learn(self, state: np.ndarray, valence: float, pain: float,
              pleasure: float, prediction_error: float,
              nearby_agents: int, empowerment: float,
              tick: int) -> None:
        """Learn from a new experience — update or form concepts.

        Maps the experience to the appropriate concept slot based on
        the dominant signal (pain → danger, pleasure → food_source, etc.)
        and updates that concept's centroid.
        """
        s = state[:self.state_size].copy()
        if len(s) < self.state_size:
            s = np.pad(s, (0, self.state_size - len(s)))

        # Determine which concept this experience belongs to
        # by checking which signal is strongest
        assignments: list[tuple[int, float]] = []

        if pain > 0.3:
            assignments.append((0, pain))         # danger
        if pleasure > 0.2:
            assignments.append((1, pleasure))     # food_source
        if pain < 0.05 and valence > 0.1:
            assignments.append((2, valence))      # safe_zone
        if nearby_agents > 0:
            assignments.append((3, 0.3))          # social
        if prediction_error > 0.4:
            assignments.append((4, prediction_error))  # novelty
        if valence > 0.0 and pain < 0.1:
            assignments.append((5, 0.2))          # home (freq. positive)
        if pain > 0.1 and valence < -0.2:
            assignments.append((6, abs(valence))) # obstacle
        if empowerment > 0.5:
            assignments.append((7, empowerment))  # opportunity

        if not assignments:
            return

        self.total_categorisations += 1

        for idx, strength in assignments:
            concept = self.concepts[idx]
            lr = self.learning_rate * strength

            # Online centroid update
            if concept.exemplar_count == 0:
                concept.centroid = s.copy()
                concept.concept_formation_tick = tick if hasattr(concept, 'concept_formation_tick') else 0
            else:
                concept.centroid = (1 - lr) * concept.centroid + lr * s

            concept.exemplar_count += 1
            concept.average_valence = (
                0.95 * concept.average_valence + 0.05 * valence
            )
            concept.last_activated_tick = tick

            # Confidence based on number of exemplars
            concept.confidence = min(1.0, concept.exemplar_count / 20.0)

    def activate(self, current_state: np.ndarray, tick: int) -> list[str]:
        """Check which concepts are active for the current state.

        Returns the names of all concepts whose centroids are close
        enough to the current state.
        """
        s = current_state[:self.state_size].copy()
        if len(s) < self.state_size:
            s = np.pad(s, (0, self.state_size - len(s)))

        self.active_concepts = []
        self.activation_scores = np.zeros(NUM_CONCEPTS, dtype=np.float32)

        for i, concept in enumerate(self.concepts):
            if concept.exemplar_count < 3:
                continue  # concept not yet formed

            # Cosine similarity between state and concept centroid
            norm_s = np.linalg.norm(s)
            norm_c = np.linalg.norm(concept.centroid)
            if norm_s < 1e-6 or norm_c < 1e-6:
                continue

            similarity = float(np.dot(s, concept.centroid) / (norm_s * norm_c))
            activation = similarity * concept.confidence

            self.activation_scores[i] = max(0.0, activation)

            if activation > 0.3:
                self.active_concepts.append(concept.name)
                concept.last_activated_tick = tick

        return self.active_concepts

    def get_concept_valence(self, concept_name: str) -> float:
        """Get the learned valence of a concept (is it good or bad?)."""
        for c in self.concepts:
            if c.name == concept_name:
                return c.average_valence
        return 0.0

    def reason(self, current_state: np.ndarray) -> dict:
        """Simple symbolic reasoning over active concepts.

        Returns advice based on which concepts are currently active:
        - If 'danger' is active → urgency signal
        - If 'food_source' is active → approach signal
        - If 'safe_zone' is active → consolidation/rest signal
        """
        advice: dict = {
            "urgency": 0.0,
            "approach": 0.0,
            "explore": 0.0,
            "socialize": 0.0,
        }

        for name in self.active_concepts:
            if name == "danger":
                advice["urgency"] += 0.5
            elif name == "food_source":
                advice["approach"] += 0.4
            elif name == "safe_zone":
                advice["explore"] += 0.1  # safe → free to explore
            elif name == "social":
                advice["socialize"] += 0.3
            elif name == "novelty":
                advice["explore"] += 0.3
            elif name == "opportunity":
                advice["approach"] += 0.2
            elif name == "obstacle":
                advice["urgency"] += 0.2

        return advice

    def get_encoding(self) -> np.ndarray:
        """Encode concept state for workspace packet."""
        enc = np.zeros(12, dtype=np.float32)
        # Activation scores for each concept
        n = min(NUM_CONCEPTS, 8)
        enc[:n] = self.activation_scores[:n]
        # Number of formed concepts
        formed = sum(1 for c in self.concepts if c.exemplar_count >= 3)
        enc[8] = formed / NUM_CONCEPTS
        # Average confidence across formed concepts
        confidences = [c.confidence for c in self.concepts if c.exemplar_count >= 3]
        enc[9] = float(np.mean(confidences)) if confidences else 0.0
        # Number of active concepts
        enc[10] = len(self.active_concepts) / NUM_CONCEPTS
        return enc

    def get_workspace_relevance(self) -> float:
        """Symbolic layer is relevant when multiple concepts are active
        (complex situation requiring abstract reasoning)."""
        n_active = len(self.active_concepts)
        if n_active >= 3:
            return 0.4
        if n_active >= 1:
            return 0.2
        return 0.05

    def get_summary(self) -> dict:
        formed = []
        for c in self.concepts:
            if c.exemplar_count >= 3:
                formed.append({
                    "name": c.name,
                    "exemplars": c.exemplar_count,
                    "valence": round(c.average_valence, 3),
                    "confidence": round(c.confidence, 3),
                })
        return {
            "formed_concepts": len(formed),
            "total_concepts": NUM_CONCEPTS,
            "active_concepts": self.active_concepts,
            "total_categorisations": self.total_categorisations,
            "concepts": formed,
        }
