"""Cultural Transmission — experienced agents actively teach novices.

Extends social learning with *active teaching*: agents that have
survived long enough and have high self-model accuracy emit targeted
signals near resource locations, effectively creating a cultural
channel for knowledge transfer.

Inspired by gene-culture co-evolution theory and Tomasello's work on
shared intentionality.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class CulturalKnowledge:
    """A piece of cultural knowledge that can be transmitted."""
    category: str          # "food_location", "danger_zone", "technique"
    data: np.ndarray       # encoded information
    confidence: float      # how reliable this knowledge is
    source_agent: int      # who originally discovered it
    generation: int = 0    # how many times this was re-transmitted


class CulturalTransmission:
    """Manages teaching and cultural knowledge propagation between agents.

    An agent qualifies as a *teacher* when it has high self-model
    accuracy and sufficient ticks alive.  Teachers emit structured
    teaching signals near important locations, and learners absorb
    these into their own cognitive maps and SNN weights.
    """

    def __init__(self, teach_threshold_ticks: int = 300,
                 teach_threshold_accuracy: float = 0.5) -> None:
        self.teach_threshold_ticks = teach_threshold_ticks
        self.teach_threshold_accuracy = teach_threshold_accuracy

        # Knowledge base — accumulated cultural knowledge
        self.knowledge: deque[CulturalKnowledge] = deque(maxlen=50)
        self.teachings_given: int = 0
        self.teachings_received: int = 0
        self.cultural_generation: int = 0

    def can_teach(self, ticks_alive: int, self_model_accuracy: float) -> bool:
        """Whether this agent qualifies as a teacher."""
        return (ticks_alive >= self.teach_threshold_ticks
                and self_model_accuracy >= self.teach_threshold_accuracy)

    def generate_teaching(self, agent_id: int,
                          cognitive_map,
                          position: tuple[float, float]) -> CulturalKnowledge | None:
        """Generate a teaching signal from the agent's cognitive map.

        Emits knowledge about the best resource location known to the agent.
        """
        if not cognitive_map.cells:
            return None

        # Find the best crystal location in the cognitive map
        best_key = None
        best_score = -1e9
        for key, cell in cognitive_map.cells.items():
            score = cell.crystal_score - cell.danger_score * 0.5
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None or best_score < 0.1:
            return None

        best_cell = cognitive_map.cells[best_key]
        data = np.zeros(8, dtype=np.float32)
        # Encode the location and its properties
        data[0] = best_key[0] * cognitive_map.cell_size / 80.0  # normalised x
        data[1] = best_key[1] * cognitive_map.cell_size / 60.0  # normalised y
        data[2] = min(1.0, best_cell.crystal_score)
        data[3] = min(1.0, best_cell.danger_score)
        # Direction from teacher to resource
        dx = best_key[0] * cognitive_map.cell_size - position[0]
        dy = best_key[1] * cognitive_map.cell_size - position[1]
        mag = max(0.01, np.sqrt(dx * dx + dy * dy))
        data[4] = dx / mag
        data[5] = dy / mag

        self.teachings_given += 1
        return CulturalKnowledge(
            category="food_location",
            data=data,
            confidence=min(1.0, best_cell.visit_count / 10.0),
            source_agent=agent_id,
            generation=self.cultural_generation,
        )

    def receive_teaching(self, knowledge: CulturalKnowledge,
                         cognitive_map,
                         trust: float = 0.5) -> None:
        """Absorb a piece of cultural knowledge into the agent's cognitive map.

        The agent partially trusts the teaching and updates its
        cognitive map accordingly.
        """
        self.knowledge.append(knowledge)
        self.teachings_received += 1

        # Decode the teaching into a map update
        x = knowledge.data[0] * 80.0
        y = knowledge.data[1] * 60.0
        crystal_score = knowledge.data[2]

        key = cognitive_map._key(x, y)
        if key not in cognitive_map.cells:
            from genesis.cognition.cognitive_map import PlaceCell
            cognitive_map.cells[key] = PlaceCell(x=key[0], y=key[1])

        cell = cognitive_map.cells[key]
        # Blend the teaching with existing knowledge
        blend = trust * knowledge.confidence
        cell.crystal_score = (1 - blend) * cell.crystal_score + blend * crystal_score

    def get_summary(self) -> dict:
        return {
            "teachings_given": self.teachings_given,
            "teachings_received": self.teachings_received,
            "knowledge_base_size": len(self.knowledge),
            "cultural_generation": self.cultural_generation,
            "can_teach": None,  # filled by caller with actual check
        }
