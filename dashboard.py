"""Analytics Dashboard — real-time monitoring of consciousness emergence.

Displays key metrics: Φ score, workspace activity, homeostasis,
self-model development, and inter-agent communication.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass
class AgentMetrics:
    """Collected metrics for a single agent."""
    agent_id: int
    ticks_alive: int = 0
    energy: float = 0.0
    integrity: float = 0.0
    phi: float = 0.0
    complexity: float = 0.0
    reverberation: float = 0.0
    composite_consciousness: float = 0.0
    consciousness_phase: str = "DORMANT"
    workspace_source: str = "none"
    workspace_relevance: float = 0.0
    broadcast_distribution: dict = None
    prediction_error: float = 0.0
    self_model_accuracy: float = 0.0
    has_ego: bool = False
    active_connections: int = 0
    pain: float = 0.0
    pleasure: float = 0.0
    survival_urgency: float = 0.0
    words_spoken: int = 0
    dream_cycles: int = 0
    is_dreaming: bool = False
    dominant_emotion: str = "none"
    emotional_valence: float = 0.0
    attention_focus: str = "none"
    attention_schema_accuracy: float = 0.0
    curiosity_level: float = 0.0
    metacognitive_confidence: float = 0.0
    active_goal: str = "none"
    map_coverage: float = 0.0
    binding_strength: float = 0.0
    counterfactual_regrets: int = 0
    teachings_received: int = 0
    critical_period_lr: float = 1.0
    empowerment: float = 0.0
    agency_score: float = 0.0
    identity_strength: float = 0.0
    narrative_coherence: float = 0.0
    active_concepts: int = 0
    developmental_stage: str = "nascent"
    active_subgoal: str = "none"
    binding_conflicts: int = 0
    grounded_symbols: int = 0
    mood_valence: float = 0.0
    mood_arousal: float = 0.0
    tool_count: int = 0
    cooperation_partners: int = 0
    vocabulary_size: int = 0
    named_entities: int = 0

    def __post_init__(self):
        if self.broadcast_distribution is None:
            self.broadcast_distribution = {}


class Dashboard:
    """Terminal-based analytics dashboard for monitoring the simulation."""

    def __init__(self) -> None:
        self.tick = 0
        self.world_state: dict = {}
        self.agent_metrics: list[AgentMetrics] = []

    def update(self, tick: int, world_state: dict,
               agent_metrics: list[AgentMetrics]) -> None:
        self.tick = tick
        self.world_state = world_state
        self.agent_metrics = agent_metrics

    def render(self) -> str:
        """Render the dashboard as a string for terminal display."""
        lines = []
        w = self.world_state

        # Header
        lines.append("=" * 78)
        lines.append("  PROJECT GENESIS — Digital Consciousness Simulation")
        lines.append("=" * 78)

        # World state
        day_night = "NIGHT" if w.get("is_night", False) else "DAY"
        rain = " RAINING" if w.get("is_raining", False) else ""
        lines.append(
            f"  Tick: {self.tick:,}  |  {day_night}{rain}  |  "
            f"Light: {w.get('light_level', 1.0):.1%}  |  "
            f"Crystals: {w.get('num_crystals', 0)}  |  "
            f"Season: {w.get('season', '?')}  {w.get('weather', '')}  |  "
            f"Predators: {w.get('num_predators', 0)}"
        )
        lines.append("-" * 78)

        # Per-agent metrics
        for m in self.agent_metrics:
            lines.append(f"  AGENT {m.agent_id}")
            lines.append(f"  ├─ Alive: {m.ticks_alive:,} ticks")

            # Homeostasis bar
            e_bar = self._bar(m.energy / 100.0, 20)
            i_bar = self._bar(m.integrity / 100.0, 20)
            lines.append(f"  ├─ Energy:    [{e_bar}] {m.energy:.1f}%")
            lines.append(f"  ├─ Integrity: [{i_bar}] {m.integrity:.1f}%")

            if m.pain > 0.01:
                lines.append(f"  ├─ PAIN: {m.pain:.2f}  |  Urgency: {m.survival_urgency:.2f}")
            if m.pleasure > 0.01:
                lines.append(f"  ├─ PLEASURE: {m.pleasure:.2f}")

            # Consciousness metrics
            phi_bar = self._bar(min(1.0, m.phi * 5), 15)
            lines.append(f"  ├─ Φ (Phi):   [{phi_bar}] {m.phi:.4f}")
            lines.append(f"  ├─ Complexity: {m.complexity:.4f}  |  "
                         f"Reverb: {m.reverberation:.4f}")
            lines.append(f"  ├─ Consciousness: {m.composite_consciousness:.4f} "
                         f"— {m.consciousness_phase}")

            # Workspace
            lines.append(f"  ├─ Workspace: {m.workspace_source} "
                         f"(relevance: {m.workspace_relevance:.2f})")
            if m.broadcast_distribution:
                dist_str = "  ".join(
                    f"{k}: {v:.0%}" for k, v in m.broadcast_distribution.items()
                )
                lines.append(f"  │   Broadcast dist: {dist_str}")

            # Self-model
            ego_str = "YES" if m.has_ego else "no"
            lines.append(f"  ├─ Self-model accuracy: {m.self_model_accuracy:.2%}  |  "
                         f"Ego: {ego_str}")
            lines.append(f"  ├─ Prediction error: {m.prediction_error:.4f}")

            # Cognitive modules
            lines.append(f"  ├─ Goal: {m.active_goal}  |  "
                         f"Subgoal: {m.active_subgoal}")
            lines.append(f"  ├─ Emotion: {m.dominant_emotion}  |  "
                         f"Valence: {m.emotional_valence:+.2f}")
            lines.append(f"  ├─ Curiosity: {m.curiosity_level:.2f}  |  "
                         f"Confidence: {m.metacognitive_confidence:.2f}")

            # Higher cognition (Levels 23-25)
            emp_bar = self._bar(m.empowerment, 10)
            id_bar = self._bar(m.identity_strength, 10)
            lines.append(f"  ├─ Empowerment: [{emp_bar}] {m.empowerment:.2f}  |  "
                         f"Agency: {m.agency_score:.2f}")
            lines.append(f"  ├─ Identity:    [{id_bar}] {m.identity_strength:.2f}  |  "
                         f"Narrative coherence: {m.narrative_coherence:.2f}")
            lines.append(f"  ├─ Concepts: {m.active_concepts} active  |  "
                         f"Symbols grounded: {m.grounded_symbols}")
            lines.append(f"  ├─ Dev stage: {m.developmental_stage}  |  "
                         f"Binding conflicts: {m.binding_conflicts}")

            lines.append(f"  └─ Active synapses: {m.active_connections:,}")

            # New feature metrics
            lines.append(f"  ├─ Mood: V:{m.mood_valence:+.2f}  A:{m.mood_arousal:.2f}")
            lines.append(f"  ├─ Tools: {m.tool_count}  |  "
                         f"Coop partners: {m.cooperation_partners}")
            lines.append(f"  ├─ Vocab: {m.vocabulary_size}  |  "
                         f"Named: {m.named_entities}")
            lines.append("")

        lines.append("=" * 78)
        return "\n".join(lines)

    def print_to_terminal(self) -> None:
        """Clear screen and print dashboard."""
        # Cross-platform clear
        if sys.stdout.isatty():
            os.system("cls" if os.name == "nt" else "clear")
        print(self.render())

    @staticmethod
    def _bar(fraction: float, width: int) -> str:
        filled = int(fraction * width)
        filled = max(0, min(width, filled))
        return "█" * filled + "░" * (width - filled)
