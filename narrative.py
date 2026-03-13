"""Narrative Self — the agent constructs an autobiography from episodic memory.

The agent builds a temporal story of its own existence by stringing
together salient episodic memories into a coherent narrative.  This
provides a sense of continuity over time: 'I was hungry, found food
at (x,y), learned to go there.'

The narrative is:
  - Updated periodically by reviewing recent episodic memory
  - Compressed into a fixed-length 'life story' vector
  - Submitted to the Global Workspace so the agent can 'reflect on
    its own history' when narrative salience is high

Inspired by Dennett's 'Center of Narrative Gravity' and Damasio's
'autobiographical self'.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NarrativeEvent:
    """A compressed event in the agent's life story."""
    tick: int
    category: str       # "survival", "discovery", "social", "achievement", "loss"
    description_vec: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32))
    emotional_valence: float = 0.0
    importance: float = 0.0


# Event categories and their detection logic
EVENT_CATEGORIES = ["survival", "discovery", "social", "achievement", "loss"]


class NarrativeSelf:
    """Constructs and maintains the agent's autobiographical narrative.

    Periodically scans recent episodic memory for significant events and
    compresses them into a running life story.  The narrative provides:
      - A sense of temporal continuity
      - Source material for inner speech
      - A workspace packet when self-reflection is relevant
      - Identity persistence across dream cycles
    """

    def __init__(self, max_events: int = 50,
                 update_interval: int = 20) -> None:
        self.life_story: deque[NarrativeEvent] = deque(maxlen=max_events)
        self.update_interval = update_interval
        self.narrative_vector: np.ndarray = np.zeros(12, dtype=np.float32)

        # Running statistics for the autobiography
        self.total_discoveries: int = 0
        self.total_losses: int = 0
        self.total_social_events: int = 0
        self.total_achievements: int = 0
        self.narrative_coherence: float = 0.0  # how connected the story is
        self.identity_strength: float = 0.0    # sense of continuous self

    def update(self, tick: int, episodic_memory,
               energy_ratio: float, integrity_ratio: float,
               pain: float, pleasure: float,
               nearby_agents: int,
               self_model_accuracy: float,
               dominant_emotion: str) -> None:
        """Scan recent experience and update the narrative."""
        if tick % self.update_interval != 0:
            return

        recent = episodic_memory.recall_recent(n=5)
        if not recent:
            return

        for episode in recent:
            # Detect significant events
            event = self._classify_event(
                episode, energy_ratio, integrity_ratio,
                pain, pleasure, nearby_agents, dominant_emotion
            )
            if event is not None and event.importance > 0.2:
                self.life_story.append(event)
                self._update_statistics(event)

        # Recompute narrative vector
        self._compute_narrative_vector(self_model_accuracy)

    def _classify_event(self, episode, energy_ratio: float,
                        integrity_ratio: float, pain: float,
                        pleasure: float, nearby_agents: int,
                        dominant_emotion: str) -> NarrativeEvent | None:
        """Classify an episodic memory into a narrative event category."""
        valence = episode.outcome_valence
        state = episode.state

        # Survival event: near-death experience
        if pain > 0.5 or energy_ratio < 0.15:
            return NarrativeEvent(
                tick=episode.tick,
                category="survival",
                description_vec=state[:8].copy() if len(state) >= 8
                else np.pad(state, (0, max(0, 8 - len(state)))),
                emotional_valence=-abs(pain),
                importance=0.5 + pain * 0.5,
            )

        # Discovery: found something rewarding
        if pleasure > 0.3:
            self.total_discoveries += 1
            return NarrativeEvent(
                tick=episode.tick,
                category="discovery",
                description_vec=state[:8].copy() if len(state) >= 8
                else np.pad(state, (0, max(0, 8 - len(state)))),
                emotional_valence=pleasure,
                importance=0.4 + pleasure * 0.4,
            )

        # Social event: interaction with other agents
        if nearby_agents > 0 and dominant_emotion in ("loneliness", "curiosity"):
            self.total_social_events += 1
            return NarrativeEvent(
                tick=episode.tick,
                category="social",
                description_vec=state[:8].copy() if len(state) >= 8
                else np.pad(state, (0, max(0, 8 - len(state)))),
                emotional_valence=0.2,
                importance=0.35,
            )

        # Loss: significant energy/integrity drop
        if valence < -0.4:
            self.total_losses += 1
            return NarrativeEvent(
                tick=episode.tick,
                category="loss",
                description_vec=state[:8].copy() if len(state) >= 8
                else np.pad(state, (0, max(0, 8 - len(state)))),
                emotional_valence=valence,
                importance=0.3 + abs(valence) * 0.4,
            )

        # Achievement: consistent positive outcomes (agent is doing well)
        if valence > 0.2 and energy_ratio > 0.6:
            self.total_achievements += 1
            return NarrativeEvent(
                tick=episode.tick,
                category="achievement",
                description_vec=state[:8].copy() if len(state) >= 8
                else np.pad(state, (0, max(0, 8 - len(state)))),
                emotional_valence=0.3,
                importance=0.25 + valence * 0.3,
            )

        return None

    def _update_statistics(self, event: NarrativeEvent) -> None:
        """Update running narrative statistics."""
        if event.category == "achievement":
            self.total_achievements += 1
        elif event.category == "loss":
            self.total_losses += 1

    def _compute_narrative_vector(self, self_model_accuracy: float) -> None:
        """Compress the life story into a fixed-length vector."""
        enc = np.zeros(12, dtype=np.float32)
        if not self.life_story:
            self.narrative_vector = enc
            return

        events = list(self.life_story)

        # Category distribution
        cat_counts = np.zeros(len(EVENT_CATEGORIES), dtype=np.float32)
        for e in events:
            if e.category in EVENT_CATEGORIES:
                cat_counts[EVENT_CATEGORIES.index(e.category)] += 1
        total = len(events)
        enc[:5] = cat_counts / max(1, total)

        # Average emotional valence of life story
        avg_valence = np.mean([e.emotional_valence for e in events])
        enc[5] = float(np.clip(avg_valence, -1.0, 1.0))

        # Narrative coherence: how much do consecutive events relate?
        if len(events) >= 2:
            coherences = []
            for i in range(len(events) - 1):
                a = events[i].description_vec
                b = events[i + 1].description_vec
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a > 1e-6 and norm_b > 1e-6:
                    coherences.append(float(np.dot(a, b) / (norm_a * norm_b)))
            self.narrative_coherence = float(np.mean(coherences)) if coherences else 0.0
        enc[6] = self.narrative_coherence

        # Story length (normalised)
        enc[7] = min(1.0, total / 30.0)

        # Identity strength: combination of narrative length, coherence, and self-model
        self.identity_strength = float(np.clip(
            0.3 * self.narrative_coherence
            + 0.4 * self_model_accuracy
            + 0.3 * min(1.0, total / 20.0),
            0.0, 1.0
        ))
        enc[8] = self.identity_strength

        # Recent trend (are things getting better or worse?)
        if len(events) >= 3:
            recent_valence = np.mean([e.emotional_valence for e in events[-3:]])
            enc[9] = float(np.clip(recent_valence, -1.0, 1.0))

        self.narrative_vector = enc

    def get_encoding(self) -> np.ndarray:
        """Return the narrative vector for workspace submission."""
        return self.narrative_vector.copy()

    def get_workspace_relevance(self) -> float:
        """Narrative becomes relevant when identity strength is high
        or when significant events have occurred recently."""
        base = 0.08
        base += self.identity_strength * 0.2
        if self.life_story:
            recent = list(self.life_story)[-3:]
            max_importance = max(e.importance for e in recent)
            base += max_importance * 0.15
        return min(0.7, base)

    def get_summary(self) -> dict:
        return {
            "story_length": len(self.life_story),
            "identity_strength": self.identity_strength,
            "narrative_coherence": self.narrative_coherence,
            "total_discoveries": self.total_discoveries,
            "total_losses": self.total_losses,
            "total_social_events": self.total_social_events,
        }
