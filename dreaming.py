"""Dreaming & Memory Consolidation — offline learning during night cycles.

During the night phase of the day/night cycle, agents enter a dream state:
  1. High-salience episodic memories are replayed through the SNN
  2. Synaptic consolidation strengthens important pathways and prunes weak ones
  3. Working memory is cleared (simulating the transition from wake to sleep)

This mirrors biological sleep functions: memory replay (hippocampal replay),
synaptic homeostasis (Tononi's SHY hypothesis), and REM-like pattern
reactivation that improves next-day performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from genesis.config import MemoryConfig


@dataclass
class DreamConfig:
    """Configuration for the dreaming subsystem."""
    enabled: bool = True
    replay_episodes_per_night: int = 5       # how many memories to replay each night
    consolidation_strength: float = 0.02     # LTP boost for replayed pathways
    prune_threshold: float = 0.005           # weights below this are pruned
    dream_noise: float = 0.15               # noise added during replay (creativity)
    night_phase_start: float = 0.5           # phase value when dreaming begins
    night_phase_end: float = 0.95            # phase value when dreaming ends
    min_episodes_for_dreaming: int = 3       # need this many LTM episodes to dream


@dataclass
class DreamStats:
    """Tracks dreaming activity over the simulation."""
    total_dream_cycles: int = 0
    total_replayed_episodes: int = 0
    total_consolidation_events: int = 0
    connections_pruned: int = 0
    connections_strengthened: int = 0
    last_dream_tick: int = 0
    is_dreaming: bool = False


class DreamEngine:
    """Manages offline memory consolidation during night cycles.

    When the environment's day cycle enters the night phase, the dream
    engine replays high-salience memories through the SNN.  This
    re-activates and strengthens the pathways that encoded important
    experiences, while weak/unused connections are pruned — mirroring
    the synaptic homeostasis hypothesis of biological sleep.
    """

    def __init__(self, config: DreamConfig) -> None:
        self.config = config
        self.stats = DreamStats()
        self._was_night = False  # track night transitions
        self._dreamed_this_night = False

    # ── public API ──────────────────────────────────────────────

    def should_dream(self, day_phase: float, ltm_count: int) -> bool:
        """Return True when conditions for dreaming are met."""
        if not self.config.enabled:
            return False
        if ltm_count < self.config.min_episodes_for_dreaming:
            return False
        is_night = self.config.night_phase_start <= day_phase <= self.config.night_phase_end
        # Dream once per night cycle (at the midpoint of the night)
        if is_night and not self._was_night:
            self._dreamed_this_night = False
        self._was_night = is_night
        if is_night and not self._dreamed_this_night:
            # Trigger dreaming near the start of night
            if day_phase < self.config.night_phase_start + 0.05:
                return True
        return False

    def dream(self, episodic_memory, snn, working_memory, tick: int) -> dict:
        """Run one dream cycle: replay memories and consolidate synapses.

        Parameters
        ----------
        episodic_memory : EpisodicMemory
            Long-term memory store to sample replay episodes from.
        snn : SpikingNeuralNetwork
            The neural network to replay through and consolidate.
        working_memory : WorkingMemory
            Cleared on entering dream state (biological analogy).
        tick : int
            Current simulation tick.

        Returns
        -------
        dict with dream cycle summary.
        """
        self._dreamed_this_night = True
        self.stats.is_dreaming = True
        self.stats.total_dream_cycles += 1
        self.stats.last_dream_tick = tick

        # Clear working memory (you don't hold thoughts while sleeping)
        working_memory.clear()

        # Select high-salience episodes for replay
        replay_episodes = self._select_replay_episodes(episodic_memory)
        replayed = len(replay_episodes)
        self.stats.total_replayed_episodes += replayed

        # Replay each episode through the SNN
        for episode in replay_episodes:
            self._replay_episode(episode, snn)

        # Synaptic consolidation: strengthen strong, prune weak
        pruned, strengthened = self._consolidate_synapses(snn)
        self.stats.connections_pruned += pruned
        self.stats.connections_strengthened += strengthened
        self.stats.total_consolidation_events += 1

        self.stats.is_dreaming = False

        return {
            "dream_cycle": self.stats.total_dream_cycles,
            "replayed": replayed,
            "pruned": pruned,
            "strengthened": strengthened,
        }

    # ── internals ───────────────────────────────────────────────

    def _select_replay_episodes(self, episodic_memory) -> list:
        """Pick episodes to replay, biased toward high salience."""
        episodes = episodic_memory.episodes
        if not episodes:
            return []

        n = min(self.config.replay_episodes_per_night, len(episodes))

        # Weight by salience for sampling
        saliences = np.array([e.salience for e in episodes], dtype=np.float32)
        total = saliences.sum()
        if total < 1e-9:
            probs = np.ones(len(episodes), dtype=np.float32) / len(episodes)
        else:
            probs = saliences / total

        indices = np.random.choice(len(episodes), size=n, replace=False, p=probs)
        return [episodes[i] for i in indices]

    def _replay_episode(self, episode, snn) -> None:
        """Replay a single memory through the SNN with noise (dream distortion)."""
        state = episode.state.copy()

        # Add dream noise (creativity / generalization)
        noise = np.random.randn(*state.shape).astype(np.float32) * self.config.dream_noise
        dream_input = state + noise

        # Inject into sensory neurons (like a hallucination)
        n_sens = min(len(dream_input), snn.config.sensory_neurons)
        sensory_input = np.zeros(snn.config.sensory_neurons, dtype=np.float32)
        sensory_input[:n_sens] = dream_input[:n_sens]
        snn.inject_sensory_input(sensory_input)

        # Run the SNN for a few micro-steps to let the pattern reverberate
        for _ in range(3):
            snn.step()

        # Strengthen pathways that co-fired during replay
        if len(snn.spike_history) >= 2:
            recent = snn.spike_history[-1]
            prev = snn.spike_history[-2]
            co_active = np.outer(prev > 0.5, recent > 0.5)
            np.fill_diagonal(co_active, False)

            # Only strengthen existing connections (no synaptogenesis)
            existing = snn.weights != 0
            co_active = co_active & existing

            # Modulate by episode valence (positive memories strengthened more)
            valence_mod = 1.0 + max(0, episode.outcome_valence) * 0.5
            snn.weights += self.config.consolidation_strength * valence_mod * co_active

            np.clip(snn.weights, snn.config.min_weight, snn.config.max_weight,
                    out=snn.weights)

    def _consolidate_synapses(self, snn) -> tuple[int, int]:
        """Synaptic homeostasis: prune weak connections, strengthen strong ones.

        Returns (pruned_count, strengthened_count).
        """
        w = snn.weights
        abs_w = np.abs(w)

        # Prune: zero out very weak connections
        weak_mask = (abs_w > 0) & (abs_w < self.config.prune_threshold)
        pruned = int(weak_mask.sum())
        w[weak_mask] = 0.0

        # Strengthen: slightly boost above-average connections
        mean_strength = abs_w[abs_w > 0].mean() if (abs_w > 0).any() else 0.0
        strong_mask = abs_w > mean_strength
        strengthened = int(strong_mask.sum())
        w[strong_mask] *= 1.0 + self.config.consolidation_strength * 0.5

        np.clip(w, snn.config.min_weight, snn.config.max_weight, out=w)

        return pruned, strengthened

    def get_summary(self) -> dict:
        """Return a summary dict for analytics / reporting."""
        return {
            "dream_cycles": self.stats.total_dream_cycles,
            "replayed_episodes": self.stats.total_replayed_episodes,
            "consolidation_events": self.stats.total_consolidation_events,
            "connections_pruned": self.stats.connections_pruned,
            "connections_strengthened": self.stats.connections_strengthened,
            "is_dreaming": self.stats.is_dreaming,
        }
