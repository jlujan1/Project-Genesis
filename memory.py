"""Memory systems — working memory, episodic memory, and long-term storage.

Working memory holds the current conscious contents (the Global Workspace broadcast).
Episodic memory records specific timestamped experiences.
Long-term memory stores compressed, high-salience patterns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from genesis.config import MemoryConfig


@dataclass
class Episode:
    """A single episodic memory — a specific event recorded in time."""
    tick: int
    state: np.ndarray           # compressed state snapshot
    action_taken: int           # what the agent did
    outcome_valence: float      # positive (reward) or negative (pain)
    source: str                 # which workspace broadcast triggered storage
    salience: float             # how important this memory is

    def similarity_to(self, query: np.ndarray) -> float:
        """Cosine similarity between this memory and a query vector."""
        a_norm = np.linalg.norm(self.state)
        b_norm = np.linalg.norm(query)
        if a_norm < 1e-9 or b_norm < 1e-9:
            return 0.0
        return float(np.dot(self.state, query) / (a_norm * b_norm))


class WorkingMemory:
    """Short-term buffer holding the most recent conscious broadcasts.

    This is the agent's 'train of thought' — the last several items
    that were broadcast through the Global Workspace.
    """

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self.buffer: deque[dict] = deque(maxlen=capacity)

    def push(self, item: dict) -> None:
        """Add an item to working memory."""
        self.buffer.append(item)

    def get_recent(self, n: int = 5) -> list[dict]:
        """Get the N most recent items."""
        items = list(self.buffer)
        return items[-n:]

    def get_state_vector(self) -> np.ndarray:
        """Flatten recent working memory into a single state vector."""
        if not self.buffer:
            return np.zeros(16, dtype=np.float32)
        vectors = []
        for item in list(self.buffer)[-4:]:
            if "data" in item and isinstance(item["data"], np.ndarray):
                vectors.append(item["data"])
        if not vectors:
            return np.zeros(16, dtype=np.float32)
        combined = np.concatenate(vectors)
        # Compress/pad to fixed size
        target_size = 16
        if len(combined) > target_size:
            combined = combined[:target_size]
        elif len(combined) < target_size:
            combined = np.pad(combined, (0, target_size - len(combined)))
        return combined

    def clear(self) -> None:
        self.buffer.clear()


class EpisodicMemory:
    """Records specific experiences in time — the 'what happened' memory.

    Stores episodes with timestamps and valence, enabling the agent to
    recall past experiences when encountering similar situations.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.episodes: list[Episode] = []
        self.short_term: deque[Episode] = deque(maxlen=config.short_term_buffer_ticks)

    def record(self, tick: int, state: np.ndarray, action: int,
               valence: float, source: str) -> None:
        """Record an experience. High-salience events go to long-term storage."""
        salience = abs(valence) + 0.1  # base salience
        episode = Episode(
            tick=tick, state=state.copy(), action_taken=action,
            outcome_valence=valence, source=source, salience=salience
        )

        # Always goes to short-term
        self.short_term.append(episode)

        # High-salience events are consolidated into long-term memory
        if salience >= self.config.consolidation_threshold:
            self._store_long_term(episode)

    def _store_long_term(self, episode: Episode) -> None:
        """Store in long-term memory, evicting least salient if full."""
        if len(self.episodes) >= self.config.long_term_capacity:
            # Evict least salient
            min_idx = min(range(len(self.episodes)),
                          key=lambda i: self.episodes[i].salience)
            if self.episodes[min_idx].salience < episode.salience:
                self.episodes[min_idx] = episode
        else:
            self.episodes.append(episode)

    def recall(self, query: np.ndarray, top_k: int = 3) -> list[Episode]:
        """Retrieve memories most similar to the current state."""
        if not self.episodes:
            return []

        scored = [(ep, ep.similarity_to(query)) for ep in self.episodes]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for ep, sim in scored[:top_k]:
            if sim >= self.config.retrieval_similarity_threshold:
                results.append(ep)
        return results

    def recall_recent(self, n: int = 5) -> list[Episode]:
        """Get the most recent short-term memories."""
        return list(self.short_term)[-n:]

    @property
    def long_term_count(self) -> int:
        return len(self.episodes)
