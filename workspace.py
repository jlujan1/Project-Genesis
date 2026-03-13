"""Global Workspace Router — the consciousness bottleneck.

Implements the Global Workspace Theory: specialized modules compete
to broadcast their data to a central 'stage'. The winning packet
is broadcast to ALL modules, creating unified conscious experience.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from genesis.config import WorkspaceConfig
from genesis.neural.modules import WorkspacePacket


@dataclass
class BroadcastRecord:
    """Record of what was broadcast and when."""
    tick: int
    packet: WorkspacePacket
    broadcast_vector: np.ndarray


class GlobalWorkspace:
    """The central hub — consciousness as competitive broadcast.

    Each tick:
    1. Specialized modules submit packets with relevance scores.
    2. The attention gate selects the highest-scoring packet.
    3. The winning packet is broadcast to all modules via the SNN.
    4. This broadcast IS the 'frame' of conscious experience.
    """

    def __init__(self, config: WorkspaceConfig) -> None:
        self.config = config
        self.current_broadcast: WorkspacePacket | None = None
        self.previous_source: str = ""
        self.attention_bias: dict[str, float] = {}  # accumulated focus per module
        self.broadcast_log: list[BroadcastRecord] = []
        self.log_capacity = 500
        self.tick_count = 0

        # Track what modules are competing
        self.module_names: set[str] = set()
        self.broadcast_counts: dict[str, int] = {}

        # Fatigue: penalise modules that win too often recently
        self.recent_winners: deque[str] = deque(maxlen=10)
        self.fatigue_penalty: float = 0.15  # per recent win

    def submit_and_broadcast(self, packets: list[WorkspacePacket],
                              tick: int) -> WorkspacePacket | None:
        """Run the competition — select winner and broadcast.

        Args:
            packets: Submissions from all brain modules this tick.
            tick: Current simulation tick.

        Returns:
            The winning packet that gets broadcast, or None.
        """
        self.tick_count = tick
        if not packets:
            return None

        # Track module names
        for p in packets:
            self.module_names.add(p.source)

        # Apply attention momentum: slightly boost recentl-relevant sources
        scored_packets = []
        for packet in packets:
            adjusted_relevance = packet.relevance

            # Attention momentum — recent focus carries slight inertia
            if packet.source == self.previous_source:
                adjusted_relevance += self.config.attention_momentum * 0.5

            # Accumulated bias (from pain/reward feedback)
            bias = self.attention_bias.get(packet.source, 0.0)
            adjusted_relevance += bias

            # Fatigue — penalise modules that dominated recently
            recent_wins = self.recent_winners.count(packet.source)
            if recent_wins > 0:
                adjusted_relevance -= self.fatigue_penalty * min(recent_wins, 4)

            scored_packets.append((packet, adjusted_relevance))

        # Sort by adjusted relevance, pick the winner
        scored_packets.sort(key=lambda x: x[1], reverse=True)
        winner = scored_packets[0][0]

        # Decay attention biases
        for key in self.attention_bias:
            self.attention_bias[key] *= (1.0 - self.config.relevance_decay)

        # Update state
        self.current_broadcast = winner
        self.previous_source = winner.source
        self.recent_winners.append(winner.source)
        self.broadcast_counts[winner.source] = \
            self.broadcast_counts.get(winner.source, 0) + 1

        # Create broadcast vector (padded/truncated to fixed size)
        broadcast_size = 24
        bv = np.zeros(broadcast_size, dtype=np.float32)
        n = min(len(winner.data), broadcast_size)
        bv[:n] = winner.data[:n]
        broadcast_vector = bv * self.config.broadcast_strength

        # Log the broadcast
        record = BroadcastRecord(tick=tick, packet=winner,
                                  broadcast_vector=broadcast_vector)
        self.broadcast_log.append(record)
        if len(self.broadcast_log) > self.log_capacity:
            self.broadcast_log.pop(0)

        return winner

    def boost_attention(self, source: str, amount: float) -> None:
        """Externally boost a module's attention priority (e.g., from pain)."""
        self.attention_bias[source] = \
            self.attention_bias.get(source, 0.0) + amount

    def get_broadcast_vector(self) -> np.ndarray:
        """Get the current broadcast data for injection into the SNN."""
        if not self.broadcast_log:
            return np.zeros(24, dtype=np.float32)
        return self.broadcast_log[-1].broadcast_vector

    def get_broadcast_summary(self) -> dict:
        """Get analytics about workspace activity."""
        total = sum(self.broadcast_counts.values()) or 1
        distribution = {k: v / total for k, v in self.broadcast_counts.items()}
        return {
            "current_source": self.current_broadcast.source if self.current_broadcast else "none",
            "current_relevance": self.current_broadcast.relevance if self.current_broadcast else 0,
            "broadcast_distribution": distribution,
            "total_broadcasts": total,
        }

    def get_recent_broadcasts(self, n: int = 10) -> list[str]:
        """Get the source names of recent broadcasts."""
        return [r.packet.source for r in self.broadcast_log[-n:]]
