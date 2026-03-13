"""Cooperation — agents can share resources, use shared shelters, and cooperate.

Extends social interaction beyond observation:
  - Crystal sharing: agent can donate energy to nearby agent
  - Shared shelter: any agent can use any shelter (not just builder)
  - Cooperative building: building near another agent costs less
  - Alliance signal: agents who cooperate form implicit bonds
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CooperationRecord:
    """Tracks cooperation between two agents."""
    partner_id: int
    total_given: float = 0.0
    total_received: float = 0.0
    cooperative_builds: int = 0
    last_interaction_tick: int = 0

    @property
    def trust(self) -> float:
        """Trust score based on mutual cooperation."""
        total = self.total_given + self.total_received
        if total < 0.1:
            return 0.0
        # Balanced exchange = higher trust
        balance = 1.0 - abs(self.total_given - self.total_received) / total
        volume = min(1.0, total / 50.0)
        return balance * 0.5 + volume * 0.5


class CooperationSystem:
    """Manages agent-to-agent cooperation mechanics."""

    def __init__(self) -> None:
        self.partners: dict[int, CooperationRecord] = {}
        self.total_energy_shared: float = 0.0
        self.total_energy_received: float = 0.0
        self.cooperative_actions: int = 0

    def share_energy(self, other_id: int, amount: float, tick: int) -> None:
        """Record an energy sharing event (giving)."""
        rec = self.partners.setdefault(other_id, CooperationRecord(other_id))
        rec.total_given += amount
        rec.last_interaction_tick = tick
        self.total_energy_shared += amount
        self.cooperative_actions += 1

    def receive_energy(self, other_id: int, amount: float, tick: int) -> None:
        """Record receiving energy from another agent."""
        rec = self.partners.setdefault(other_id, CooperationRecord(other_id))
        rec.total_received += amount
        rec.last_interaction_tick = tick
        self.total_energy_received += amount

    def record_cooperative_build(self, other_id: int, tick: int) -> None:
        """Record building near a partner (cooperative construction)."""
        rec = self.partners.setdefault(other_id, CooperationRecord(other_id))
        rec.cooperative_builds += 1
        rec.last_interaction_tick = tick
        self.cooperative_actions += 1

    def get_trust(self, other_id: int) -> float:
        """Get trust level with a specific agent."""
        rec = self.partners.get(other_id)
        return rec.trust if rec else 0.0

    def get_most_trusted(self) -> int | None:
        """Return the agent_id of the most trusted partner."""
        if not self.partners:
            return None
        best = max(self.partners.values(), key=lambda r: r.trust)
        return best.partner_id if best.trust > 0.1 else None

    def should_share(self, own_energy_ratio: float, other_energy_ratio: float,
                     other_id: int) -> bool:
        """Decide whether to share energy with another agent.

        More likely when:
        - Agent has plenty of energy
        - Other agent is struggling
        - There's existing trust
        """
        if own_energy_ratio < 0.5:
            return False  # can't afford to share
        trust = self.get_trust(other_id)
        need = max(0.0, 0.4 - other_energy_ratio)
        generosity = own_energy_ratio * 0.3 + trust * 0.3 + need * 0.4
        return generosity > 0.35

    def get_cooperation_bias(self, num_actions: int,
                             nearby_agent_ids: list[int]) -> np.ndarray:
        """Return motor bias toward cooperative actions when trusted ally nearby."""
        bias = np.zeros(num_actions, dtype=np.float32)
        if not nearby_agent_ids:
            return bias
        # Check if any nearby agent is a trusted partner
        max_trust = max(
            (self.get_trust(aid) for aid in nearby_agent_ids), default=0.0
        )
        if max_trust > 0.3:
            # Bias toward social/cooperative actions
            if num_actions > 6:
                bias[6] += max_trust * 0.1  # EMIT_SOUND (communicate)
            if num_actions > 9:
                bias[9] += max_trust * 0.05  # BUILD (cooperative building)
        return bias

    def get_build_discount(self, nearby_agent_ids: list[int]) -> float:
        """Return energy cost multiplier for building (lower = cheaper).

        Building near trusted allies is cheaper.
        """
        if not nearby_agent_ids:
            return 1.0
        max_trust = max(
            (self.get_trust(aid) for aid in nearby_agent_ids), default=0.0
        )
        # Up to 40% discount with max trust
        return max(0.6, 1.0 - max_trust * 0.4)

    def get_summary(self) -> dict:
        return {
            "partners": len(self.partners),
            "energy_shared": round(self.total_energy_shared, 1),
            "energy_received": round(self.total_energy_received, 1),
            "cooperative_actions": self.cooperative_actions,
            "most_trusted": self.get_most_trusted(),
        }

    # ── Barter / Trade System ──

    def propose_trade(self, other_id: int, offer_energy: float,
                      own_energy_ratio: float, tick: int) -> float | None:
        """Propose a trade: offer energy to another agent.

        Returns the amount to trade if the proposal is valid, else None.
        Higher trust = willing to trade larger amounts.
        """
        trust = self.get_trust(other_id)
        # Won't trade if we have too little energy or no trust relationship
        if own_energy_ratio < 0.4 or offer_energy <= 0:
            return None
        # Cap trade by trust level — trade more with trusted partners
        max_trade = 5.0 + trust * 15.0
        trade_amount = min(offer_energy, max_trade)
        rec = self.partners.setdefault(other_id, CooperationRecord(other_id))
        rec.total_given += trade_amount
        rec.last_interaction_tick = tick
        self.total_energy_shared += trade_amount
        self.cooperative_actions += 1
        return trade_amount

    def evaluate_request(self, requester_id: int,
                         own_energy_ratio: float) -> float:
        """Evaluate how much energy to give in response to a request signal.

        Returns the amount to give (0 = refuse).
        """
        trust = self.get_trust(requester_id)
        if own_energy_ratio < 0.5:
            return 0.0
        # Generosity scales with trust and surplus
        surplus = max(0.0, own_energy_ratio - 0.5) * 20.0
        willingness = trust * 0.6 + 0.2  # some base willingness
        return min(surplus, 8.0) * willingness
