"""Evolution — natural selection and genetic variation across generations.

When an agent dies, it can be replaced by a new agent whose SNN weights
are *inherited* (with mutation) from the most successful surviving agent
(or the longest-lived deceased agent if none are alive).

Fitness is measured by total ticks survived × average energy level,
encouraging both longevity and resource-gathering ability.

This creates a population-level evolutionary pressure that works
*alongside* individual Hebbian learning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenerationRecord:
    """Tracks one agent's lifetime for evolutionary fitness."""
    agent_id: int
    generation: int = 0
    parent_id: int = -1
    ticks_survived: int = 0
    total_energy_gathered: float = 0.0
    mutations_applied: int = 0
    # Knowledge inherited
    cognitive_map_cells: int = 0
    cultural_items: int = 0
    personality_traits: np.ndarray | None = None

    @property
    def fitness(self) -> float:
        return self.ticks_survived * (1.0 + self.total_energy_gathered * 0.01)


class EvolutionEngine:
    """Manages generational inheritance and mutation of neural weights.

    Parameters
    ----------
    mutation_rate : float
        Fraction of weights that are perturbed during inheritance.
    mutation_strength : float
        Standard deviation of the Gaussian noise added to mutated weights.
    elitism : bool
        If True, the fittest agent is always the parent. If False,
        fitness-proportionate selection is used.
    """

    def __init__(self, mutation_rate: float = 0.05,
                 mutation_strength: float = 0.1,
                 elitism: bool = True) -> None:
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism = elitism

        self.generation_counter: int = 0
        self.records: dict[int, GenerationRecord] = {}
        self.graveyard: list[GenerationRecord] = []  # deceased records

    def register_agent(self, agent_id: int, parent_id: int = -1) -> None:
        """Register a new agent in the evolutionary ledger."""
        self.records[agent_id] = GenerationRecord(
            agent_id=agent_id,
            generation=self.generation_counter,
            parent_id=parent_id,
        )

    def record_tick(self, agent_id: int, energy_gained: float) -> None:
        """Update evolutionary fitness counters each tick."""
        rec = self.records.get(agent_id)
        if rec:
            rec.ticks_survived += 1
            rec.total_energy_gathered += energy_gained

    def agent_died(self, agent_id: int) -> None:
        """Move a dead agent's record to the graveyard."""
        rec = self.records.pop(agent_id, None)
        if rec:
            self.graveyard.append(rec)

    def select_parent(self, alive_agents: list) -> object | None:
        """Select the best parent for reproduction.

        Prefers live agents; falls back to the graveyard if none alive.
        Returns the agent object (or None).
        """
        if not alive_agents and not self.graveyard:
            return None

        if alive_agents:
            candidates = alive_agents
            # Fitness-proportionate or elitist selection
            if self.elitism:
                best = max(candidates,
                           key=lambda a: self.records.get(a.agent_id,
                                                          GenerationRecord(a.agent_id)).fitness)
                return best
            else:
                fits = np.array([
                    self.records.get(a.agent_id,
                                     GenerationRecord(a.agent_id)).fitness
                    for a in candidates
                ], dtype=np.float32)
                total = fits.sum()
                if total < 1e-9:
                    return random.choice(candidates)
                probs = fits / total
                return candidates[int(np.random.choice(len(candidates), p=probs))]

        # All dead — pick best from graveyard
        best_rec = max(self.graveyard, key=lambda r: r.fitness)
        return best_rec  # returns record, not agent

    def inherit_weights(self, parent_weights: np.ndarray) -> np.ndarray:
        """Create a child weight matrix by copying and mutating the parent's.

        Returns the mutated weight matrix.
        """
        child = parent_weights.copy()

        # Mutation mask
        mask = np.random.random(child.shape) < self.mutation_rate
        noise = np.random.randn(*child.shape).astype(np.float32) * self.mutation_strength
        child[mask] += noise[mask]

        mutations = int(mask.sum())
        self.generation_counter += 1

        return child, mutations

    def inherit_personality(self, parent_traits: np.ndarray) -> np.ndarray:
        """Create child personality traits by mutating parent's.

        Small perturbations so personality runs in families.
        """
        child = parent_traits.copy()
        noise = np.random.randn(*child.shape).astype(np.float32) * 0.08
        child += noise
        return np.clip(child, 0.1, 0.9)

    def inherit_cognitive_map(self, parent_map, child_map) -> int:
        """Transfer a fraction of the parent's spatial knowledge to the child.

        Child inherits ~40% of parent's explored cells (with noise).
        Returns number of cells inherited.
        """
        inherited = 0
        cells = list(parent_map.cells.items())
        random.shuffle(cells)
        count = int(len(cells) * 0.4)
        for (kx, ky), cell in cells[:count]:
            if (kx, ky) not in child_map.cells:
                from copy import copy
                new_cell = copy(cell)
                # Degrade knowledge slightly (not perfect memory)
                new_cell.crystal_score *= 0.6
                new_cell.danger_score *= 0.6
                new_cell.visit_count = max(1, new_cell.visit_count // 3)
                child_map.cells[(kx, ky)] = new_cell
                inherited += 1
        return inherited

    def get_summary(self) -> dict:
        alive_records = list(self.records.values())
        dead_records = self.graveyard
        best_fitness = 0.0
        if alive_records:
            best_fitness = max(r.fitness for r in alive_records)
        if dead_records:
            best_fitness = max(best_fitness, max(r.fitness for r in dead_records))

        return {
            "generation": self.generation_counter,
            "alive_agents": len(alive_records),
            "total_deaths": len(dead_records),
            "best_fitness": best_fitness,
        }
