"""Predators — mobile threats that roam the world and hunt agents.

Predators are autonomous NPCs that patrol the map, pursue nearby agents,
and deal damage on contact.  Agents can flee, cooperate to scare them
away, or hide in shelters for protection.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from genesis.environment.physics import Vec2


@dataclass
class Predator:
    """A roaming threat in the environment."""
    predator_id: int
    position: Vec2
    speed: float = 0.35
    damage: float = 3.0          # energy damage per contact tick
    integrity_damage: float = 1.5  # integrity damage per contact
    detection_range: float = 10.0
    attack_range: float = 1.8
    # Patrol state
    patrol_target: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    patrol_cooldown: int = 0
    # Scare mechanics — enough nearby agents scare the predator away
    flee_threshold: int = 3  # agents within range to scare it
    scared_ticks: int = 0
    alive: bool = True

    def grid_pos(self) -> tuple[int, int]:
        return int(self.position.x), int(self.position.y)


class PredatorSystem:
    """Manages all predators in the simulation.

    Predators spawn periodically in dangerous biomes and roam the map.
    They chase agents they detect and deal damage on contact.
    """

    def __init__(self, width: int, height: int, max_predators: int = 6,
                 spawn_interval: int = 2500, seed: int = 42) -> None:
        self.width = width
        self.height = height
        self.max_predators = max_predators
        self.spawn_interval = spawn_interval
        self.rng = random.Random(seed + 999)
        self.predators: list[Predator] = []
        self._next_id = 0
        self._spawn_timer = spawn_interval // 2  # first spawn comes sooner

    def tick(self, agent_positions: list[tuple[Vec2, bool]],
             obstacles: set[tuple[int, int]],
             biome_map=None) -> list[tuple[int, float, float]]:
        """Advance all predators one tick.

        Parameters
        ----------
        agent_positions : list of (position, alive) for each agent
        obstacles : set of blocked grid positions
        biome_map : optional BiomeMap for biome-aware spawning

        Returns
        -------
        contacts : list of (agent_index, energy_damage, integrity_damage)
            for each agent currently being attacked
        """
        contacts: list[tuple[int, float, float]] = []

        # Spawn new predators periodically
        self._spawn_timer -= 1
        if self._spawn_timer <= 0 and len(self.predators) < self.max_predators:
            self._spawn_predator(obstacles, biome_map)
            self._spawn_timer = self.spawn_interval

        alive_positions = [
            (i, pos) for i, (pos, alive) in enumerate(agent_positions)
            if alive
        ]

        for pred in self.predators:
            if not pred.alive:
                continue

            # Check scare mechanic — enough nearby agents scare it away
            agents_nearby = sum(
                1 for _, apos in alive_positions
                if pred.position.distance_to(apos) < pred.detection_range
            )
            if agents_nearby >= pred.flee_threshold:
                pred.scared_ticks = 60  # flee for 60 ticks

            if pred.scared_ticks > 0:
                pred.scared_ticks -= 1
                # Flee away from centroid of nearby agents
                if alive_positions:
                    cx = sum(p.x for _, p in alive_positions) / len(alive_positions)
                    cy = sum(p.y for _, p in alive_positions) / len(alive_positions)
                    dx = pred.position.x - cx
                    dy = pred.position.y - cy
                    dist = math.sqrt(dx * dx + dy * dy) + 0.01
                    pred.position = Vec2(
                        max(1.0, min(self.width - 2.0,
                                     pred.position.x + dx / dist * pred.speed * 1.5)),
                        max(1.0, min(self.height - 2.0,
                                     pred.position.y + dy / dist * pred.speed * 1.5)),
                    )
                continue

            # Find nearest agent
            nearest_idx = -1
            nearest_dist = pred.detection_range + 1.0
            for agent_idx, apos in alive_positions:
                d = pred.position.distance_to(apos)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = agent_idx

            if nearest_idx >= 0 and nearest_dist < pred.detection_range:
                # Chase agent
                target = alive_positions[[i for i, _ in
                                          [(j, p) for j, p in enumerate(alive_positions)
                                           if p[0] == nearest_idx]][0]][1] \
                    if False else None
                # Simpler: find the position
                for aidx, apos in alive_positions:
                    if aidx == nearest_idx:
                        target = apos
                        break
                if target is not None:
                    dx = target.x - pred.position.x
                    dy = target.y - pred.position.y
                    dist = math.sqrt(dx * dx + dy * dy) + 0.01
                    new_x = pred.position.x + dx / dist * pred.speed
                    new_y = pred.position.y + dy / dist * pred.speed
                    # Don't move into obstacles
                    gx, gy = int(new_x), int(new_y)
                    if (gx, gy) not in obstacles:
                        pred.position = Vec2(
                            max(1.0, min(self.width - 2.0, new_x)),
                            max(1.0, min(self.height - 2.0, new_y)),
                        )

                # Attack if close enough
                if nearest_dist < pred.attack_range:
                    contacts.append((nearest_idx, pred.damage, pred.integrity_damage))
            else:
                # Patrol — wander randomly
                self._patrol(pred, obstacles)

        # Remove dead predators
        self.predators = [p for p in self.predators if p.alive]
        return contacts

    def _patrol(self, pred: Predator, obstacles: set[tuple[int, int]]) -> None:
        """Random wandering behaviour."""
        pred.patrol_cooldown -= 1
        if pred.patrol_cooldown <= 0:
            pred.patrol_target = Vec2(
                self.rng.uniform(5, self.width - 5),
                self.rng.uniform(5, self.height - 5),
            )
            pred.patrol_cooldown = self.rng.randint(80, 200)

        dx = pred.patrol_target.x - pred.position.x
        dy = pred.patrol_target.y - pred.position.y
        dist = math.sqrt(dx * dx + dy * dy) + 0.01
        speed = pred.speed * 0.5  # patrol is slower
        new_x = pred.position.x + dx / dist * speed
        new_y = pred.position.y + dy / dist * speed
        gx, gy = int(new_x), int(new_y)
        if (gx, gy) not in obstacles:
            pred.position = Vec2(
                max(1.0, min(self.width - 2.0, new_x)),
                max(1.0, min(self.height - 2.0, new_y)),
            )

    def _spawn_predator(self, obstacles: set[tuple[int, int]],
                        biome_map=None) -> None:
        """Spawn a predator at the map edge (they enter from outside)."""
        for _ in range(50):
            # Spawn at edge
            edge = self.rng.choice(["top", "bottom", "left", "right"])
            if edge == "top":
                x, y = self.rng.randint(1, self.width - 2), 1
            elif edge == "bottom":
                x, y = self.rng.randint(1, self.width - 2), self.height - 2
            elif edge == "left":
                x, y = 1, self.rng.randint(1, self.height - 2)
            else:
                x, y = self.width - 2, self.rng.randint(1, self.height - 2)
            if (x, y) not in obstacles:
                pred = Predator(
                    predator_id=self._next_id,
                    position=Vec2(float(x), float(y)),
                    patrol_target=Vec2(float(self.width // 2),
                                       float(self.height // 2)),
                )
                self.predators.append(pred)
                self._next_id += 1
                return

    def get_visible_predators(self, pos: Vec2, vision_range: float
                              ) -> list[dict]:
        """Return predator info visible from a position."""
        result = []
        for pred in self.predators:
            if not pred.alive:
                continue
            d = pos.distance_to(pred.position)
            if d <= vision_range:
                result.append({
                    "id": pred.predator_id,
                    "x": pred.position.x,
                    "y": pred.position.y,
                    "distance": d,
                    "scared": pred.scared_ticks > 0,
                })
        return result

    def get_nearest_predator_direction(self, pos: Vec2, vision_range: float
                                       ) -> Vec2 | None:
        """Return unit vector toward nearest visible predator, or None."""
        nearest_dist = vision_range + 1
        nearest_pos = None
        for pred in self.predators:
            if not pred.alive:
                continue
            d = pos.distance_to(pred.position)
            if d < nearest_dist:
                nearest_dist = d
                nearest_pos = pred.position
        if nearest_pos is None:
            return None
        dx = nearest_pos.x - pos.x
        dy = nearest_pos.y - pos.y
        dist = math.sqrt(dx * dx + dy * dy) + 0.01
        return Vec2(dx / dist, dy / dist)
