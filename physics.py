"""Physics engine for the sandbox environment."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Vec2:
    """2D vector for positions and velocities."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self) -> Vec2:
        mag = self.magnitude()
        if mag < 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / mag, self.y / mag)

    def distance_to(self, other: Vec2) -> float:
        return (self - other).magnitude()

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def grid_pos(self) -> tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))


GRAVITY = 0.3  # downward pull per tick (used for fall calculations)
FRICTION = 0.65  # velocity damping per tick (higher = more sliding)
MAX_SPEED = 1.8  # max cells per tick


def apply_physics(position: Vec2, velocity: Vec2) -> tuple[Vec2, Vec2]:
    """Apply friction and speed cap, return new position and velocity."""
    velocity = velocity * FRICTION
    speed = velocity.magnitude()
    if speed > MAX_SPEED:
        velocity = velocity.normalized() * MAX_SPEED
    new_pos = position + velocity
    return new_pos, velocity


def check_collision(pos: Vec2, obstacles: set[tuple[int, int]],
                    width: int, height: int) -> tuple[Vec2, bool]:
    """Clamp position to world bounds and check obstacle collision.

    Returns corrected position and whether a collision occurred.
    """
    collided = False
    gx, gy = pos.grid_pos()

    # World boundary
    if pos.x < 0 or pos.x >= width or pos.y < 0 or pos.y >= height:
        pos = Vec2(max(0, min(width - 1, pos.x)),
                   max(0, min(height - 1, pos.y)))
        collided = True

    # Obstacle collision
    gx, gy = pos.grid_pos()
    if (gx, gy) in obstacles:
        collided = True

    return pos, collided


def check_agent_collision(pos: Vec2, other_positions: list[Vec2],
                          push_force: float = 0.5,
                          collision_radius: float = 1.2) -> tuple[Vec2, bool]:
    """Check for and resolve agent-agent collisions.

    Returns the adjusted position and whether a collision occurred.
    Agents push each other apart when overlapping.
    """
    collided = False
    for other_pos in other_positions:
        dist = pos.distance_to(other_pos)
        if dist < collision_radius and dist > 0.01:
            # Push away from other agent
            diff = pos - other_pos
            push = diff.normalized() * push_force * (collision_radius - dist)
            pos = pos + push
            collided = True
    return pos, collided
