"""Specialized brain modules — the subconscious processing clusters.

Each module processes data independently and submits packets to the
Global Workspace with a relevance/confidence score for competitive broadcast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorkspacePacket:
    """A data packet submitted by a module to the Global Workspace.

    The module with the highest relevance score wins the broadcast.
    """
    source: str            # which module produced this
    data: np.ndarray       # the encoded representation
    relevance: float       # how important/urgent this data is (0–1+)
    metadata: dict = field(default_factory=dict)


class VisionModule:
    """Processes visible cell data into a compressed neural representation.

    Encodes nearby obstacles, crystals, agents, and terrain features
    into a fixed-size vector for the SNN sensory neurons.
    """

    def __init__(self, encoding_size: int = 32) -> None:
        self.encoding_size = encoding_size
        self.last_crystal_distance = float("inf")
        self.last_obstacle_proximity = 0.0
        self.last_agent_distance = float("inf")

    def process(self, visible_cells: list[dict],
                agent_positions: list[tuple[float, float]] | None = None,
                own_pos: tuple[float, float] = (0, 0)) -> WorkspacePacket:
        """Convert raw visual data into an encoded packet."""
        encoding = np.zeros(self.encoding_size, dtype=np.float32)
        if not visible_cells:
            return WorkspacePacket(source="vision", data=encoding, relevance=0.1)

        # Summarize visual field
        num_obstacles = 0
        nearest_crystal_dist = float("inf")
        nearest_crystal_dir = np.zeros(2)
        nearest_obstacle_dist = float("inf")
        total_elevation = 0.0
        crystal_count = 0

        # Berry / fungi / ruin tracking
        nearest_berry_dist = float("inf")
        nearest_berry_dir = np.zeros(2)
        nearest_fungus_dist = float("inf")
        nearest_fungus_dir = np.zeros(2)
        nearest_ruin_dist = float("inf")
        nearest_ruin_dir = np.zeros(2)
        berry_count = 0
        fungus_count = 0

        for cell in visible_cells:
            d = cell["distance"]
            if cell["is_obstacle"]:
                num_obstacles += 1
                if d < nearest_obstacle_dist:
                    nearest_obstacle_dist = d
            if cell["has_crystal"]:
                crystal_count += 1
                if d < nearest_crystal_dist:
                    nearest_crystal_dist = d
                    dx = cell["x"] - own_pos[0]
                    dy = cell["y"] - own_pos[1]
                    mag = max(0.01, math.sqrt(dx * dx + dy * dy))
                    nearest_crystal_dir = np.array([dx / mag, dy / mag])
            if cell.get("has_berry"):
                berry_count += 1
                if d < nearest_berry_dist:
                    nearest_berry_dist = d
                    dx = cell["x"] - own_pos[0]
                    dy = cell["y"] - own_pos[1]
                    mag = max(0.01, math.sqrt(dx * dx + dy * dy))
                    nearest_berry_dir = np.array([dx / mag, dy / mag])
            if cell.get("has_fungus"):
                fungus_count += 1
                if d < nearest_fungus_dist:
                    nearest_fungus_dist = d
                    dx = cell["x"] - own_pos[0]
                    dy = cell["y"] - own_pos[1]
                    mag = max(0.01, math.sqrt(dx * dx + dy * dy))
                    nearest_fungus_dir = np.array([dx / mag, dy / mag])
            if cell.get("has_ruin"):
                if d < nearest_ruin_dist:
                    nearest_ruin_dist = d
                    dx = cell["x"] - own_pos[0]
                    dy = cell["y"] - own_pos[1]
                    mag = max(0.01, math.sqrt(dx * dx + dy * dy))
                    nearest_ruin_dir = np.array([dx / mag, dy / mag])
            total_elevation += cell["elevation"]

        avg_elevation = total_elevation / max(1, len(visible_cells))
        self.last_crystal_distance = nearest_crystal_dist
        self.last_obstacle_proximity = 1.0 / max(1.0, nearest_obstacle_dist)

        # Encode into vector
        encoding[0] = min(1.0, num_obstacles / 20.0)  # obstacle density
        encoding[1] = 1.0 / max(1.0, nearest_obstacle_dist)  # obstacle proximity
        encoding[2] = 1.0 / max(1.0, nearest_crystal_dist)  # crystal proximity
        encoding[3] = nearest_crystal_dir[0] if crystal_count > 0 else 0.0  # crystal dir x
        encoding[4] = nearest_crystal_dir[1] if crystal_count > 0 else 0.0  # crystal dir y
        encoding[5] = min(1.0, crystal_count / 5.0)  # crystal abundance
        encoding[6] = avg_elevation / 3.0  # terrain
        encoding[7] = min(1.0, len(visible_cells) / 100.0)  # visibility extent

        # Other agents in vision
        if agent_positions:
            nearest_agent_dist = float("inf")
            nearest_agent_dir = np.zeros(2)
            for ax, ay in agent_positions:
                ad = math.sqrt((ax - own_pos[0]) ** 2 + (ay - own_pos[1]) ** 2)
                if ad < nearest_agent_dist and ad > 0.1:
                    nearest_agent_dist = ad
                    dx = ax - own_pos[0]
                    dy = ay - own_pos[1]
                    mag = max(0.01, math.sqrt(dx * dx + dy * dy))
                    nearest_agent_dir = np.array([dx / mag, dy / mag])

            self.last_agent_distance = nearest_agent_dist
            encoding[8] = 1.0 / max(1.0, nearest_agent_dist)
            encoding[9] = nearest_agent_dir[0]
            encoding[10] = nearest_agent_dir[1]

        # Split crystal direction into positive-only channels for motor wiring
        if crystal_count > 0:
            encoding[11] = max(0.0, nearest_crystal_dir[0])   # crystal to right
            encoding[12] = max(0.0, -nearest_crystal_dir[0])  # crystal to left
            encoding[13] = max(0.0, nearest_crystal_dir[1])   # crystal below
            encoding[14] = max(0.0, -nearest_crystal_dir[1])  # crystal above
            # Proximity-gated approach signal (stronger when closer)
            encoding[15] = min(1.0, 1.0 / max(1.0, nearest_crystal_dist))

        # Biome awareness — encode the dominant nearby biome
        biome_counts = [0] * 5  # one per biome type
        for cell in visible_cells:
            b = cell.get("biome", 2)  # default grasslands
            if 0 <= b < 5:
                biome_counts[b] += 1
        total_vis = max(1, sum(biome_counts))
        for bi in range(5):
            encoding[16 + bi] = biome_counts[bi] / total_vis

        # Berry bush direction & proximity (slots 21-23)
        if berry_count > 0:
            encoding[21] = max(0.0, nearest_berry_dir[0])   # berry right
            encoding[22] = max(0.0, -nearest_berry_dir[0])  # berry left
            encoding[23] = min(1.0, 1.0 / max(1.0, nearest_berry_dist))  # berry proximity

        # Fungus direction & proximity (slots 24-26)
        if fungus_count > 0:
            encoding[24] = max(0.0, nearest_fungus_dir[0])   # fungus right
            encoding[25] = max(0.0, -nearest_fungus_dir[0])  # fungus left
            encoding[26] = min(1.0, 1.0 / max(1.0, nearest_fungus_dist))  # fungus proximity

        # Ruin direction & proximity (slots 27-29)
        if nearest_ruin_dist < float("inf"):
            encoding[27] = max(0.0, nearest_ruin_dir[0])    # ruin right
            encoding[28] = max(0.0, -nearest_ruin_dir[0])   # ruin left
            encoding[29] = min(1.0, 1.0 / max(1.0, nearest_ruin_dist))  # ruin proximity

        # River proximity (slot 30), shelter proximity (slot 31)
        nearest_river_dist = float("inf")
        nearest_shelter_dist = float("inf")
        for cell in visible_cells:
            d = cell["distance"]
            if cell.get("is_river") and d < nearest_river_dist:
                nearest_river_dist = d
            if cell.get("has_shelter") and d < nearest_shelter_dist:
                nearest_shelter_dist = d
        encoding[30] = min(1.0, 1.0 / max(1.0, nearest_river_dist))
        encoding[31] = min(1.0, 1.0 / max(1.0, nearest_shelter_dist))

        # Relevance: obstacles nearby = high, crystals nearby = high
        relevance = 0.2
        if nearest_obstacle_dist < 3:
            relevance += 0.4
        if nearest_crystal_dist < 5:
            relevance += 0.3
        if nearest_berry_dist < 5 or nearest_fungus_dist < 5:
            relevance += 0.2
        if agent_positions and self.last_agent_distance < 8:
            relevance += 0.3

        return WorkspacePacket(
            source="vision", data=encoding, relevance=min(1.0, relevance),
            metadata={"nearest_crystal": nearest_crystal_dist,
                      "nearest_obstacle": nearest_obstacle_dist,
                      "crystal_count": crystal_count}
        )


class AudioModule:
    """Processes audio events into directional awareness signals."""

    def __init__(self, encoding_size: int = 8) -> None:
        self.encoding_size = encoding_size

    def process(self, audio_events: list[tuple], own_pos: tuple[float, float],
                own_id: int) -> WorkspacePacket:
        """Encode audio events (direction, intensity, novelty)."""
        encoding = np.zeros(self.encoding_size, dtype=np.float32)
        if not audio_events:
            return WorkspacePacket(source="audio", data=encoding, relevance=0.05)

        loudest_intensity = 0.0
        loudest_dir = np.zeros(2)
        total_events = 0

        for pos, intensity, src_id in audio_events:
            if src_id == own_id:
                continue  # ignore own sounds
            total_events += 1
            if intensity > loudest_intensity:
                loudest_intensity = intensity
                dx = pos.x - own_pos[0]
                dy = pos.y - own_pos[1]
                mag = max(0.01, math.sqrt(dx * dx + dy * dy))
                loudest_dir = np.array([dx / mag, dy / mag])

        encoding[0] = min(1.0, loudest_intensity)
        encoding[1] = loudest_dir[0]
        encoding[2] = loudest_dir[1]
        encoding[3] = min(1.0, total_events / 5.0)

        relevance = min(1.0, loudest_intensity * 0.8)
        return WorkspacePacket(
            source="audio", data=encoding, relevance=relevance,
            metadata={"num_sounds": total_events, "loudest": loudest_intensity}
        )


class ProprioceptionModule:
    """Monitors the agent's own body state — the 'inside data'.

    Reports joint angles, momentum, body boundary awareness, and
    internal homeostasis readings.
    """

    def __init__(self, encoding_size: int = 12) -> None:
        self.encoding_size = encoding_size

    def process(self, energy: float, max_energy: float,
                integrity: float, max_integrity: float,
                velocity: tuple[float, float],
                position: tuple[float, float]) -> WorkspacePacket:
        """Encode the agent's internal state."""
        encoding = np.zeros(self.encoding_size, dtype=np.float32)

        e_ratio = energy / max(1.0, max_energy)
        i_ratio = integrity / max(1.0, max_integrity)
        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        encoding[0] = e_ratio
        encoding[1] = i_ratio
        encoding[2] = velocity[0] / 3.0  # normalized velocity
        encoding[3] = velocity[1] / 3.0
        encoding[4] = speed / 3.0
        encoding[5] = position[0] / 80.0  # normalized position
        encoding[6] = position[1] / 60.0
        # Derived signals
        encoding[7] = 1.0 - e_ratio  # hunger signal (inverted energy)
        encoding[8] = 1.0 - i_ratio  # damage signal (inverted integrity)

        # Relevance: urgent when low on energy or integrity
        relevance = 0.1
        if e_ratio < 0.3:
            relevance += 0.5
        if i_ratio < 0.3:
            relevance += 0.4
        if e_ratio < 0.1 or i_ratio < 0.1:
            relevance = 1.0  # critical alarm

        return WorkspacePacket(
            source="proprioception", data=encoding, relevance=relevance,
            metadata={"energy_ratio": e_ratio, "integrity_ratio": i_ratio}
        )


class PredictionModule:
    """Constantly predicts the next state conditioned on the last action.

    Uses a simple action-conditional linear model:
        next_state = W_state @ state + W_action[:, action]
    The error signal drives attention and learning — large errors mean
    something unexpected happened, demanding conscious attention.
    """

    def __init__(self, state_size: int = 16, num_actions: int = 8) -> None:
        self.state_size = state_size
        self.num_actions = num_actions
        self.last_prediction = np.zeros(state_size, dtype=np.float32)
        self.last_actual = np.zeros(state_size, dtype=np.float32)
        self.last_action: int = 0
        # Action-conditional prediction weights
        self.pred_weights = np.random.randn(state_size, state_size).astype(np.float32) * 0.1
        self.action_weights = np.random.randn(state_size, num_actions).astype(np.float32) * 0.05
        self.learning_rate = 0.02
        self.cumulative_error = 0.0

    def process(self, current_state: np.ndarray,
                last_action: int = 0) -> WorkspacePacket:
        """Compare prediction to reality and generate error-based packet."""
        state = current_state[:self.state_size]
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)))

        # Prediction error
        error = state - self.last_prediction
        error_magnitude = float(np.sqrt(np.mean(error ** 2)))
        self.cumulative_error = 0.9 * self.cumulative_error + 0.1 * error_magnitude

        # Learn from error (gradient descent on action-conditional model)
        if np.any(self.last_actual != 0):
            gradient_state = np.outer(error, self.last_actual)
            self.pred_weights += self.learning_rate * gradient_state
            # Action-specific gradient
            if 0 <= self.last_action < self.num_actions:
                self.action_weights[:, self.last_action] += self.learning_rate * error
            # Prevent weights from exploding
            norm = np.linalg.norm(self.pred_weights)
            if norm > 10.0:
                self.pred_weights *= 10.0 / norm
            norm_a = np.linalg.norm(self.action_weights)
            if norm_a > 5.0:
                self.action_weights *= 5.0 / norm_a

        # Make next prediction (action-conditional)
        self.last_prediction = self.pred_weights @ state
        if 0 <= last_action < self.num_actions:
            self.last_prediction += self.action_weights[:, last_action]
        self.last_actual = state.copy()
        self.last_action = last_action

        # Encode for workspace
        encoding = np.zeros(self.state_size, dtype=np.float32)
        encoding[:len(error)] = error
        # High error = high relevance (something unexpected happened)
        relevance = min(1.0, error_magnitude * 2.0)

        return WorkspacePacket(
            source="prediction", data=encoding, relevance=relevance,
            metadata={"error_magnitude": error_magnitude,
                      "cumulative_error": self.cumulative_error}
        )
