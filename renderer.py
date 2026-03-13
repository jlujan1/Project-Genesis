"""Terminal-based renderer for the sandbox environment.

Draws the world map, agent positions, crystals, and obstacles
using Unicode characters in the terminal.
"""

from __future__ import annotations

import sys

from genesis.agent.agent import ConsciousAgent
from genesis.environment.sandbox import Sandbox
from genesis.environment.resources import (
    BIOME_OCEAN, BIOME_WETLANDS, BIOME_GRASSLANDS, BIOME_DESERT, BIOME_MOUNTAINS,
)


# Characters for rendering
EMPTY = "·"
OBSTACLE = "█"
CRYSTAL = "◆"
CRYSTAL_DIM = "◇"
AGENT_CHARS = ["A", "B", "C", "D"]  # up to 4 agents
AGENT_DEAD = "✕"
SHELTER = "⌂"
PREDATOR = "☠"

# Biome terrain characters
BIOME_CHARS = {
    BIOME_OCEAN: "~",
    BIOME_WETLANDS: "≈",
    BIOME_GRASSLANDS: "·",
    BIOME_DESERT: "∵",
    BIOME_MOUNTAINS: "▲",
}


class Renderer:
    """Renders a viewport of the sandbox to the terminal."""

    def __init__(self, viewport_width: int = 50, viewport_height: int = 25) -> None:
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

    def render(self, sandbox: Sandbox, agents: list[ConsciousAgent],
               follow_agent: int = 0) -> str:
        """Render the sandbox centered on a specific agent."""
        lines = []

        # Determine viewport center
        if agents and follow_agent < len(agents):
            center_x = int(agents[follow_agent].position.x)
            center_y = int(agents[follow_agent].position.y)
        else:
            center_x = sandbox.width // 2
            center_y = sandbox.height // 2

        # Calculate viewport bounds
        half_w = self.viewport_width // 2
        half_h = self.viewport_height // 2
        start_x = max(0, center_x - half_w)
        start_y = max(0, center_y - half_h)
        end_x = min(sandbox.width, start_x + self.viewport_width)
        end_y = min(sandbox.height, start_y + self.viewport_height)

        # Build agent position lookup
        agent_positions: dict[tuple[int, int], tuple[int, bool]] = {}
        for agent in agents:
            gx, gy = agent.position.grid_pos()
            agent_positions[(gx, gy)] = (agent.agent_id, agent.alive)

        # Build crystal position lookup
        crystal_positions: dict[tuple[int, int], float] = {}
        for crystal in sandbox.crystals:
            if not crystal.consumed and not crystal.is_expired:
                cx, cy = crystal.position.grid_pos()
                crystal_positions[(cx, cy)] = crystal.freshness

        # Build predator position lookup
        predator_positions: set[tuple[int, int]] = set()
        for p in sandbox.predators.predators:
            predator_positions.add((int(p.x), int(p.y)))

        # Header
        ws = sandbox.weather.get_summary()
        dn = "NIGHT" if sandbox.day_cycle.is_night else "DAY"
        header_info = f" {dn} | {ws['season']} {ws['weather']} | ☠{len(sandbox.predators.predators)} "
        border_len = end_x - start_x
        header_pad = max(0, border_len - len(header_info))
        lines.append("┌" + header_info + "─" * header_pad + "┐")

        # Render grid
        for y in range(start_y, end_y):
            row = "│"
            for x in range(start_x, end_x):
                if (x, y) in agent_positions:
                    aid, alive = agent_positions[(x, y)]
                    if alive:
                        char = AGENT_CHARS[aid % len(AGENT_CHARS)]
                    else:
                        char = AGENT_DEAD
                elif (x, y) in predator_positions:
                    char = PREDATOR
                elif (x, y) in crystal_positions:
                    freshness = crystal_positions[(x, y)]
                    char = CRYSTAL if freshness > 0.5 else CRYSTAL_DIM
                elif (x, y) in sandbox.obstacles:
                    char = OBSTACLE
                elif (x, y) in sandbox.shelters:
                    char = SHELTER
                else:
                    # Show biome character
                    biome = sandbox.biome_map.biome_at(x, y)
                    char = BIOME_CHARS.get(biome, EMPTY)
                row += char
            row += "│"
            lines.append(row)

        # Footer
        lines.append("└" + "─" * (end_x - start_x) + "┘")

        # Legend
        legend = (f"  {AGENT_CHARS[0]}=Agent 0  {AGENT_CHARS[1]}=Agent 1  "
                  f"{CRYSTAL}=Crystal  {OBSTACLE}=Wall  "
                  f"{PREDATOR}=Predator  {SHELTER}=Shelter")
        lines.append(legend)

        return "\n".join(lines)

    def render_compact_stats(self, agents: list[ConsciousAgent]) -> str:
        """Compact one-line stat bar per agent."""
        lines = []
        for agent in agents:
            status = "ALIVE" if agent.alive else "DEAD"
            e = agent.body.energy
            i = agent.body.integrity
            phi_data = agent.phi_calculator.get_consciousness_assessment(
                self_model_accuracy=agent.self_model.model_accuracy,
                attention_accuracy=agent.attention_schema.schema_accuracy,
                metacognitive_confidence=agent.inner_speech.confidence,
                binding_coherence=agent.binding.coherence,
                empowerment=agent.empowerment.empowerment,
                narrative_identity=agent.narrative.identity_strength,
                curiosity_level=agent.curiosity.curiosity_level,
            )
            phi = phi_data["composite_score"]
            ego = "EGO" if agent.self_model.has_ego else "---"
            ws = agent.workspace.get_broadcast_summary()["current_source"]

            lines.append(
                f"  Agent {agent.agent_id} [{status}] "
                f"E:{e:5.1f} I:{i:5.1f} "
                f"Φ:{phi:.3f} {ego} ws:{ws}"
            )
        return "\n".join(lines)
