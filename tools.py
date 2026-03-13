"""Tool Crafting — agents can create and use simple tools.

Tools provide persistent bonuses when carried.  Each costs energy to
craft and degrades over time.  Agents start with no tools and must
learn (via curiosity/study actions) that crafting is beneficial.

Tool types:
  HARVEST_TOOL — 50% more energy from crystals
  COMPASS — +30% vision range
  REPAIR_KIT — passive integrity recovery boost
"""

from __future__ import annotations

from dataclasses import dataclass, field

TOOL_NONE = -1
TOOL_HARVEST = 0
TOOL_COMPASS = 1
TOOL_REPAIR_KIT = 2
NUM_TOOL_TYPES = 3
TOOL_NAMES = ["harvest_tool", "compass", "repair_kit"]

# Crafting costs and stats
TOOL_CRAFT_ENERGY = [8.0, 6.0, 10.0]     # energy cost to craft
TOOL_DURABILITY = [800, 600, 1000]         # max ticks before breaking
TOOL_MAX_CARRY = 2                         # max tools an agent can hold


@dataclass
class Tool:
    """A single crafted tool."""
    tool_type: int
    durability: int = 0  # remaining ticks
    max_durability: int = 0

    def __post_init__(self) -> None:
        if self.max_durability == 0:
            self.max_durability = TOOL_DURABILITY[self.tool_type]
            self.durability = self.max_durability

    @property
    def is_broken(self) -> bool:
        return self.durability <= 0

    @property
    def condition(self) -> float:
        """0.0 = broken, 1.0 = pristine."""
        return max(0.0, self.durability / max(1, self.max_durability))

    def tick(self) -> None:
        """Age the tool by one tick."""
        if self.durability > 0:
            self.durability -= 1


class ToolInventory:
    """Manages an agent's tool collection."""

    def __init__(self) -> None:
        self.tools: list[Tool] = []
        self.total_crafted: int = 0

    def can_craft(self, tool_type: int, energy: float) -> bool:
        """Check if crafting is possible."""
        if tool_type < 0 or tool_type >= NUM_TOOL_TYPES:
            return False
        if len(self.tools) >= TOOL_MAX_CARRY:
            return False
        if energy < TOOL_CRAFT_ENERGY[tool_type]:
            return False
        return True

    def craft(self, tool_type: int) -> float:
        """Craft a tool. Returns energy cost (caller must deduct)."""
        tool = Tool(tool_type=tool_type)
        self.tools.append(tool)
        self.total_crafted += 1
        return TOOL_CRAFT_ENERGY[tool_type]

    def tick(self) -> None:
        """Age all tools, remove broken ones."""
        for t in self.tools:
            t.tick()
        self.tools = [t for t in self.tools if not t.is_broken]

    def has_tool(self, tool_type: int) -> bool:
        """Check if agent has a working tool of this type."""
        return any(t.tool_type == tool_type and not t.is_broken
                   for t in self.tools)

    def get_harvest_bonus(self) -> float:
        """Crystal energy multiplier from harvest tools."""
        if self.has_tool(TOOL_HARVEST):
            return 1.5
        return 1.0

    def get_vision_bonus(self) -> float:
        """Vision range multiplier from compass."""
        if self.has_tool(TOOL_COMPASS):
            return 1.3
        return 1.0

    def get_repair_rate(self) -> float:
        """Extra integrity recovery per tick from repair kit."""
        if self.has_tool(TOOL_REPAIR_KIT):
            return 0.03
        return 0.0

    def get_summary(self) -> dict:
        return {
            "tools": [TOOL_NAMES[t.tool_type] for t in self.tools],
            "total_crafted": self.total_crafted,
            "tool_count": len(self.tools),
        }

    def get_encoding(self) -> list[float]:
        """Return a fixed-size encoding for SNN/workspace injection."""
        enc = [0.0] * NUM_TOOL_TYPES
        for t in self.tools:
            enc[t.tool_type] = t.condition
        return enc
