"""Civilization progression — emergent societal development mirroring Earth's history.

Agents collectively advance through epochs as they accumulate technologies,
build structures, farm, specialise, and grow their population.  Each epoch
unlocks richer capabilities, creating a positive feedback loop that pushes
the simulation from nomadic bands toward city-states.

Epoch ladder:
  0  Nomadic          – wandering survival, basic tool use
  1  Hunter-Gatherer  – cooperation, proto-language, crafted tools
  2  Agricultural     – farming, food surplus, first permanent shelters
  3  Settlement       – multiple building types, role specialisation
  4  Village          – trade routes, governance, cultural identity
  5  Town             – writing, astronomy, medicine, organised society
  6  City-State       – mathematics, engineering, law, philosophy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.environment.physics import Vec2


# ═══════════════════════════════════════════════════════════════
#  Epochs
# ═══════════════════════════════════════════════════════════════

class Epoch(IntEnum):
    NOMADIC = 0
    HUNTER_GATHERER = 1
    AGRICULTURAL = 2
    SETTLEMENT = 3
    VILLAGE = 4
    TOWN = 5
    CITY_STATE = 6

EPOCH_NAMES = [
    "Nomadic", "Hunter-Gatherer", "Agricultural",
    "Settlement", "Village", "Town", "City-State",
]

# Max population allowed per epoch
EPOCH_POP_CAP = [4, 6, 10, 16, 24, 32, 48]

# ═══════════════════════════════════════════════════════════════
#  Technologies
# ═══════════════════════════════════════════════════════════════

class Tech(IntEnum):
    FIRE            = 0
    STONE_TOOLS     = 1
    POTTERY         = 2
    WEAVING         = 3
    AGRICULTURE     = 4
    IRRIGATION      = 5
    ANIMAL_HUSBANDRY = 6
    ARCHITECTURE    = 7
    MASONRY         = 8
    FOOD_PRESERVATION = 9
    METALLURGY      = 10
    WHEEL           = 11
    TRADE_ROUTES    = 12
    WRITING         = 13
    ASTRONOMY       = 14
    MEDICINE        = 15
    GOVERNANCE      = 16
    MATHEMATICS     = 17
    ENGINEERING     = 18
    LAW             = 19
    PHILOSOPHY      = 20

NUM_TECHS = 21

TECH_NAMES = [
    "Fire", "Stone Tools", "Pottery", "Weaving",
    "Agriculture", "Irrigation", "Animal Husbandry",
    "Architecture", "Masonry", "Food Preservation",
    "Metallurgy", "Wheel", "Trade Routes",
    "Writing", "Astronomy", "Medicine", "Governance",
    "Mathematics", "Engineering", "Law", "Philosophy",
]

# Prerequisites: tech → list of required techs
TECH_PREREQUISITES: dict[Tech, list[Tech]] = {
    Tech.FIRE:              [],
    Tech.STONE_TOOLS:       [],
    Tech.POTTERY:           [Tech.FIRE],
    Tech.WEAVING:           [Tech.STONE_TOOLS],
    Tech.AGRICULTURE:       [Tech.STONE_TOOLS],
    Tech.IRRIGATION:        [Tech.AGRICULTURE],
    Tech.ANIMAL_HUSBANDRY:  [Tech.AGRICULTURE],
    Tech.ARCHITECTURE:      [Tech.STONE_TOOLS, Tech.FIRE],
    Tech.MASONRY:           [Tech.ARCHITECTURE],
    Tech.FOOD_PRESERVATION: [Tech.POTTERY, Tech.AGRICULTURE],
    Tech.METALLURGY:        [Tech.FIRE, Tech.STONE_TOOLS],
    Tech.WHEEL:             [Tech.STONE_TOOLS],
    Tech.TRADE_ROUTES:      [Tech.WHEEL],
    Tech.WRITING:           [Tech.POTTERY],
    Tech.ASTRONOMY:         [Tech.AGRICULTURE],
    Tech.MEDICINE:          [Tech.FIRE, Tech.AGRICULTURE],
    Tech.GOVERNANCE:        [Tech.WRITING],
    Tech.MATHEMATICS:       [Tech.WRITING, Tech.ASTRONOMY],
    Tech.ENGINEERING:       [Tech.MATHEMATICS, Tech.MASONRY],
    Tech.LAW:               [Tech.GOVERNANCE, Tech.WRITING],
    Tech.PHILOSOPHY:        [Tech.MATHEMATICS, Tech.LAW],
}

# Which epoch each tech belongs to
TECH_EPOCH: dict[Tech, Epoch] = {
    Tech.FIRE:              Epoch.NOMADIC,
    Tech.STONE_TOOLS:       Epoch.NOMADIC,
    Tech.POTTERY:           Epoch.HUNTER_GATHERER,
    Tech.WEAVING:           Epoch.HUNTER_GATHERER,
    Tech.AGRICULTURE:       Epoch.AGRICULTURAL,
    Tech.IRRIGATION:        Epoch.AGRICULTURAL,
    Tech.ANIMAL_HUSBANDRY:  Epoch.AGRICULTURAL,
    Tech.ARCHITECTURE:      Epoch.SETTLEMENT,
    Tech.MASONRY:           Epoch.SETTLEMENT,
    Tech.FOOD_PRESERVATION: Epoch.SETTLEMENT,
    Tech.METALLURGY:        Epoch.VILLAGE,
    Tech.WHEEL:             Epoch.VILLAGE,
    Tech.TRADE_ROUTES:      Epoch.VILLAGE,
    Tech.WRITING:           Epoch.TOWN,
    Tech.ASTRONOMY:         Epoch.TOWN,
    Tech.MEDICINE:          Epoch.TOWN,
    Tech.GOVERNANCE:        Epoch.TOWN,
    Tech.MATHEMATICS:       Epoch.CITY_STATE,
    Tech.ENGINEERING:       Epoch.CITY_STATE,
    Tech.LAW:               Epoch.CITY_STATE,
    Tech.PHILOSOPHY:        Epoch.CITY_STATE,
}

# ── Discovery requirements ──────────────────────────────────
# Each tech requires cumulative action XP **plus** optional context flags.
# Keys:   study  / craft / build / social / trade  → cumulative count threshold
#         near_berry / near_river / near_wildlife / near_ruin / near_fungus
#         night / population / consciousness

TECH_DISCOVERY: dict[Tech, dict] = {
    Tech.FIRE:              {"study": 15},
    Tech.STONE_TOOLS:       {"craft": 8},
    Tech.POTTERY:           {"study": 25, "craft": 5},
    Tech.WEAVING:           {"craft": 15},
    Tech.AGRICULTURE:       {"study": 40, "near_berry": True},
    Tech.IRRIGATION:        {"study": 25, "near_river": True},
    Tech.ANIMAL_HUSBANDRY:  {"study": 25, "near_wildlife": True},
    Tech.ARCHITECTURE:      {"build": 15},
    Tech.MASONRY:           {"build": 25, "near_ruin": True},
    Tech.FOOD_PRESERVATION: {"study": 30, "craft": 10},
    Tech.METALLURGY:        {"study": 40, "near_ruin": True},
    Tech.WHEEL:             {"craft": 20},
    Tech.TRADE_ROUTES:      {"trade": 10},
    Tech.WRITING:           {"study": 50, "near_ruin": True},
    Tech.ASTRONOMY:         {"study": 40, "night": True},
    Tech.MEDICINE:          {"study": 35, "near_fungus": True},
    Tech.GOVERNANCE:        {"social": 40, "population": 8},
    Tech.MATHEMATICS:       {"study": 55, "near_ruin": True},
    Tech.ENGINEERING:       {"build": 35, "near_ruin": True},
    Tech.LAW:               {"social": 50, "population": 12},
    Tech.PHILOSOPHY:        {"study": 70, "consciousness": 0.65},
}

# ═══════════════════════════════════════════════════════════════
#  Building Types
# ═══════════════════════════════════════════════════════════════

class BuildingType(IntEnum):
    SHELTER   = 0
    FARM      = 1
    GRANARY   = 2
    WORKSHOP  = 3
    WALL      = 4
    MONUMENT  = 5
    LIBRARY   = 6

BUILDING_NAMES = [
    "Shelter", "Farm", "Granary", "Workshop",
    "Wall", "Monument", "Library",
]

BUILDING_COSTS: dict[BuildingType, float] = {
    BuildingType.SHELTER:   5.0,
    BuildingType.FARM:      8.0,
    BuildingType.GRANARY:  12.0,
    BuildingType.WORKSHOP: 15.0,
    BuildingType.WALL:     10.0,
    BuildingType.MONUMENT: 25.0,
    BuildingType.LIBRARY:  20.0,
}

# Which tech is needed to unlock each building (None = always available)
BUILDING_TECH: dict[BuildingType, Tech | None] = {
    BuildingType.SHELTER:   None,
    BuildingType.FARM:      Tech.AGRICULTURE,
    BuildingType.GRANARY:   Tech.FOOD_PRESERVATION,
    BuildingType.WORKSHOP:  Tech.METALLURGY,
    BuildingType.WALL:      Tech.MASONRY,
    BuildingType.MONUMENT:  Tech.ENGINEERING,
    BuildingType.LIBRARY:   Tech.WRITING,
}

# Which buildings provide shelter effects
SHELTER_BUILDINGS = {BuildingType.SHELTER, BuildingType.GRANARY,
                     BuildingType.WORKSHOP, BuildingType.LIBRARY}


# ═══════════════════════════════════════════════════════════════
#  Structures & Crops
# ═══════════════════════════════════════════════════════════════

@dataclass
class Structure:
    """A built structure in the world."""
    building_type: BuildingType
    builder_id: int
    built_tick: int = 0


@dataclass
class Crop:
    """A planted crop that grows over time and can be harvested."""
    x: int
    y: int
    growth: float = 0.0       # 0.0 → 1.0
    planted_tick: int = 0
    tended_count: int = 0
    harvested: bool = False

    GROWTH_TICKS: int = 600        # base ticks from planted → harvestable
    HARVEST_ENERGY: float = 20.0   # energy gained on harvest
    TEND_SPEEDUP: float = 1.4      # growth multiplier per tend (diminishing)

    @property
    def is_mature(self) -> bool:
        return self.growth >= 1.0 and not self.harvested


# ═══════════════════════════════════════════════════════════════
#  Specialisation
# ═══════════════════════════════════════════════════════════════

class Role(IntEnum):
    GENERALIST = 0
    FARMER     = 1
    BUILDER    = 2
    CRAFTER    = 3
    SCHOLAR    = 4
    SOCIALITE  = 5
    HUNTER     = 6

ROLE_NAMES = ["Generalist", "Farmer", "Builder", "Crafter",
              "Scholar", "Socialite", "Hunter"]

_XP_FIELDS = ["farming", "building", "crafting", "scholarly",
              "social", "hunting"]


@dataclass
class SpecProfile:
    """Tracks per-agent action XP and derives specialisation."""
    farming_xp:   float = 0.0
    building_xp:  float = 0.0
    crafting_xp:  float = 0.0
    scholarly_xp: float = 0.0
    social_xp:    float = 0.0
    hunting_xp:   float = 0.0

    # cumulative counters for tech discovery
    study_count:  int = 0
    craft_count:  int = 0
    build_count:  int = 0
    social_count: int = 0
    trade_count:  int = 0
    plant_count:  int = 0
    teach_count:  int = 0

    @property
    def primary_role(self) -> Role:
        xps = [
            (self.farming_xp,   Role.FARMER),
            (self.building_xp,  Role.BUILDER),
            (self.crafting_xp,  Role.CRAFTER),
            (self.scholarly_xp, Role.SCHOLAR),
            (self.social_xp,    Role.SOCIALITE),
            (self.hunting_xp,   Role.HUNTER),
        ]
        best_val, best_role = max(xps, key=lambda x: x[0])
        return best_role if best_val >= 15.0 else Role.GENERALIST

    def get_bonus(self, role: Role) -> float:
        """Specialisation bonus 0.0–0.5 for a given role."""
        xp_map = {
            Role.FARMER:    self.farming_xp,
            Role.BUILDER:   self.building_xp,
            Role.CRAFTER:   self.crafting_xp,
            Role.SCHOLAR:   self.scholarly_xp,
            Role.SOCIALITE: self.social_xp,
            Role.HUNTER:    self.hunting_xp,
        }
        xp = xp_map.get(role, 0.0)
        return min(0.5, math.sqrt(xp / 100.0) * 0.5)

    def record_action(self, action: int) -> None:
        """Call every tick with the agent's chosen action."""
        from genesis.agent.body import (
            ACTION_COLLECT, ACTION_BUILD, ACTION_CRAFT, ACTION_STUDY,
            ACTION_EXAMINE, ACTION_SHARE, ACTION_EMIT_SOUND, ACTION_PLANT,
            ACTION_TEACH,
        )
        if action == ACTION_PLANT:
            self.farming_xp += 1.0
            self.plant_count += 1
        elif action == ACTION_COLLECT:
            self.hunting_xp += 0.5
            self.farming_xp += 0.3
        elif action == ACTION_BUILD:
            self.building_xp += 1.0
            self.build_count += 1
        elif action == ACTION_CRAFT:
            self.crafting_xp += 1.0
            self.craft_count += 1
        elif action in (ACTION_STUDY, ACTION_EXAMINE):
            self.scholarly_xp += 1.0
            self.study_count += 1
        elif action in (ACTION_SHARE, ACTION_EMIT_SOUND):
            self.social_xp += 0.5
            self.social_count += 1
        elif action == ACTION_TEACH:
            self.social_xp += 1.0
            self.scholarly_xp += 0.5
            self.teach_count += 1


# ═══════════════════════════════════════════════════════════════
#  Civilisation State  (shared singleton per simulation)
# ═══════════════════════════════════════════════════════════════

class CivilizationState:
    """Global civilisation tracker — one instance shared by all agents."""

    def __init__(self) -> None:
        self.epoch: Epoch = Epoch.NOMADIC
        self.discovered_techs: set[Tech] = set()
        self.epoch_transitions: list[tuple[int, Epoch]] = []  # (tick, epoch)

        # Per-agent specialisation profiles  (agent_id → SpecProfile)
        self.profiles: dict[int, SpecProfile] = {}

        # Structures placed in the world  (grid_pos → Structure)
        self.structures: dict[tuple[int, int], Structure] = {}

        # Crops growing in the world
        self.crops: list[Crop] = []
        self._crop_grid: dict[tuple[int, int], Crop] = {}

        # Population tracking
        self.births: int = 0
        self._pop_cooldown: int = 0   # ticks until next birth allowed

    # ── tech API ─────────────────────────────────────────────

    def has_tech(self, tech: Tech) -> bool:
        return tech in self.discovered_techs

    def can_discover(self, tech: Tech) -> bool:
        if tech in self.discovered_techs:
            return False
        return all(p in self.discovered_techs
                   for p in TECH_PREREQUISITES[tech])

    def try_discover(self, agent_id: int, tick: int,
                     *, population: int = 4,
                     is_night: bool = False,
                     near_berry: bool = False,
                     near_river: bool = False,
                     near_wildlife: bool = False,
                     near_ruin: bool = False,
                     near_fungus: bool = False,
                     consciousness: float = 0.0) -> Tech | None:
        """Check if the agent's accumulated XP unlocks a new technology.

        Returns the newly discovered Tech, or None.
        """
        profile = self.profiles.get(agent_id)
        if profile is None:
            return None

        for tech in Tech:
            if not self.can_discover(tech):
                continue
            req = TECH_DISCOVERY[tech]

            # Action-count thresholds
            if profile.study_count < req.get("study", 0):
                continue
            if profile.craft_count < req.get("craft", 0):
                continue
            if profile.build_count < req.get("build", 0):
                continue
            if profile.social_count < req.get("social", 0):
                continue
            if profile.trade_count < req.get("trade", 0):
                continue

            # Context flags
            if req.get("near_berry") and not near_berry:
                continue
            if req.get("near_river") and not near_river:
                continue
            if req.get("near_wildlife") and not near_wildlife:
                continue
            if req.get("near_ruin") and not near_ruin:
                continue
            if req.get("near_fungus") and not near_fungus:
                continue
            if req.get("night") and not is_night:
                continue
            if population < req.get("population", 0):
                continue
            if consciousness < req.get("consciousness", 0.0):
                continue

            # All requirements met — discover!
            self.discovered_techs.add(tech)
            self._check_epoch(tick)
            return tech

        return None

    def _check_epoch(self, tick: int) -> None:
        """Re-evaluate civilisation epoch based on discovered techs."""
        for target in range(int(self.epoch) + 1, len(Epoch)):
            tier_techs = [t for t, e in TECH_EPOCH.items() if e <= Epoch(target)]
            discovered = sum(1 for t in tier_techs if t in self.discovered_techs)
            needed = max(1, len(tier_techs) // 2)
            if discovered >= needed:
                self.epoch = Epoch(target)
                self.epoch_transitions.append((tick, self.epoch))

    # ── building API ─────────────────────────────────────────

    def available_buildings(self) -> list[BuildingType]:
        """Return building types unlocked by current techs."""
        out: list[BuildingType] = []
        for bt in BuildingType:
            req = BUILDING_TECH[bt]
            if req is None or req in self.discovered_techs:
                out.append(bt)
        return out

    def best_building_for(self, gx: int, gy: int,
                          has_shelter: bool) -> BuildingType:
        """Choose the most useful building to place at (gx, gy)."""
        avail = self.available_buildings()
        # Priority: farm > granary > workshop > wall > monument > library > shelter
        priority = [
            BuildingType.FARM,
            BuildingType.GRANARY,
            BuildingType.WORKSHOP,
            BuildingType.WALL,
            BuildingType.MONUMENT,
            BuildingType.LIBRARY,
            BuildingType.SHELTER,
        ]
        existing = self.structures.get((gx, gy))
        if existing is not None:
            return existing.building_type  # already built here

        # Don't duplicate nearby same-type buildings (within 5 cells)
        nearby_types = set()
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                s = self.structures.get((gx + dx, gy + dy))
                if s is not None:
                    nearby_types.add(s.building_type)

        for bt in priority:
            if bt in avail and bt not in nearby_types:
                return bt
        # Fallback to shelter if nothing else available
        return BuildingType.SHELTER if BuildingType.SHELTER in avail else avail[0]

    def place_structure(self, gx: int, gy: int, bt: BuildingType,
                        builder_id: int, tick: int) -> Structure:
        s = Structure(building_type=bt, builder_id=builder_id, built_tick=tick)
        self.structures[(gx, gy)] = s
        return s

    # ── crop API ─────────────────────────────────────────────

    def plant_crop(self, gx: int, gy: int, tick: int) -> Crop | None:
        """Plant a new crop at (gx, gy). Returns None if already occupied."""
        if (gx, gy) in self._crop_grid:
            c = self._crop_grid[(gx, gy)]
            if not c.harvested:
                return None
        crop = Crop(x=gx, y=gy, planted_tick=tick)
        self.crops.append(crop)
        self._crop_grid[(gx, gy)] = crop
        return crop

    def tend_crop_at(self, gx: int, gy: int) -> bool:
        """Tend a crop nearby (boost its growth). Returns True if tended."""
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                c = self._crop_grid.get((gx + dx, gy + dy))
                if c and not c.harvested and c.growth < 1.0:
                    c.tended_count += 1
                    return True
        return False

    def harvest_crop_at(self, gx: int, gy: int) -> float:
        """Harvest a mature crop near (gx, gy). Returns energy gained."""
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                c = self._crop_grid.get((gx + dx, gy + dy))
                if c and c.is_mature:
                    c.harvested = True
                    bonus = 1.0 + min(0.5, c.tended_count * 0.1)
                    return c.HARVEST_ENERGY * bonus
        return 0.0

    def tick_crops(self, tick: int, season: str,
                   river_cells: set[tuple[int, int]]) -> None:
        """Advance crop growth.  Called once per simulation tick."""
        season_mult = {"spring": 1.4, "summer": 1.2,
                       "autumn": 0.7, "winter": 0.3}.get(season, 1.0)

        alive: list[Crop] = []
        for crop in self.crops:
            if crop.harvested:
                # Remove harvested crops after a cooldown
                if tick - crop.planted_tick > crop.GROWTH_TICKS + 200:
                    self._crop_grid.pop((crop.x, crop.y), None)
                    continue
                alive.append(crop)
                continue
            # Base growth per tick
            rate = 1.0 / crop.GROWTH_TICKS
            # Tend bonus (diminishing)
            tend_mult = 1.0 + min(1.0, crop.tended_count * 0.15)
            # Irrigation: near river
            irr_mult = 1.4 if (crop.x, crop.y) in river_cells else 1.0
            # Farm building bonus
            farm_mult = 1.5 if self._is_on_farm(crop.x, crop.y) else 1.0
            # Irrigation tech bonus
            if self.has_tech(Tech.IRRIGATION):
                irr_mult += 0.3

            crop.growth = min(1.0,
                              crop.growth + rate * season_mult * tend_mult
                              * irr_mult * farm_mult)
            alive.append(crop)
        self.crops = alive

    def _is_on_farm(self, gx: int, gy: int) -> bool:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                s = self.structures.get((gx + dx, gy + dy))
                if s and s.building_type == BuildingType.FARM:
                    return True
        return False

    # ── population API ───────────────────────────────────────

    def get_pop_cap(self) -> int:
        base = EPOCH_POP_CAP[self.epoch]
        # Granaries increase cap by 2 each
        granaries = sum(1 for s in self.structures.values()
                        if s.building_type == BuildingType.GRANARY)
        return base + granaries * 2

    def should_spawn(self, current_pop: int, avg_energy_ratio: float,
                     num_shelters: int, tick: int) -> bool:
        """Return True if conditions are met for population growth."""
        if self._pop_cooldown > 0:
            self._pop_cooldown -= 1
            return False
        cap = self.get_pop_cap()
        if current_pop >= cap:
            return False
        # Need food surplus (avg energy > 60%) and shelter > pop/2
        if avg_energy_ratio < 0.55:
            return False
        if num_shelters < current_pop // 2:
            return False
        # Agricultural epoch minimum for natural growth
        if self.epoch < Epoch.AGRICULTURAL:
            return False
        self._pop_cooldown = max(200, 800 - current_pop * 30)
        self.births += 1
        return True

    # ── summaries ────────────────────────────────────────────

    def get_summary(self) -> dict:
        return {
            "epoch": EPOCH_NAMES[self.epoch],
            "epoch_id": int(self.epoch),
            "techs_discovered": len(self.discovered_techs),
            "tech_names": [TECH_NAMES[t] for t in sorted(self.discovered_techs)],
            "structures": len(self.structures),
            "crops": sum(1 for c in self.crops if not c.harvested),
            "mature_crops": sum(1 for c in self.crops if c.is_mature),
            "pop_cap": self.get_pop_cap(),
            "births": self.births,
        }
