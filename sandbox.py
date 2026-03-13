"""The Sandbox — a continuous, non-episodic virtual world with scarcity and entropy."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from genesis.config import WorldConfig
from genesis.environment.physics import Vec2, check_collision
from genesis.environment.predators import PredatorSystem
from genesis.environment.resources import (
    EnergyCrystal, HeightMap, BiomeMap, BiomeProperties,
    RESOURCE_NORMAL, RESOURCE_RARE, RESOURCE_TOXIC,
    BIOME_OCEAN, BIOME_WETLANDS, BIOME_GRASSLANDS, BIOME_DESERT, BIOME_MOUNTAINS,
    BIOME_NAMES, BIOME_PROPS,
)
from genesis.environment.weather import WeatherSystem

import math


@dataclass
class DayCycle:
    """Day/night cycle affecting visibility and resource spawns."""
    cycle_length: int  # ticks per full cycle
    current_tick: int = 0

    @property
    def phase(self) -> float:
        """0.0–1.0 through the day cycle."""
        return self.current_tick / self.cycle_length

    @property
    def is_night(self) -> bool:
        return self.phase > 0.5

    @property
    def light_level(self) -> float:
        """1.0 = noon, 0.2 = midnight."""
        import math
        return 0.6 + 0.4 * math.cos(2 * math.pi * self.phase)

    @property
    def is_raining(self) -> bool:
        """Rain occurs during specific phase windows."""
        p = self.phase
        return 0.3 < p < 0.4 or 0.7 < p < 0.8

    def tick(self) -> None:
        self.current_tick = (self.current_tick + 1) % self.cycle_length


# ── New world features ──────────────────────────────────────

@dataclass
class BerryBush:
    """Renewable food source that regrows berries over time."""
    position: Vec2
    berries: int = 3          # current harvestable berries
    max_berries: int = 3
    regrow_cooldown: int = 0  # ticks until next berry regrows
    regrow_rate: int = 300    # ticks per berry

    def tick(self) -> None:
        if self.berries < self.max_berries:
            self.regrow_cooldown -= 1
            if self.regrow_cooldown <= 0:
                self.berries += 1
                self.regrow_cooldown = self.regrow_rate

    def harvest(self) -> float:
        """Pick a berry. Returns energy gained (0 if empty)."""
        if self.berries > 0:
            self.berries -= 1
            self.regrow_cooldown = self.regrow_rate
            return 12.0
        return 0.0


@dataclass
class GlowingFungus:
    """Bioluminescent fungi in wetlands — glow at night, small energy source."""
    position: Vec2
    energy: float = 8.0
    consumed: bool = False
    glow_phase: float = 0.0  # animation phase

    def tick(self) -> None:
        self.glow_phase += 0.02


@dataclass
class AncientRuin:
    """Mysterious structures — provide shelter + curiosity/knowledge reward."""
    position: Vec2
    radius: float = 2.5
    knowledge_given: set = field(default_factory=set)  # agent_ids that already studied here

    def can_study(self, agent_id: int) -> bool:
        return agent_id not in self.knowledge_given

    def study(self, agent_id: int) -> float:
        """Study the ruin. Returns knowledge/pleasure reward."""
        if agent_id not in self.knowledge_given:
            self.knowledge_given.add(agent_id)
            return 0.6  # strong novelty signal
        return 0.05     # diminishing returns


@dataclass
class Wildlife:
    """A passive wildlife creature (fish or bird) that moves in the world."""
    position: Vec2
    kind: str          # "fish" or "bird"
    vx: float = 0.0
    vy: float = 0.0
    alive: bool = True


class Sandbox:
    """The virtual world — continuous time, procedural terrain, scarcity, and entropy."""

    def __init__(self, config: WorldConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.width = config.width
        self.height = config.height
        self.tick_count = 0

        # Terrain
        self.heightmap = HeightMap(self.width, self.height)
        self.heightmap.generate(self.rng)

        # Biomes (derived from elevation + moisture)
        self.biome_map = BiomeMap(self.width, self.height)
        self.biome_map.generate(self.heightmap, self.rng)

        # Day/night (must be initialized before resource spawning)
        self.day_cycle = DayCycle(cycle_length=config.day_cycle_ticks)

        # Weather system (seasons + weather events) — before resource spawning
        self.weather = WeatherSystem()

        # Predator system (mobile threats)
        self.predators = PredatorSystem(width=self.width,
                                         height=self.height)
        self.predator_contacts: list[tuple[int, float, float]] = []

        # Obstacles (walls, boulders)
        self.obstacles: set[tuple[int, int]] = set()
        self._generate_obstacles()

        # Resources
        self.crystals: list[EnergyCrystal] = []
        self._spawn_initial_resources()

        # Spatial index for crystals: grid_pos → list of crystals
        self._crystal_grid: dict[tuple[int, int], list[EnergyCrystal]] = {}
        self._rebuild_crystal_grid()

        # Audio events in the world this tick (position, intensity, source_id)
        self.audio_events: list[tuple[Vec2, float, int]] = []

        # Dynamic environment state
        self._prev_is_night: bool = False
        self._prev_is_raining: bool = False
        self.environment_change: float = 0.0  # signal for curiosity engines
        self._resource_pulse_cooldown: int = 0
        self._hazard_zones: list[tuple[Vec2, float, int]] = []  # (center, radius, remaining_ticks)
        self._spawn_positions: list[Vec2] = []  # track agent spawns for spacing

        # Shelters built by agents: (x, y) -> builder_agent_id
        self.shelters: dict[tuple[int, int], int] = {}

        # ── New world features ──
        self.berry_bushes: list[BerryBush] = []
        self.fungi: list[GlowingFungus] = []
        self.ruins: list[AncientRuin] = []
        self.rivers: set[tuple[int, int]] = set()   # cells that are river
        self._generate_features()

        # Spatial index for berry bushes: grid_pos → BerryBush
        self._berry_grid: dict[tuple[int, int], BerryBush] = {}
        for bush in self.berry_bushes:
            k = bush.position.grid_pos()
            self._berry_grid[k] = bush
        # Spatial index for fungi: grid_pos → GlowingFungus
        self._fungi_grid: dict[tuple[int, int], GlowingFungus] = {}
        for fungus in self.fungi:
            k = fungus.position.grid_pos()
            self._fungi_grid[k] = fungus
        # Spatial index for ruins: grid_pos → AncientRuin (radius area)
        self._ruin_grid: dict[tuple[int, int], AncientRuin] = {}
        for ruin in self.ruins:
            rx, ry = int(ruin.position.x), int(ruin.position.y)
            r = int(ruin.radius) + 1
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    self._ruin_grid[(rx + dx, ry + dy)] = ruin

    def _generate_features(self) -> None:
        """Generate berry bushes, fungi, ruins, and rivers."""
        w, h = self.width, self.height

        # Rivers — sinusoidal paths from top to bottom
        num_rivers = self.rng.randint(2, 4)
        for r in range(num_rivers):
            rx = self.rng.randint(int(w * 0.15), int(w * 0.85))
            amplitude = self.rng.uniform(8, 20)
            freq = self.rng.uniform(0.02, 0.06)
            phase = self.rng.uniform(0, 2 * math.pi)
            for ry in range(h):
                cx = int(rx + amplitude * math.sin(freq * ry + phase))
                for dx in range(-1, 2):  # 3-wide river
                    nx = cx + dx
                    if 0 < nx < w - 1:
                        self.rivers.add((nx, ry))
                        self.obstacles.discard((nx, ry))

        # Berry bushes — scattered in grasslands and wetlands
        num_bushes = w * h // 400  # ~1 per 400 cells
        for _ in range(num_bushes):
            x = self.rng.randint(2, w - 3)
            y = self.rng.randint(2, h - 3)
            if (x, y) in self.obstacles or (x, y) in self.rivers:
                continue
            biome = self.biome_map.biome_at(x, y)
            if biome in (BIOME_GRASSLANDS, BIOME_WETLANDS):
                self.berry_bushes.append(BerryBush(
                    position=Vec2(float(x), float(y)),
                    max_berries=self.rng.randint(2, 5),
                    berries=self.rng.randint(1, 3),
                    regrow_rate=self.rng.randint(200, 400),
                ))

        # Glowing fungi — clustered in wetlands
        num_fungi = w * h // 600
        for _ in range(num_fungi):
            x = self.rng.randint(2, w - 3)
            y = self.rng.randint(2, h - 3)
            if (x, y) in self.obstacles:
                continue
            biome = self.biome_map.biome_at(x, y)
            if biome == BIOME_WETLANDS:
                self.fungi.append(GlowingFungus(
                    position=Vec2(float(x), float(y)),
                    energy=self.rng.uniform(5.0, 12.0),
                ))

        # Ancient ruins — rare, in desert and mountains
        num_ruins = max(3, w * h // 8000)
        for _ in range(num_ruins):
            for _attempt in range(50):
                x = self.rng.randint(5, w - 6)
                y = self.rng.randint(5, h - 6)
                if (x, y) in self.obstacles:
                    continue
                biome = self.biome_map.biome_at(x, y)
                if biome in (BIOME_DESERT, BIOME_MOUNTAINS, BIOME_GRASSLANDS):
                    # Don't place too close to another ruin
                    pos = Vec2(float(x), float(y))
                    too_close = any(
                        pos.distance_to(r.position) < 25 for r in self.ruins
                    )
                    if not too_close:
                        self.ruins.append(AncientRuin(position=pos))
                        break

        # Passive wildlife — fish in rivers, birds in forests/grasslands
        self.wildlife: list[Wildlife] = []
        river_cells = list(self.rivers)
        num_fish = min(len(river_cells) // 20, 50)
        for _ in range(num_fish):
            cell = self.rng.choice(river_cells)
            self.wildlife.append(Wildlife(
                position=Vec2(float(cell[0]), float(cell[1])),
                kind="fish",
                vx=self.rng.uniform(-0.3, 0.3),
                vy=self.rng.uniform(-0.1, 0.1),
            ))
        num_birds = w * h // 3000
        for _ in range(num_birds):
            x = self.rng.randint(2, w - 3)
            y = self.rng.randint(2, h - 3)
            biome = self.biome_map.biome_at(x, y)
            if biome in (BIOME_GRASSLANDS, BIOME_WETLANDS):
                self.wildlife.append(Wildlife(
                    position=Vec2(float(x), float(y)),
                    kind="bird",
                    vx=self.rng.uniform(-0.5, 0.5),
                    vy=self.rng.uniform(-0.5, 0.5),
                ))

    def _generate_obstacles(self) -> None:
        """Place obstacles on the map based on density config."""
        for y in range(self.height):
            for x in range(self.width):
                # Higher terrain = more likely to be rocky/blocked
                elev = self.heightmap.elevation_at(x, y)
                threshold = self.config.obstacle_density * (1 + max(0, elev) * 0.3)
                if self.rng.random() < threshold:
                    self.obstacles.add((x, y))
        # Clear a border strip to guarantee agent can spawn
        for x in range(self.width):
            self.obstacles.discard((x, 0))
            self.obstacles.discard((x, self.height - 1))
        for y in range(self.height):
            self.obstacles.discard((0, y))
            self.obstacles.discard((self.width - 1, y))

    def _rebuild_crystal_grid(self) -> None:
        """Rebuild spatial hash for crystals (call after crystal list changes)."""
        self._crystal_grid.clear()
        for c in self.crystals:
            if not c.consumed and not c.is_expired:
                key = c.position.grid_pos()
                if key not in self._crystal_grid:
                    self._crystal_grid[key] = []
                self._crystal_grid[key].append(c)

    def _spawn_initial_resources(self) -> None:
        for _ in range(self.config.max_resources // 2):
            self._try_spawn_crystal()

    def _try_spawn_crystal(self) -> bool:
        """Attempt to spawn a crystal at a valid location, biome-aware."""
        weather_mods = self.weather.get_modifiers()
        for _ in range(20):  # retry limit
            x = self.rng.randint(1, self.width - 2)
            y = self.rng.randint(1, self.height - 2)
            if (x, y) in self.obstacles:
                continue
            bp = self.biome_map.props_at(x, y)
            # Biome + weather modulate spawn chance
            bonus = bp.resource_spawn_mult * weather_mods.resource_spawn_mult
            if self.heightmap.is_valley(x, y):
                bonus *= 1.5
            if self.day_cycle.is_raining:
                bonus *= 2.0
            if self.rng.random() < 0.5 * bonus:
                # Determine resource type (biome shifts ratios)
                toxic_r = max(0.0, self.config.toxic_crystal_ratio + bp.toxic_ratio_bonus)
                rare_r = max(0.0, self.config.rare_crystal_ratio + bp.rare_ratio_bonus)
                roll = self.rng.random()
                if roll < toxic_r:
                    crystal = EnergyCrystal(
                        position=Vec2(float(x), float(y)),
                        energy_value=-self.config.toxic_crystal_damage,
                        resource_type=RESOURCE_TOXIC,
                    )
                elif roll < toxic_r + rare_r:
                    crystal = EnergyCrystal(
                        position=Vec2(float(x), float(y)),
                        energy_value=self.config.rare_crystal_energy * bp.crystal_energy_mult * weather_mods.crystal_energy_mult,
                        max_age=400,
                        resource_type=RESOURCE_RARE,
                    )
                else:
                    crystal = EnergyCrystal(
                        position=Vec2(float(x), float(y)),
                        energy_value=25.0 * bp.crystal_energy_mult * weather_mods.crystal_energy_mult,
                        resource_type=RESOURCE_NORMAL,
                    )
                self.crystals.append(crystal)
                return True
        return False

    def get_spawn_position(self) -> Vec2:
        """Find a valid spawn position for an agent, spaced from existing spawns."""
        min_agent_dist = 20  # minimum distance between agents
        best_pos = None
        best_score = -1
        for _ in range(200):
            x = self.rng.randint(4, self.width - 5)
            y = self.rng.randint(4, self.height - 5)
            if (x, y) in self.obstacles:
                continue
            # Enforce distance from previously spawned agents
            pos = Vec2(float(x), float(y))
            too_close = any(
                pos.distance_to(prev) < min_agent_dist
                for prev in self._spawn_positions
            )
            if too_close:
                continue
            # Score by proximity to existing crystals
            nearby = sum(
                1 for c in self.crystals
                if not c.consumed and abs(c.position.x - x) + abs(c.position.y - y) < 15
            )
            if nearby > best_score:
                best_score = nearby
                best_pos = pos
        if best_pos is None:
            best_pos = Vec2(float(self.width // 2), float(self.height // 2))
        self._spawn_positions.append(best_pos)
        return best_pos

    def check_position(self, pos: Vec2) -> tuple[Vec2, bool]:
        """Check if a position is valid, returns corrected pos and collision flag."""
        return check_collision(pos, self.obstacles, self.width, self.height)

    def collect_crystal_at(self, pos: Vec2, collect_radius: float = 1.5) -> float:
        """Try to collect a crystal near the given position. Returns energy gained."""
        for crystal in self.crystals:
            if crystal.consumed or crystal.is_expired:
                continue
            if pos.distance_to(crystal.position) <= collect_radius:
                energy = crystal.effective_energy
                crystal.consumed = True
                return energy
        # Try berry bushes
        for bush in self.berry_bushes:
            if pos.distance_to(bush.position) <= collect_radius and bush.berries > 0:
                return bush.harvest()
        # Try fungi
        for fungus in self.fungi:
            if not fungus.consumed and pos.distance_to(fungus.position) <= collect_radius:
                fungus.consumed = True
                return fungus.energy
        return 0.0

    def study_ruin_at(self, pos: Vec2, agent_id: int) -> float:
        """Try to study a nearby ruin. Returns pleasure/knowledge reward."""
        for ruin in self.ruins:
            if pos.distance_to(ruin.position) <= ruin.radius:
                return ruin.study(agent_id)
        return 0.0

    def is_river(self, x: int, y: int) -> bool:
        """Check if a cell is part of a river."""
        return (x, y) in self.rivers

    def is_near_ruin(self, pos: Vec2) -> bool:
        """Check if position is near an ancient ruin (provides shelter)."""
        for ruin in self.ruins:
            if pos.distance_to(ruin.position) <= ruin.radius:
                return True
        return False

    def get_visible_cells(self, pos: Vec2, vision_range: int) -> list[dict]:
        """Return data about cells visible from a position (ray-cast approximation)."""
        visible = []
        gx, gy = pos.grid_pos()
        weather_mods = self.weather.get_modifiers()
        effective_range = vision_range * self.day_cycle.light_level * weather_mods.vision_mult
        eff_range_sq = effective_range * effective_range
        is_night = self.day_cycle.is_night
        vr_sq = vision_range * vision_range
        obstacles = self.obstacles
        shelters = self.shelters
        rivers = self.rivers
        crystal_grid = self._crystal_grid
        berry_grid = self._berry_grid
        fungi_grid = self._fungi_grid
        ruin_grid = self._ruin_grid
        hm = self.heightmap
        bm = self.biome_map
        w, h = self.width, self.height
        # Pre-check hazard zones once
        has_hazards = len(self._hazard_zones) > 0
        for dy in range(-vision_range, vision_range + 1):
            ny = gy + dy
            if ny < 0 or ny >= h:
                continue
            for dx in range(-vision_range, vision_range + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq > vr_sq:
                    continue
                if dist_sq > eff_range_sq:
                    continue
                nx = gx + dx
                if nx < 0 or nx >= w:
                    continue
                dist = dist_sq ** 0.5
                key = (nx, ny)
                # Crystal lookup via spatial hash — O(1)
                has_crystal = False
                crystal_energy = 0.0
                crystal_type = 0
                cl = crystal_grid.get(key)
                if cl:
                    for c in cl:
                        if not c.consumed and not c.is_expired:
                            has_crystal = True
                            crystal_energy = c.effective_energy
                            crystal_type = c.resource_type
                            break
                # Berry / fungi / ruin lookup via spatial hash — O(1)
                bush = berry_grid.get(key)
                has_berry = bush is not None and bush.berries > 0
                fung = fungi_grid.get(key)
                has_fungus = fung is not None and not fung.consumed
                has_ruin = key in ruin_grid
                cell = {
                    "x": nx, "y": ny,
                    "distance": dist,
                    "elevation": hm.elevation_at(nx, ny),
                    "is_obstacle": key in obstacles,
                    "has_crystal": has_crystal,
                    "crystal_energy": crystal_energy,
                    "crystal_type": crystal_type,
                    "hazard": self.get_hazard_at(Vec2(float(nx), float(ny))) if has_hazards else 0.0,
                    "is_night": is_night,
                    "biome": bm.biome_at(nx, ny),
                    "has_shelter": key in shelters,
                    "is_river": key in rivers,
                    "has_berry": has_berry,
                    "has_fungus": has_fungus,
                    "has_ruin": has_ruin,
                }
                visible.append(cell)
        return visible

    def get_audible_events(self, pos: Vec2, hearing_range: int) -> list[tuple[Vec2, float, int]]:
        """Return audio events within hearing range."""
        return [
            (apos, intensity, src_id)
            for apos, intensity, src_id in self.audio_events
            if pos.distance_to(apos) <= hearing_range
        ]

    def emit_audio(self, position: Vec2, intensity: float, source_id: int) -> None:
        """Emit an audio event into the world."""
        self.audio_events.append((position, intensity, source_id))

    def tick(self, agent_positions: list | None = None) -> None:
        """Advance the world by one tick with dynamic events."""
        self.tick_count += 1
        self.day_cycle.tick()
        weather_env_change = self.weather.tick()
        self.environment_change = 0.0

        # Clear audio from last tick
        self.audio_events.clear()

        # Detect day/night and weather transitions — generate change signals
        if self.day_cycle.is_night != self._prev_is_night:
            self.environment_change = max(self.environment_change, 0.6)
            # Emit a global ambient sound at transition
            center = Vec2(float(self.width // 2), float(self.height // 2))
            self.emit_audio(center, 0.5, -1)  # environmental sound
        self._prev_is_night = self.day_cycle.is_night

        if self.day_cycle.is_raining != self._prev_is_raining:
            self.environment_change = max(self.environment_change, 0.4)
        self._prev_is_raining = self.day_cycle.is_raining

        # Weather transitions generate change signals
        if weather_env_change > 0:
            self.environment_change = max(self.environment_change, weather_env_change)

        # Weather modifiers
        weather_mods = self.weather.get_modifiers()

        # Age and cull crystals
        for crystal in self.crystals:
            crystal.tick()
        self.crystals = [c for c in self.crystals if not c.consumed and not c.is_expired]

        # Rebuild spatial index every tick (cheap — just dict insertions)
        self._rebuild_crystal_grid()

        # Spawn new crystals (weather modulates spawn rate)
        if len(self.crystals) < self.config.max_resources:
            if self.rng.random() < self.config.resource_spawn_rate * weather_mods.resource_spawn_mult:
                self._try_spawn_crystal()

        # Resource pulse: periodic crystal surges that create foraging hotspots
        if self._resource_pulse_cooldown > 0:
            self._resource_pulse_cooldown -= 1
        elif self.rng.random() < 0.003:  # ~every 333 ticks on average
            self._resource_pulse()

        # Hazard zone maintenance (weather modulates hazard spawning)
        self._tick_hazard_zones()
        if self.rng.random() < 0.002 * weather_mods.hazard_chance_mult:
            self._spawn_hazard_zone()

        # Rain spawns extra crystals in valleys
        if self.day_cycle.is_raining and self.rng.random() < 0.25:
            self._try_spawn_crystal()

        # Tick berry bushes (season modulates regrowth speed)
        season = self.weather.season
        # Spring/summer = faster regrowth, winter = slower
        berry_speed = {0: 1.5, 1: 1.2, 2: 0.8, 3: 0.4}.get(season, 1.0)
        for bush in self.berry_bushes:
            # Accelerated berry tick based on season
            if bush.berries < bush.max_berries:
                bush.regrow_cooldown -= berry_speed
                if bush.regrow_cooldown <= 0:
                    bush.berries += 1
                    bush.regrow_cooldown = bush.regrow_rate
            else:
                bush.tick()  # still runs tick for consistency

        # Tick fungi
        for fungus in self.fungi:
            fungus.tick()

        # Respawn consumed fungi occasionally
        if self.rng.random() < 0.002:
            consumed_fungi = [f for f in self.fungi if f.consumed]
            if consumed_fungi:
                f = self.rng.choice(consumed_fungi)
                f.consumed = False
                f.energy = self.rng.uniform(5.0, 12.0)

        # Tick predators
        positions = agent_positions or []
        self.predator_contacts = self.predators.tick(
            positions, self.obstacles, self.biome_map
        )

        # Tick passive wildlife
        self._tick_wildlife()

        # Tick civilisation crops
        civ = getattr(self, 'civilization', None)
        if civ:
            season_name = self.weather.get_summary()['season']
            civ.tick_crops(self.tick_count, season_name, self.rivers)

    def _resource_pulse(self) -> None:
        """Spawn a cluster of crystals in one area — creates foraging hotspots."""
        cx = self.rng.randint(10, self.width - 10)
        cy = self.rng.randint(10, self.height - 10)
        count = self.rng.randint(4, 8)
        for _ in range(count):
            ox = self.rng.randint(-5, 5)
            oy = self.rng.randint(-5, 5)
            x, y = cx + ox, cy + oy
            if 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles:
                crystal = EnergyCrystal(
                    position=Vec2(float(x), float(y)),
                    energy_value=self.rng.uniform(20, 50),
                    max_age=600,
                    resource_type=RESOURCE_NORMAL if self.rng.random() > 0.2 else RESOURCE_RARE,
                )
                self.crystals.append(crystal)
        # Emit audio beacon at pulse location to attract agents
        self.emit_audio(Vec2(float(cx), float(cy)), 1.0, -2)
        self.environment_change = max(self.environment_change, 0.7)
        self._resource_pulse_cooldown = 200

    def _spawn_hazard_zone(self) -> None:
        """Create a temporary hazard zone, more likely in mountains/desert."""
        cx = self.rng.randint(5, self.width - 5)
        cy = self.rng.randint(5, self.height - 5)
        bp = self.biome_map.props_at(cx, cy)
        # Biome modulates whether the hazard actually spawns
        if self.rng.random() > bp.hazard_chance_mult / 2.0:
            return
        radius = self.rng.uniform(4.0, 8.0)
        duration = self.rng.randint(100, 300)
        self._hazard_zones.append((Vec2(float(cx), float(cy)), radius, duration))
        self.environment_change = max(self.environment_change, 0.5)

    def _tick_hazard_zones(self) -> None:
        """Age hazard zones and remove expired ones."""
        updated = []
        for center, radius, remaining in self._hazard_zones:
            if remaining > 1:
                updated.append((center, radius, remaining - 1))
        self._hazard_zones = updated

    def _tick_wildlife(self) -> None:
        """Move passive wildlife — fish swim along rivers, birds wander."""
        w, h = self.width, self.height
        rivers = self.rivers
        for creature in self.wildlife:
            if creature.kind == "fish":
                # Fish drift along river cells with random turning
                nx = creature.position.x + creature.vx
                ny = creature.position.y + creature.vy
                gx, gy = int(nx), int(ny)
                if 0 <= gx < w and 0 <= gy < h and (gx, gy) in rivers:
                    creature.position = Vec2(nx, ny)
                else:
                    # Bounce back — reverse direction with jitter
                    creature.vx = -creature.vx + self.rng.uniform(-0.1, 0.1)
                    creature.vy = -creature.vy + self.rng.uniform(-0.1, 0.1)
                # Random turning
                if self.rng.random() < 0.05:
                    creature.vx += self.rng.uniform(-0.15, 0.15)
                    creature.vy += self.rng.uniform(-0.15, 0.15)
                    speed = max(0.01, (creature.vx**2 + creature.vy**2)**0.5)
                    creature.vx = creature.vx / speed * 0.3
                    creature.vy = creature.vy / speed * 0.3
            elif creature.kind == "bird":
                # Birds wander freely with occasional direction changes
                creature.position = Vec2(
                    max(1.0, min(w - 2.0, creature.position.x + creature.vx)),
                    max(1.0, min(h - 2.0, creature.position.y + creature.vy)),
                )
                if self.rng.random() < 0.03:
                    creature.vx = self.rng.uniform(-0.5, 0.5)
                    creature.vy = self.rng.uniform(-0.5, 0.5)

    def get_hazard_at(self, pos: Vec2) -> float:
        """Return hazard intensity at a position (0.0 = safe, 1.0 = dangerous).
        Shelters reduce hazard by 70%."""
        max_hazard = 0.0
        for center, radius, _ in self._hazard_zones:
            dist = pos.distance_to(center)
            if dist < radius:
                max_hazard = max(max_hazard, 1.0 - dist / radius)
        # Shelters provide protection
        if max_hazard > 0 and self.is_sheltered(pos):
            max_hazard *= 0.3
        return max_hazard

    def add_shelter(self, x: int, y: int, agent_id: int) -> None:
        """Place a shelter at a grid position."""
        self.shelters[(x, y)] = agent_id

    def is_sheltered(self, pos: Vec2) -> bool:
        """Check if position is at or adjacent to a shelter or near a ruin."""
        gx, gy = pos.grid_pos()
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if (gx + dx, gy + dy) in self.shelters:
                    return True
        return self.is_near_ruin(pos)

    def get_night_energy_drain(self) -> float:
        """Extra energy drain at night — creates pressure for day activity."""
        weather_mods = self.weather.get_modifiers()
        base = 0.0
        if self.day_cycle.is_night:
            base = 0.01 * (1.0 - self.day_cycle.light_level)
        # Weather energy drain multiplier
        base *= weather_mods.energy_drain_mult
        # Exposure damage from weather
        base += weather_mods.integrity_decay * 0.3
        return base

    def get_night_energy_drain_at(self, pos: Vec2) -> float:
        """Night energy drain modified by shelter (sheltered = half drain)."""
        base = self.get_night_energy_drain()
        if base > 0 and self.is_sheltered(pos):
            return base * 0.5
        return base

    def get_biome_energy_drain(self, pos: Vec2) -> float:
        """Extra energy drain based on biome (desert/ocean are harsher)."""
        gx, gy = pos.grid_pos()
        bp = self.biome_map.props_at(gx, gy)
        # Return the delta above baseline (1.0 = no extra drain)
        return max(0.0, (bp.energy_drain_mult - 1.0) * 0.01)

    def get_state_summary(self) -> dict:
        """Return a summary of the world state for analytics."""
        normal = sum(1 for c in self.crystals if c.resource_type == RESOURCE_NORMAL)
        rare = sum(1 for c in self.crystals if c.resource_type == RESOURCE_RARE)
        toxic = sum(1 for c in self.crystals if c.resource_type == RESOURCE_TOXIC)
        weather_summary = self.weather.get_summary()
        return {
            "tick": self.tick_count,
            "num_crystals": len(self.crystals),
            "normal_crystals": normal,
            "rare_crystals": rare,
            "toxic_crystals": toxic,
            "is_night": self.day_cycle.is_night,
            "is_raining": self.day_cycle.is_raining,
            "light_level": self.day_cycle.light_level,
            "day_phase": self.day_cycle.phase,
            "hazard_zones": len(self._hazard_zones),
            "environment_change": self.environment_change,
            "biome_distribution": self.biome_map.get_distribution(),
            "shelters": len(self.shelters),
            "berry_bushes": len(self.berry_bushes),
            "fungi": sum(1 for f in self.fungi if not f.consumed),
            "ruins": len(self.ruins),
            "rivers": len(self.rivers),
            "season": weather_summary["season"],
            "weather": weather_summary["weather"],
            "temperature": weather_summary["temperature"],
            "predators": len(self.predators.predators),
        }

    def get_terrain_movement_cost(self, from_pos: Vec2, to_pos: Vec2) -> float:
        """Return extra energy cost for moving between two positions (biome-aware)."""
        fx, fy = from_pos.grid_pos()
        tx, ty = to_pos.grid_pos()
        bp = self.biome_map.props_at(tx, ty)
        base = self.heightmap.movement_cost(fx, fy, tx, ty) * self.config.terrain_movement_cost
        cost = base * bp.move_cost_mult
        # Rivers slow movement
        if (tx, ty) in self.rivers:
            cost += 0.01
        return cost
