"""Energy crystals, toxic resources, biomes, and environmental features."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from genesis.environment.physics import Vec2


# Resource types
RESOURCE_NORMAL = 0
RESOURCE_RARE = 1
RESOURCE_TOXIC = 2
RESOURCE_NAMES = ["normal", "rare", "toxic"]

# ── Biome system ─────────────────────────────────────────────────
BIOME_OCEAN = 0
BIOME_WETLANDS = 1
BIOME_GRASSLANDS = 2
BIOME_DESERT = 3
BIOME_MOUNTAINS = 4
BIOME_NAMES = ["ocean", "wetlands", "grasslands", "desert", "mountains"]


@dataclass(frozen=True)
class BiomeProperties:
    """Per-biome modifiers for environment behaviour."""
    name: str
    move_cost_mult: float      # multiplier on terrain movement cost
    energy_drain_mult: float   # multiplier on passive energy drain
    resource_spawn_mult: float # multiplier on crystal spawn chance
    rare_ratio_bonus: float    # added to rare_crystal_ratio
    toxic_ratio_bonus: float   # added to toxic_crystal_ratio
    hazard_chance_mult: float  # multiplier on spontaneous hazard spawning
    crystal_energy_mult: float # multiplier on crystal energy value


BIOME_PROPS: dict[int, BiomeProperties] = {
    BIOME_OCEAN: BiomeProperties(
        name="ocean", move_cost_mult=2.0, energy_drain_mult=1.4,
        resource_spawn_mult=0.3, rare_ratio_bonus=0.0, toxic_ratio_bonus=0.0,
        hazard_chance_mult=0.5, crystal_energy_mult=0.8,
    ),
    BIOME_WETLANDS: BiomeProperties(
        name="wetlands", move_cost_mult=1.3, energy_drain_mult=0.9,
        resource_spawn_mult=1.8, rare_ratio_bonus=-0.04, toxic_ratio_bonus=0.04,
        hazard_chance_mult=0.7, crystal_energy_mult=1.0,
    ),
    BIOME_GRASSLANDS: BiomeProperties(
        name="grasslands", move_cost_mult=1.0, energy_drain_mult=1.0,
        resource_spawn_mult=1.0, rare_ratio_bonus=0.0, toxic_ratio_bonus=0.0,
        hazard_chance_mult=1.0, crystal_energy_mult=1.0,
    ),
    BIOME_DESERT: BiomeProperties(
        name="desert", move_cost_mult=1.2, energy_drain_mult=1.5,
        resource_spawn_mult=0.5, rare_ratio_bonus=0.08, toxic_ratio_bonus=-0.03,
        hazard_chance_mult=1.5, crystal_energy_mult=1.3,
    ),
    BIOME_MOUNTAINS: BiomeProperties(
        name="mountains", move_cost_mult=1.8, energy_drain_mult=1.2,
        resource_spawn_mult=0.6, rare_ratio_bonus=0.10, toxic_ratio_bonus=0.05,
        hazard_chance_mult=2.0, crystal_energy_mult=1.5,
    ),
}


@dataclass
class EnergyCrystal:
    """A collectible energy resource that decays over time."""
    position: Vec2
    energy_value: float = 25.0
    max_age: int = 800  # ticks before it spoils
    age: int = 0
    consumed: bool = False
    resource_type: int = RESOURCE_NORMAL

    @property
    def is_expired(self) -> bool:
        return self.age >= self.max_age

    @property
    def freshness(self) -> float:
        """1.0 = fresh, 0.0 = spoiled. Energy scales with freshness."""
        return max(0.0, 1.0 - self.age / self.max_age)

    @property
    def effective_energy(self) -> float:
        return self.energy_value * self.freshness

    @property
    def is_toxic(self) -> bool:
        return self.resource_type == RESOURCE_TOXIC

    @property
    def is_rare(self) -> bool:
        return self.resource_type == RESOURCE_RARE

    def tick(self) -> None:
        self.age += 1


def _perlin_fade(t: float) -> float:
    """Perlin smoothstep: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _perlin_lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


class _PerlinGradients:
    """Stores gradient vectors for Perlin noise generation."""

    def __init__(self, width: int, height: int, rng: random.Random,
                 cell_size: int = 8) -> None:
        self.cell_size = cell_size
        self.gw = width // cell_size + 2
        self.gh = height // cell_size + 2
        # Random unit-vector gradients at each grid vertex
        self.gradients: list[list[tuple[float, float]]] = []
        for _ in range(self.gh):
            row = []
            for _ in range(self.gw):
                angle = rng.uniform(0, 2 * math.pi)
                row.append((math.cos(angle), math.sin(angle)))
            self.gradients.append(row)

    def _dot_grid(self, ix: int, iy: int, x: float, y: float) -> float:
        gx, gy = self.gradients[iy % self.gh][ix % self.gw]
        dx = x - ix
        dy = y - iy
        return gx * dx + gy * dy

    def noise(self, x: float, y: float) -> float:
        cs = self.cell_size
        xf = x / cs
        yf = y / cs
        x0 = int(math.floor(xf))
        y0 = int(math.floor(yf))
        sx = xf - x0
        sy = yf - y0

        n0 = self._dot_grid(x0, y0, xf, yf)
        n1 = self._dot_grid(x0 + 1, y0, xf, yf)
        ix0 = _perlin_lerp(n0, n1, _perlin_fade(sx))

        n2 = self._dot_grid(x0, y0 + 1, xf, yf)
        n3 = self._dot_grid(x0 + 1, y0 + 1, xf, yf)
        ix1 = _perlin_lerp(n2, n3, _perlin_fade(sx))

        return _perlin_lerp(ix0, ix1, _perlin_fade(sy))


@dataclass
class HeightMap:
    """Procedural terrain elevation using multi-octave Perlin noise."""
    width: int
    height: int
    data: list[list[float]] = field(default_factory=list)

    def generate(self, rng: random.Random) -> None:
        """Generate terrain with octaved Perlin noise for natural appearance."""
        self.data = [[0.0] * self.width for _ in range(self.height)]

        # Multiple octaves for detail at different scales
        octaves = [
            (8, 2.0),    # large hills (cell_size=8, amplitude=2.0)
            (4, 1.0),    # medium features
            (2, 0.5),    # fine detail
        ]
        perlins = [_PerlinGradients(self.width, self.height, rng, cs) for cs, _ in octaves]

        for y in range(self.height):
            for x in range(self.width):
                val = 0.0
                for (_, amp), perlin in zip(octaves, perlins):
                    val += perlin.noise(float(x), float(y)) * amp
                self.data[y][x] = val

    def elevation_at(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y][x]
        return 0.0

    def is_valley(self, x: int, y: int) -> bool:
        return self.elevation_at(x, y) < -0.5

    def movement_cost(self, from_x: int, from_y: int,
                      to_x: int, to_y: int) -> float:
        """Extra energy cost for climbing uphill. Downhill is free."""
        e_from = self.elevation_at(from_x, from_y)
        e_to = self.elevation_at(to_x, to_y)
        climb = max(0.0, e_to - e_from)
        return climb  # returned as raw elevation delta; scaled by config


class BiomeMap:
    """Classifies each cell into a biome using elevation + moisture noise."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # 2-D grid of biome IDs
        self.data: list[list[int]] = [[BIOME_GRASSLANDS] * width for _ in range(height)]

    def generate(self, heightmap: HeightMap, rng: random.Random) -> None:
        """Build biome map from elevation + a separate moisture noise layer."""
        moisture_noise = _PerlinGradients(self.width, self.height, rng, cell_size=10)
        for y in range(self.height):
            for x in range(self.width):
                elev = heightmap.elevation_at(x, y)
                moisture = moisture_noise.noise(float(x), float(y))
                self.data[y][x] = self._classify(elev, moisture)

    @staticmethod
    def _classify(elevation: float, moisture: float) -> int:
        if elevation < -1.2:
            return BIOME_OCEAN
        if elevation < -0.3 and moisture > 0.1:
            return BIOME_WETLANDS
        if elevation > 1.5:
            return BIOME_MOUNTAINS
        if moisture < -0.2 and elevation > -0.3:
            return BIOME_DESERT
        return BIOME_GRASSLANDS

    def biome_at(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y][x]
        return BIOME_GRASSLANDS

    def props_at(self, x: int, y: int) -> BiomeProperties:
        return BIOME_PROPS[self.biome_at(x, y)]

    def get_distribution(self) -> dict[str, int]:
        """Count cells per biome for analytics."""
        counts: dict[str, int] = {n: 0 for n in BIOME_NAMES}
        for row in self.data:
            for b in row:
                counts[BIOME_NAMES[b]] += 1
        return counts
