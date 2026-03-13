"""Weather & Seasons — dynamic environmental conditions over time.

Adds seasonal cycles (spring/summer/autumn/winter) on top of day/night,
plus weather events (storms, droughts, fog) that modify biome behaviour,
resource availability, and hazard frequency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


SEASON_SPRING = 0
SEASON_SUMMER = 1
SEASON_AUTUMN = 2
SEASON_WINTER = 3
SEASON_NAMES = ["spring", "summer", "autumn", "winter"]
NUM_SEASONS = 4

WEATHER_CLEAR = 0
WEATHER_RAIN = 1
WEATHER_STORM = 2
WEATHER_DROUGHT = 3
WEATHER_FOG = 4
WEATHER_NAMES = ["clear", "rain", "storm", "drought", "fog"]
NUM_WEATHER = 5


@dataclass
class SeasonalModifiers:
    """How the current season + weather modify the environment."""
    resource_spawn_mult: float = 1.0   # crystal spawn rate multiplier
    energy_drain_mult: float = 1.0     # metabolic cost multiplier
    hazard_chance_mult: float = 1.0    # hazard spawn rate multiplier
    vision_mult: float = 1.0          # vision range multiplier
    predator_spawn_mult: float = 1.0   # predator frequency multiplier
    crystal_energy_mult: float = 1.0   # crystal value multiplier
    movement_cost_mult: float = 1.0    # terrain difficulty multiplier
    integrity_decay: float = 0.0       # extra integrity loss per tick


class WeatherSystem:
    """Manages seasons and weather events.

    One full year = 4 seasons × season_length ticks.
    Weather events happen within seasons with biased probabilities.
    """

    def __init__(self, season_length: int = 6000, seed: int = 42) -> None:
        self.season_length = season_length
        self.year_length = season_length * NUM_SEASONS
        self.rng = random.Random(seed + 777)
        self.tick_count = 0

        # Current weather state
        self.current_weather: int = WEATHER_CLEAR
        self._weather_duration: int = 0
        self._weather_timer: int = 0

        # Season transition tracking
        self._prev_season: int = -1

    @property
    def season(self) -> int:
        """Current season index (0-3)."""
        return (self.tick_count % self.year_length) // self.season_length

    @property
    def season_name(self) -> str:
        return SEASON_NAMES[self.season]

    @property
    def weather_name(self) -> str:
        return WEATHER_NAMES[self.current_weather]

    @property
    def season_progress(self) -> float:
        """0.0 to 1.0 through current season."""
        return (self.tick_count % self.season_length) / self.season_length

    @property
    def year_progress(self) -> float:
        """0.0 to 1.0 through the full year."""
        return (self.tick_count % self.year_length) / self.year_length

    def tick(self) -> float:
        """Advance one tick. Returns environment_change signal (0-1)."""
        self.tick_count += 1
        env_change = 0.0

        # Season transition detection
        s = self.season
        if s != self._prev_season:
            env_change = 0.8
            self._prev_season = s

        # Weather event lifecycle
        if self._weather_timer > 0:
            self._weather_timer -= 1
            if self._weather_timer <= 0:
                if self.current_weather != WEATHER_CLEAR:
                    env_change = max(env_change, 0.5)
                self.current_weather = WEATHER_CLEAR
        else:
            # Chance of new weather event
            if self.rng.random() < self._weather_chance():
                self._start_weather_event()
                env_change = max(env_change, 0.6)

        return env_change

    def _weather_chance(self) -> float:
        """Per-tick chance of starting a weather event, season-dependent."""
        base = 0.001  # ~every 1000 ticks
        season_mods = {
            SEASON_SPRING: 1.5,  # more rain in spring
            SEASON_SUMMER: 0.8,  # calmer summer
            SEASON_AUTUMN: 1.2,  # autumn storms
            SEASON_WINTER: 1.0,  # moderate
        }
        return base * season_mods.get(self.season, 1.0)

    def _start_weather_event(self) -> None:
        """Begin a weather event based on seasonal probabilities."""
        s = self.season
        # Weather probabilities per season
        probs = {
            SEASON_SPRING: [0.0, 0.50, 0.15, 0.05, 0.30],
            SEASON_SUMMER: [0.0, 0.20, 0.10, 0.40, 0.30],
            SEASON_AUTUMN: [0.0, 0.30, 0.35, 0.10, 0.25],
            SEASON_WINTER: [0.0, 0.15, 0.25, 0.10, 0.50],
        }
        weights = probs.get(s, [0.0, 0.25, 0.25, 0.25, 0.25])
        # Select weather (skip CLEAR=0)
        roll = self.rng.random()
        cumul = 0.0
        for i in range(1, NUM_WEATHER):
            cumul += weights[i]
            if roll < cumul:
                self.current_weather = i
                break
        else:
            self.current_weather = WEATHER_RAIN

        # Duration depends on type
        durations = {
            WEATHER_RAIN: (150, 400),
            WEATHER_STORM: (80, 200),
            WEATHER_DROUGHT: (300, 600),
            WEATHER_FOG: (100, 300),
        }
        lo, hi = durations.get(self.current_weather, (100, 300))
        self._weather_timer = self.rng.randint(lo, hi)

    def get_modifiers(self) -> SeasonalModifiers:
        """Get combined season + weather modifiers for the current tick."""
        m = SeasonalModifiers()
        s = self.season

        # Seasonal base modifiers
        if s == SEASON_SPRING:
            m.resource_spawn_mult = 1.4
            m.crystal_energy_mult = 1.1
            m.predator_spawn_mult = 0.7
        elif s == SEASON_SUMMER:
            m.resource_spawn_mult = 1.0
            m.energy_drain_mult = 1.15  # heat
            m.crystal_energy_mult = 0.9
            m.predator_spawn_mult = 1.3
        elif s == SEASON_AUTUMN:
            m.resource_spawn_mult = 1.2
            m.crystal_energy_mult = 1.2  # harvest bounty
            m.predator_spawn_mult = 1.0
        elif s == SEASON_WINTER:
            m.resource_spawn_mult = 0.5
            m.energy_drain_mult = 1.3  # cold
            m.crystal_energy_mult = 0.7
            m.movement_cost_mult = 1.2  # snow
            m.integrity_decay = 0.002  # exposure
            m.predator_spawn_mult = 0.5

        # Weather overlays
        w = self.current_weather
        if w == WEATHER_RAIN:
            m.resource_spawn_mult *= 1.5
            m.vision_mult = 0.8
        elif w == WEATHER_STORM:
            m.resource_spawn_mult *= 0.5
            m.hazard_chance_mult = 3.0
            m.vision_mult = 0.5
            m.energy_drain_mult *= 1.2
            m.integrity_decay += 0.005  # storm damage
            m.predator_spawn_mult *= 0.3  # predators hide too
        elif w == WEATHER_DROUGHT:
            m.resource_spawn_mult *= 0.3
            m.energy_drain_mult *= 1.3
            m.crystal_energy_mult *= 0.6
        elif w == WEATHER_FOG:
            m.vision_mult = 0.4
            m.predator_spawn_mult *= 1.5  # predators hunt in fog

        return m

    def get_temperature(self) -> float:
        """Normalized temperature (0=freezing, 1=hot).

        Sinusoidal cycle across the year, with weather perturbations.
        """
        base = 0.5 + 0.4 * math.sin(2 * math.pi * (self.year_progress - 0.25))
        # Weather adjustments
        if self.current_weather == WEATHER_STORM:
            base -= 0.1
        elif self.current_weather == WEATHER_DROUGHT:
            base += 0.15
        return max(0.0, min(1.0, base))

    def get_summary(self) -> dict:
        return {
            "season": self.season_name,
            "weather": self.weather_name,
            "temperature": round(self.get_temperature(), 2),
            "season_progress": round(self.season_progress, 2),
            "year_progress": round(self.year_progress, 2),
        }
