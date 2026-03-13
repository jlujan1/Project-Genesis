"""Agent body — the digital avatar with sensors, actuators, and homeostasis.

The body connects the neural architecture to the sandbox environment.
It manages energy, integrity, proprioception, and translates motor
neuron outputs into physical actions.
"""

from __future__ import annotations

import math

import numpy as np

from genesis.config import AgentConfig
from genesis.environment.physics import Vec2, apply_physics, check_agent_collision
from genesis.environment.sandbox import Sandbox


from genesis.agent.tools import ToolInventory, NUM_TOOL_TYPES


# Action definitions
ACTION_NONE = 0
ACTION_MOVE_UP = 1
ACTION_MOVE_DOWN = 2
ACTION_MOVE_LEFT = 3
ACTION_MOVE_RIGHT = 4
ACTION_COLLECT = 5
ACTION_EMIT_SOUND = 6
ACTION_SPRINT = 7
ACTION_EXAMINE = 8
ACTION_BUILD = 9
ACTION_REST = 10
ACTION_STUDY = 11
ACTION_CRAFT = 12
ACTION_SHARE = 13
ACTION_PLANT = 14
ACTION_TEACH = 15
NUM_ACTIONS = 16

ACTION_NAMES = [
    "idle", "up", "down", "left", "right",
    "collect", "emit_sound", "sprint",
    "examine", "build", "rest", "study",
    "craft", "share", "plant", "teach",
]


class AgentBody:
    """The physical avatar in the sandbox — sensors, actuators, homeostasis."""

    def __init__(self, agent_id: int, config: AgentConfig, sandbox: Sandbox) -> None:
        self.agent_id = agent_id
        self.config = config
        self.sandbox = sandbox

        # Physical state
        self.position = sandbox.get_spawn_position()
        self.velocity = Vec2(0.0, 0.0)

        # Homeostasis variables
        self.energy = config.max_energy
        self.integrity = config.max_integrity
        self.alive = True

        # Pain/pleasure signals this tick
        self.pain_signal = 0.0
        self.pleasure_signal = 0.0

        # Latched signals (visible to other agents for multiple ticks)
        self.recent_reward_ticks: int = 0  # counts down from 5 on reward
        self.recent_pain_ticks: int = 0    # counts down from 5 on pain

        # Action tracking
        self.last_action = ACTION_NONE
        self.last_action_succeeded = True
        self.ticks_alive = 0

        # Sound emission state
        self.emitted_sound_id: int = -1  # which 'word' was emitted

        # Shelters built by agent
        self.shelters: set[tuple[int, int]] = set()
        self.max_shelters: int = 5

        # Discovery / examine state
        self.last_examine_discovery: float = 0.0  # novelty signal from examine

        # Study state
        self.study_boost_active: bool = False  # signals SNN to boost traces

        # Tool inventory
        self.tools = ToolInventory()

        # Sharing state — set when agent performs SHARE action
        self.sharing_energy: float = 0.0  # energy offered to nearby agent

    @property
    def energy_ratio(self) -> float:
        return self.energy / self.config.max_energy

    @property
    def integrity_ratio(self) -> float:
        return self.integrity / self.config.max_integrity

    @property
    def is_critical(self) -> bool:
        return (self.energy < self.config.critical_threshold or
                self.integrity < self.config.critical_threshold)

    @property
    def in_pain(self) -> bool:
        return (self.energy < self.config.pain_threshold or
                self.integrity < self.config.pain_threshold)

    def execute_action(self, action: int,
                       other_bodies: list[AgentBody] | None = None) -> None:
        """Translate an action index into physical movement/interaction."""
        self.pain_signal = 0.0
        self.pleasure_signal = 0.0
        self.last_action = action
        self.last_action_succeeded = True
        self.emitted_sound_id = -1

        # Decay latched signals
        self.recent_reward_ticks = max(0, self.recent_reward_ticks - 1)
        self.recent_pain_ticks = max(0, self.recent_pain_ticks - 1)

        old_pos = Vec2(self.position.x, self.position.y)
        move_force = 0.6

        if action == ACTION_MOVE_UP:
            self.velocity = self.velocity + Vec2(0, -move_force)
        elif action == ACTION_MOVE_DOWN:
            self.velocity = self.velocity + Vec2(0, move_force)
        elif action == ACTION_MOVE_LEFT:
            self.velocity = self.velocity + Vec2(-move_force, 0)
        elif action == ACTION_MOVE_RIGHT:
            self.velocity = self.velocity + Vec2(move_force, 0)
        elif action == ACTION_SPRINT:
            # Sprint in direction of current velocity
            if self.velocity.magnitude() > 0.1:
                sprint_dir = self.velocity.normalized()
                self.velocity = self.velocity + sprint_dir * (move_force * self.config.sprint_energy_multiplier)
            self.energy -= self.config.move_energy_cost * self.config.sprint_energy_multiplier
        elif action == ACTION_COLLECT:
            energy_gained = self.sandbox.collect_crystal_at(self.position)
            if energy_gained > 0:
                # Tool bonus for harvesting
                harvest_bonus = self.tools.get_harvest_bonus()
                energy_gained *= (1.0 + harvest_bonus)
                self.energy = min(self.config.max_energy, self.energy + energy_gained)
                self.pleasure_signal = energy_gained / self.config.crystal_energy_value
                self.recent_reward_ticks = 5
            elif energy_gained < 0:
                # Toxic crystal — hurts on collect
                self.energy = max(0.0, self.energy + energy_gained)
                self.pain_signal += abs(energy_gained) / self.config.max_energy
                self.recent_pain_ticks = 5
            else:
                # Try harvesting a mature crop
                civ = getattr(self.sandbox, 'civilization', None)
                gx, gy = self.position.grid_pos()
                crop_energy = civ.harvest_crop_at(gx, gy) if civ else 0.0
                if crop_energy > 0:
                    self.energy = min(self.config.max_energy,
                                     self.energy + crop_energy)
                    self.pleasure_signal = crop_energy / self.config.crystal_energy_value
                    self.recent_reward_ticks = 5
                else:
                    self.last_action_succeeded = False
        elif action == ACTION_EMIT_SOUND:
            self.sandbox.emit_audio(self.position, 0.8, self.agent_id)
            self.emitted_sound_id = 0  # basic call
        elif action == ACTION_EXAMINE:
            # Examine current cell — discover biome features, novelty
            gx, gy = self.position.grid_pos()
            biome = self.sandbox.biome_map.biome_at(gx, gy)
            elev = self.sandbox.heightmap.elevation_at(gx, gy)
            hazard = self.sandbox.get_hazard_at(self.position)
            # Novelty = how unfamiliar this cell type is
            novelty = 0.0
            cell_key = (gx // 4, gy // 4, biome)  # chunk+biome key
            if not hasattr(self, '_examined_chunks'):
                self._examined_chunks: set[tuple[int, int, int]] = set()
            if cell_key not in self._examined_chunks:
                self._examined_chunks.add(cell_key)
                novelty = 0.5 + abs(elev) * 0.1 + hazard * 0.2
            else:
                novelty = 0.05  # diminishing returns
            # Ruins give extra knowledge reward
            ruin_reward = self.sandbox.study_ruin_at(self.position, self.agent_id)
            novelty += ruin_reward
            self.last_examine_discovery = min(1.0, novelty)
            self.pleasure_signal += novelty * 0.3
            self.energy -= 0.005  # small cost
        elif action == ACTION_BUILD:
            # Build a structure at current position
            gx, gy = self.position.grid_pos()
            civ = getattr(self.sandbox, 'civilization', None)
            if (gx, gy) not in self.sandbox.obstacles:
                # If near a young crop, tend it instead of building
                if civ and civ.tend_crop_at(gx, gy):
                    self.pleasure_signal += 0.1
                    self.energy -= 1.0
                elif len(self.shelters) < self.max_shelters:
                    # Choose building type based on civ state
                    if civ:
                        from genesis.cognition.civilization import (
                            BUILDING_COSTS, SHELTER_BUILDINGS, BuildingType,
                        )
                        has_shelter = (gx, gy) in self.sandbox.shelters
                        bt = civ.best_building_for(gx, gy, has_shelter)
                        cost = BUILDING_COSTS[bt]
                        if self.energy > cost + 5.0:
                            civ.place_structure(gx, gy, bt, self.agent_id,
                                                self.ticks_alive)
                            if bt in SHELTER_BUILDINGS:
                                self.shelters.add((gx, gy))
                                self.sandbox.add_shelter(gx, gy, self.agent_id)
                            self.pleasure_signal += 0.25
                            self.energy -= cost
                        else:
                            self.last_action_succeeded = False
                    else:
                        self.shelters.add((gx, gy))
                        self.sandbox.add_shelter(gx, gy, self.agent_id)
                        self.pleasure_signal += 0.2
                        self.energy -= 5.0
                else:
                    self.last_action_succeeded = False
            else:
                self.last_action_succeeded = False
        elif action == ACTION_REST:
            # Active rest — recover integrity, gain pleasure
            if self.energy_ratio > 0.3:
                self.integrity = min(self.config.max_integrity,
                                     self.integrity + 0.15)
                self.pleasure_signal += 0.15
                self.energy -= 0.002  # tiny cost
            else:
                self.last_action_succeeded = False
        elif action == ACTION_STUDY:
            # Focused learning — boosts STDP traces, costs energy
            self.study_boost_active = True
            self.energy -= 0.02  # moderate cost
            self.pleasure_signal += 0.05
        elif action == ACTION_CRAFT:
            # Craft a tool — cycle through types, pick first craftable
            crafted = False
            for tt in range(NUM_TOOL_TYPES):
                if self.tools.can_craft(tt, self.energy):
                    cost = self.tools.craft(tt)
                    self.energy -= cost
                    self.pleasure_signal += 0.25
                    crafted = True
                    break
            if not crafted:
                self.last_action_succeeded = False
        elif action == ACTION_SHARE:
            # Share 5 energy with nearest agent (resolved in agent.py tick)
            if self.energy > 15.0:
                self.sharing_energy = 5.0
                self.energy -= 5.0
                self.pleasure_signal += 0.1
            else:
                self.last_action_succeeded = False
                self.sharing_energy = 0.0
        elif action == ACTION_PLANT:
            # Plant a crop (requires AGRICULTURE tech — checked in agent.py)
            gx, gy = self.position.grid_pos()
            civ = getattr(self.sandbox, 'civilization', None)
            if civ and civ.has_tech(4):  # Tech.AGRICULTURE == 4
                crop = civ.plant_crop(gx, gy, self.ticks_alive)
                if crop is not None:
                    self.energy -= 3.0
                    self.pleasure_signal += 0.15
                else:
                    self.last_action_succeeded = False
            else:
                self.last_action_succeeded = False
        elif action == ACTION_TEACH:
            # Teach knowledge to nearest agent (resolved in agent.py tick)
            self.energy -= 0.01
            self.pleasure_signal += 0.05
        new_pos, new_vel = apply_physics(self.position, self.velocity)

        # Collision detection (walls/obstacles)
        corrected_pos, collided = self.sandbox.check_position(new_pos)
        if collided:
            self.integrity -= self.config.collision_damage
            self.pain_signal += self.config.collision_damage / self.config.max_integrity
            self.velocity = Vec2(0.0, 0.0)
            self.last_action_succeeded = False
            # Stay at old position on collision
            self.position, _ = self.sandbox.check_position(self.position)
        else:
            self.position = corrected_pos
            self.velocity = new_vel

        # Agent-agent collision
        if other_bodies:
            other_positions = [
                b.position for b in other_bodies
                if b.agent_id != self.agent_id and b.alive
            ]
            if other_positions:
                pushed_pos, agent_collided = check_agent_collision(
                    self.position, other_positions,
                    push_force=self.config.agent_push_force,
                )
                if agent_collided:
                    # Clamp pushed position to world bounds
                    self.position, _ = self.sandbox.check_position(pushed_pos)
                    self.integrity -= self.config.agent_collision_damage
                    self.pain_signal += self.config.agent_collision_damage / self.config.max_integrity

        # Terrain movement cost (climbing uphill costs extra energy)
        if action in (ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT, ACTION_SPRINT):
            terrain_cost = self.sandbox.get_terrain_movement_cost(old_pos, self.position)
            self.energy -= terrain_cost

        # Energy drain (movement costs energy)
        if action in (ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT):
            self.energy -= self.config.move_energy_cost

        # Passive energy drain
        self.energy -= self.config.energy_drain_rate

        # Night energy drain — creates temporal behavioral pressure (sheltered = reduced)
        self.energy -= self.sandbox.get_night_energy_drain_at(self.position)

        # Biome energy drain — desert/ocean are harsher environments
        self.energy -= self.sandbox.get_biome_energy_drain(self.position)

        # Hazard zone damage — forces relocation
        hazard = self.sandbox.get_hazard_at(self.position)
        if hazard > 0.0:
            hazard_damage = hazard * 0.3
            self.energy -= hazard_damage * 0.5
            self.integrity -= hazard_damage
            self.pain_signal += hazard_damage / self.config.max_integrity

        # Passive integrity regeneration (slow self-repair when well-fed)
        if self.energy_ratio > 0.5 and self.integrity < self.config.max_integrity:
            repair = 0.01 + self.tools.get_repair_rate()
            self.integrity = min(self.config.max_integrity,
                                 self.integrity + repair)

        # Tick tools (durability decay)
        self.tools.tick()

        # Generate pain signal if below threshold
        if self.energy < self.config.pain_threshold:
            self.pain_signal += (self.config.pain_threshold - self.energy) / self.config.pain_threshold
            self.recent_pain_ticks = 5  # latch for 5 ticks
        if self.integrity < self.config.pain_threshold:
            self.pain_signal += (self.config.pain_threshold - self.integrity) / self.config.pain_threshold
            self.recent_pain_ticks = 5

        # Check death
        if self.energy <= 0 or self.integrity <= 0:
            self.energy = max(0.0, self.energy)
            self.integrity = max(0.0, self.integrity)
            self.alive = False

        self.ticks_alive += 1

    def get_sensory_input(self, other_agents: list[AgentBody]) -> dict:
        """Gather all sensory data from the environment."""
        own_pos = self.position.as_tuple()

        # Vision
        visible = self.sandbox.get_visible_cells(self.position, self.config.vision_range)

        # Other agent positions (for vision module)
        other_positions = [
            a.position.as_tuple() for a in other_agents
            if a.agent_id != self.agent_id and a.alive
        ]

        # Audio
        audio = self.sandbox.get_audible_events(self.position, self.config.hearing_range)

        # Proprioception
        proprio = {
            "energy": self.energy,
            "max_energy": self.config.max_energy,
            "integrity": self.integrity,
            "max_integrity": self.config.max_integrity,
            "velocity": self.velocity.as_tuple(),
            "position": own_pos,
        }

        return {
            "vision": visible,
            "audio": audio,
            "proprioception": proprio,
            "other_agents": other_positions,
            "own_pos": own_pos,
        }

    def get_state_vector(self) -> np.ndarray:
        """Compress body state into a vector for neural processing."""
        return np.array([
            self.position.x / self.sandbox.width,
            self.position.y / self.sandbox.height,
            self.velocity.x / 3.0,
            self.velocity.y / 3.0,
            self.energy / self.config.max_energy,
            self.integrity / self.config.max_integrity,
            self.pain_signal,
            self.pleasure_signal,
        ], dtype=np.float32)
