"""Hierarchical Goal System — multi-level drives with priority switching.

The agent has a *hierarchy* of goals (survive → forage → explore →
socialize) whose priorities shift dynamically based on homeostasis,
emotions, and environmental context.  Lower-level goals pre-empt
higher-level ones when urgency is high (e.g. flee when damaged).

Now includes subgoal decomposition: each top-level goal can spawn
concrete subgoals (navigate_to, collect, flee_from, approach_agent)
creating two-level hierarchical planning.

Inspired by Maslow's hierarchy of needs and subsumption architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Goal IDs (lowest index = highest priority when urgent)
GOAL_SURVIVE = 0
GOAL_FORAGE = 1
GOAL_EXPLORE = 2
GOAL_SOCIALIZE = 3
GOAL_LEARN = 4
GOAL_CREATE = 5
GOAL_FARM = 6
NUM_GOALS = 7
GOAL_NAMES = ["survive", "forage", "explore", "socialize", "learn", "create", "farm"]

# Subgoal types
SUBGOAL_NONE = 0
SUBGOAL_NAVIGATE = 1      # move toward a specific location
SUBGOAL_COLLECT = 2        # pick up a nearby crystal
SUBGOAL_FLEE = 3           # move away from danger
SUBGOAL_APPROACH_AGENT = 4 # move toward another agent
SUBGOAL_EMIT_SIGNAL = 5    # communicate
SUBGOAL_EXAMINE = 6        # study the current environment
SUBGOAL_BUILD = 7          # construct a shelter
SUBGOAL_REST = 8           # active rest and recovery
SUBGOAL_STUDY = 9          # focused learning
SUBGOAL_CRAFT = 10         # craft a tool
SUBGOAL_SHARE = 11         # share energy with nearby agent
SUBGOAL_FLEE_PREDATOR = 12 # flee from predator specifically
SUBGOAL_PLANT = 13         # plant a crop
SUBGOAL_TEACH = 14         # teach knowledge to nearby agent
NUM_SUBGOALS = 15
SUBGOAL_NAMES = ["none", "navigate", "collect", "flee",
                 "approach_agent", "emit_signal",
                 "examine", "build", "rest", "study",
                 "craft", "share", "flee_predator",
                 "plant", "teach"]


@dataclass
class Goal:
    """A single goal with a dynamically computed priority."""
    name: str
    base_priority: float          # resting priority
    current_priority: float = 0.0
    active_ticks: int = 0         # how long this goal has been active
    total_active_ticks: int = 0


@dataclass
class Subgoal:
    """A concrete, actionable subgoal spawned by a top-level goal."""
    subgoal_type: int = SUBGOAL_NONE
    target_position: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32))
    target_agent_id: int = -1
    urgency: float = 0.0
    ticks_active: int = 0
    completed: bool = False


class HierarchicalGoals:
    """Maintains and switches between a hierarchy of goals with subgoal planning.

    Each tick, goal priorities are recomputed from internal state:
      survive   — rises with low energy/integrity, high pain
      forage    — rises with moderate energy deficit
      explore   — rises when curiosity is high, resources nearby are low
      socialize — rises when loneliness emotion is high
      learn     — rises when curiosity is high and agent is safe
      create    — rises when energy is high and agent has explored

    The active goal spawns concrete subgoals (navigate, collect, flee, etc.)
    that generate targeted motor biases.
    """

    def __init__(self) -> None:
        self.goals = [
            Goal("survive",   base_priority=0.1),
            Goal("forage",    base_priority=0.3),
            Goal("explore",   base_priority=0.2),
            Goal("socialize", base_priority=0.1),
            Goal("learn",     base_priority=0.15),
            Goal("create",    base_priority=0.05),
            Goal("farm",      base_priority=0.0),
        ]
        self.active_goal: int = GOAL_FORAGE
        self.switch_count: int = 0
        self.active_subgoal: Subgoal = Subgoal()
        self.subgoal_completions: int = 0

    def update(self, *, energy_ratio: float, integrity_ratio: float,
               pain: float, curiosity: float, loneliness: float,
               nearby_crystals: int, nearby_agents: int,
               own_position: np.ndarray | None = None,
               best_crystal_dir: np.ndarray | None = None,
               threat_direction: np.ndarray | None = None,
               nearest_agent_pos: np.ndarray | None = None,
               forage_boost: float = 0.0,
               socialize_boost: float = 0.0,
               survive_boost: float = 0.0,
               map_coverage: float = 0.0,
               has_shelter_nearby: bool = False,
               has_agriculture: bool = False,
               has_mature_crops: bool = False) -> int:
        """Recompute priorities, switch goals, and decompose into subgoals.

        Returns the index of the currently active goal.
        """
        g = self.goals

        # === Survive ===
        survival_urgency = (max(0.0, 0.4 - energy_ratio) * 2.0
                            + max(0.0, 0.3 - integrity_ratio) * 3.0
                            + pain * 1.5)
        g[GOAL_SURVIVE].current_priority = (g[GOAL_SURVIVE].base_priority
                                            + survival_urgency + survive_boost)

        # === Forage ===
        hunger = max(0.0, 0.7 - energy_ratio)
        crystal_scarcity = max(0.0, 1.0 - nearby_crystals / 3.0)
        g[GOAL_FORAGE].current_priority = (g[GOAL_FORAGE].base_priority
                                           + hunger * 1.5
                                           + crystal_scarcity * 0.2
                                           + forage_boost)

        # === Explore ===
        g[GOAL_EXPLORE].current_priority = (g[GOAL_EXPLORE].base_priority
                                            + curiosity * 1.0
                                            + 0.1)  # exploration floor

        # === Socialize ===
        g[GOAL_SOCIALIZE].current_priority = (g[GOAL_SOCIALIZE].base_priority
                                              + min(0.4, loneliness * 0.6)
                                              + (0.15 if nearby_agents > 0 else 0.0)
                                              + socialize_boost)

        # === Learn ===
        # Rises when agent is safe, curious, and not starving
        safety = min(1.0, energy_ratio + integrity_ratio) / 2.0
        g[GOAL_LEARN].current_priority = (g[GOAL_LEARN].base_priority
                                          + curiosity * 0.8 * safety
                                          + 0.05)

        # === Create ===
        # Rises when well-fed, explored a lot, and no immediate threats
        creative_drive = (max(0.0, energy_ratio - 0.5) * 0.4
                          + map_coverage * 0.3)
        if pain > 0.2:
            creative_drive *= 0.2  # suppress when in pain
        if not has_shelter_nearby:
            creative_drive += 0.15  # want to build when no shelter
        g[GOAL_CREATE].current_priority = (g[GOAL_CREATE].base_priority
                                           + creative_drive)

        # === Farm ===
        # Only activates after agriculture tech discovered
        if has_agriculture:
            farm_drive = 0.3 + max(0.0, 0.6 - energy_ratio) * 0.5
            if has_mature_crops:
                farm_drive += 0.3  # harvest ready!
            g[GOAL_FARM].current_priority = farm_drive
        else:
            g[GOAL_FARM].current_priority = 0.0

        # Select highest priority
        new_goal = int(np.argmax([gg.current_priority for gg in g]))
        if new_goal != self.active_goal:
            self.switch_count += 1
            g[self.active_goal].active_ticks = 0
            # Goal changed — spawn new subgoal
            self.active_subgoal = Subgoal()
        self.active_goal = new_goal
        g[self.active_goal].active_ticks += 1
        g[self.active_goal].total_active_ticks += 1

        # === Subgoal decomposition ===
        self._decompose_subgoal(
            energy_ratio=energy_ratio,
            integrity_ratio=integrity_ratio,
            pain=pain,
            nearby_crystals=nearby_crystals,
            nearby_agents=nearby_agents,
            own_position=own_position,
            best_crystal_dir=best_crystal_dir,
            threat_direction=threat_direction,
            nearest_agent_pos=nearest_agent_pos,
        )

        return self.active_goal

    def _decompose_subgoal(self, *, energy_ratio: float,
                           integrity_ratio: float, pain: float,
                           nearby_crystals: int, nearby_agents: int,
                           own_position: np.ndarray | None,
                           best_crystal_dir: np.ndarray | None,
                           threat_direction: np.ndarray | None,
                           nearest_agent_pos: np.ndarray | None) -> None:
        """Decompose the active goal into a concrete subgoal."""
        sg = self.active_subgoal
        sg.ticks_active += 1

        # Check for subgoal completion / timeout
        if sg.ticks_active > 100 or sg.completed:
            self.subgoal_completions += int(sg.completed)
            sg = Subgoal()
            self.active_subgoal = sg

        goal = self.active_goal
        pos = own_position if own_position is not None else np.zeros(2, dtype=np.float32)

        if sg.subgoal_type != SUBGOAL_NONE:
            return  # already have an active subgoal

        if goal == GOAL_SURVIVE:
            if pain > 0.3 and threat_direction is not None:
                # Flee from threat
                sg.subgoal_type = SUBGOAL_FLEE
                sg.target_position = pos - threat_direction * 10.0
                sg.urgency = pain
            else:
                # Navigate away from danger
                sg.subgoal_type = SUBGOAL_FLEE
                sg.target_position = pos + np.array(
                    [np.random.uniform(-5, 5), np.random.uniform(-5, 5)],
                    dtype=np.float32)
                sg.urgency = max(0.3, 1.0 - energy_ratio)

        elif goal == GOAL_FORAGE:
            if nearby_crystals > 0:
                # Crystal nearby — collect it
                sg.subgoal_type = SUBGOAL_COLLECT
                sg.urgency = 0.7
            elif best_crystal_dir is not None:
                # Navigate toward known crystal location
                sg.subgoal_type = SUBGOAL_NAVIGATE
                sg.target_position = pos + best_crystal_dir * 8.0
                sg.urgency = 0.5
            else:
                # Explore to find crystals
                sg.subgoal_type = SUBGOAL_NAVIGATE
                sg.target_position = pos + np.array(
                    [np.random.uniform(-10, 10), np.random.uniform(-10, 10)],
                    dtype=np.float32)
                sg.urgency = 0.3

        elif goal == GOAL_EXPLORE:
            sg.subgoal_type = SUBGOAL_NAVIGATE
            sg.target_position = pos + np.array(
                [np.random.uniform(-15, 15), np.random.uniform(-15, 15)],
                dtype=np.float32)
            sg.urgency = 0.4

        elif goal == GOAL_SOCIALIZE:
            if nearby_agents > 0 and nearest_agent_pos is not None:
                sg.subgoal_type = SUBGOAL_APPROACH_AGENT
                sg.target_position = nearest_agent_pos
                sg.urgency = 0.5
            else:
                sg.subgoal_type = SUBGOAL_EMIT_SIGNAL
                sg.urgency = 0.3

        elif goal == GOAL_LEARN:
            if energy_ratio > 0.4:
                # Alternate between examining surroundings and focused study
                if np.random.random() < 0.5:
                    sg.subgoal_type = SUBGOAL_EXAMINE
                    sg.urgency = 0.5
                else:
                    sg.subgoal_type = SUBGOAL_STUDY
                    sg.urgency = 0.5
            else:
                # Too tired to learn — rest first
                sg.subgoal_type = SUBGOAL_REST
                sg.urgency = 0.4

        elif goal == GOAL_CREATE:
            if energy_ratio > 0.5 and integrity_ratio > 0.4:
                # Alternate between building and crafting
                if np.random.random() < 0.4:
                    sg.subgoal_type = SUBGOAL_CRAFT
                    sg.urgency = 0.5
                else:
                    sg.subgoal_type = SUBGOAL_BUILD
                    sg.urgency = 0.6
            else:
                # Recover before creating
                sg.subgoal_type = SUBGOAL_REST
                sg.urgency = 0.4

        elif goal == GOAL_FARM:
            if energy_ratio > 0.4:
                r = np.random.random()
                if r < 0.4:
                    sg.subgoal_type = SUBGOAL_PLANT
                    sg.urgency = 0.6
                elif r < 0.7:
                    sg.subgoal_type = SUBGOAL_BUILD  # tend crops
                    sg.urgency = 0.5
                else:
                    sg.subgoal_type = SUBGOAL_COLLECT  # harvest mature
                    sg.urgency = 0.7
            else:
                sg.subgoal_type = SUBGOAL_COLLECT
                sg.urgency = 0.6

        # Predator override — if threat is from predator, use FLEE_PREDATOR
        if threat_direction is not None and pain > 0.2:
            sg.subgoal_type = SUBGOAL_FLEE_PREDATOR
            if own_position is not None:
                sg.target_position = own_position - threat_direction * 15.0
            sg.urgency = min(1.0, pain + 0.3)

    def get_motor_bias(self, num_actions: int,
                       own_position: np.ndarray | None = None) -> np.ndarray:
        """Return an action bias vector based on the current subgoal.

        When a subgoal with a target position is active, biases movement
        toward that target. Falls back to goal-level bias otherwise.
        """
        bias = np.zeros(num_actions, dtype=np.float32)
        sg = self.active_subgoal
        g = self.active_goal

        # Subgoal-level biases (more targeted than goal-level)
        if sg.subgoal_type == SUBGOAL_NAVIGATE and own_position is not None:
            delta = sg.target_position - own_position
            scale = sg.urgency * 0.5
            if abs(delta[0]) > abs(delta[1]):
                if delta[0] > 0 and num_actions > 4:
                    bias[4] = scale  # right
                elif num_actions > 3:
                    bias[3] = scale  # left
            else:
                if delta[1] > 0 and num_actions > 2:
                    bias[2] = scale  # down
                elif num_actions > 1:
                    bias[1] = scale  # up
            return bias

        if sg.subgoal_type == SUBGOAL_COLLECT:
            if num_actions > 5:
                bias[5] = 0.8  # COLLECT — strong bias when crystal is nearby
            return bias

        if sg.subgoal_type == SUBGOAL_FLEE and own_position is not None:
            delta = sg.target_position - own_position
            scale = sg.urgency * 0.5
            if abs(delta[0]) > abs(delta[1]):
                if delta[0] > 0 and num_actions > 4:
                    bias[4] = scale
                elif num_actions > 3:
                    bias[3] = scale
            else:
                if delta[1] > 0 and num_actions > 2:
                    bias[2] = scale
                elif num_actions > 1:
                    bias[1] = scale
            return bias

        if sg.subgoal_type == SUBGOAL_APPROACH_AGENT and own_position is not None:
            delta = sg.target_position - own_position
            scale = sg.urgency * 0.15
            if abs(delta[0]) > abs(delta[1]):
                if delta[0] > 0 and num_actions > 4:
                    bias[4] = scale
                elif num_actions > 3:
                    bias[3] = scale
            else:
                if delta[1] > 0 and num_actions > 2:
                    bias[2] = scale
                elif num_actions > 1:
                    bias[1] = scale
            if num_actions > 6:
                bias[6] += 0.1  # also communicate
            return bias

        if sg.subgoal_type == SUBGOAL_EMIT_SIGNAL:
            if num_actions > 6:
                bias[6] = 0.2  # EMIT_SOUND
            return bias

        if sg.subgoal_type == SUBGOAL_EXAMINE:
            if num_actions > 8:
                bias[8] = 0.6 * sg.urgency  # EXAMINE
            return bias

        if sg.subgoal_type == SUBGOAL_BUILD:
            if num_actions > 9:
                bias[9] = 0.7 * sg.urgency  # BUILD
            return bias

        if sg.subgoal_type == SUBGOAL_REST:
            if num_actions > 10:
                bias[10] = 0.5 * sg.urgency  # REST
            return bias

        if sg.subgoal_type == SUBGOAL_STUDY:
            if num_actions > 11:
                bias[11] = 0.5 * sg.urgency  # STUDY
            return bias

        if sg.subgoal_type == SUBGOAL_CRAFT:
            if num_actions > 12:
                bias[12] = 0.6 * sg.urgency  # CRAFT
            return bias

        if sg.subgoal_type == SUBGOAL_SHARE:
            if num_actions > 13:
                bias[13] = 0.5 * sg.urgency  # SHARE
            return bias

        if sg.subgoal_type == SUBGOAL_PLANT:
            if num_actions > 14:
                bias[14] = 0.6 * sg.urgency  # PLANT
            return bias

        if sg.subgoal_type == SUBGOAL_TEACH:
            if num_actions > 15:
                bias[15] = 0.5 * sg.urgency  # TEACH
            return bias

        if sg.subgoal_type == SUBGOAL_FLEE_PREDATOR and own_position is not None:
            delta = sg.target_position - own_position
            scale = sg.urgency * 0.7  # stronger flee bias for predators
            if abs(delta[0]) > abs(delta[1]):
                if delta[0] > 0 and num_actions > 4:
                    bias[4] = scale
                elif num_actions > 3:
                    bias[3] = scale
            else:
                if delta[1] > 0 and num_actions > 2:
                    bias[2] = scale
                elif num_actions > 1:
                    bias[1] = scale
            if num_actions > 7:
                bias[7] = scale * 0.5  # SPRINT away
            return bias

        # Fallback: goal-level bias (original behaviour)
        if g == GOAL_SURVIVE:
            n_move = min(4, num_actions)
            bias[:n_move] = 0.15
        elif g == GOAL_FORAGE:
            if num_actions > 5:
                bias[5] = 0.25
            n_move = min(4, num_actions)
            bias[:n_move] = 0.05
        elif g == GOAL_EXPLORE:
            n_move = min(4, num_actions)
            bias[:n_move] = 0.1
            bias += np.random.randn(num_actions).astype(np.float32) * 0.05
        elif g == GOAL_SOCIALIZE:
            if num_actions > 7:
                bias[7] = 0.2
        elif g == GOAL_LEARN:
            if num_actions > 8:
                bias[8] = 0.15   # EXAMINE
            if num_actions > 11:
                bias[11] = 0.15  # STUDY
        elif g == GOAL_CREATE:
            if num_actions > 9:
                bias[9] = 0.2    # BUILD

        return bias

    def mark_subgoal_completed(self) -> None:
        """Mark the current subgoal as completed (e.g. crystal collected)."""
        self.active_subgoal.completed = True

    def get_encoding(self) -> np.ndarray:
        """Encode goal state for a workspace packet."""
        enc = np.zeros(10, dtype=np.float32)
        for i, g in enumerate(self.goals):
            enc[i] = g.current_priority
        enc[6] = self.active_goal / max(1, NUM_GOALS)
        enc[7] = min(1.0, self.goals[self.active_goal].active_ticks / 100.0)
        enc[8] = self.active_subgoal.subgoal_type / max(1, NUM_SUBGOALS)
        enc[9] = self.active_subgoal.urgency
        return enc

    def get_summary(self) -> dict:
        result = {
            "active_goal": GOAL_NAMES[self.active_goal],
            "active_subgoal": SUBGOAL_NAMES[self.active_subgoal.subgoal_type],
            "subgoal_urgency": self.active_subgoal.urgency,
            "subgoal_ticks": self.active_subgoal.ticks_active,
            "subgoal_completions": self.subgoal_completions,
            "switch_count": self.switch_count,
        }
        for g in self.goals:
            result[g.name + "_priority"] = g.current_priority
            result[g.name + "_total_ticks"] = g.total_active_ticks
        return result
