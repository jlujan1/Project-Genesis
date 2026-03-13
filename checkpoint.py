"""Save/Load — serialise and restore the full simulation state.

Captures agent neural weights, memory, self-model, emotion state,
dream stats, environment state, and tick counter so a simulation can
be paused and resumed later.

Uses Python's pickle wrapped in gzip for compact binary snapshots.
"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def save_simulation(path: str, sandbox, agents, tick: int,
                    config) -> None:
    """Save the full simulation state to a compressed file.

    Parameters
    ----------
    path : str
        File path to write (e.g. "checkpoints/sim_t5000.gen").
    sandbox : Sandbox
        The world state.
    agents : list[ConsciousAgent]
        All agent instances.
    tick : int
        Current simulation tick.
    config : SimulationConfig
        The configuration used.
    """
    state: dict[str, Any] = {
        "version": 2,
        "tick": tick,
        "config": config,
        # Sandbox state
        "sandbox": _serialise_sandbox(sandbox),
        # Agents
        "agents": [_serialise_agent(a) for a in agents],
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(out), "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_simulation(path: str):
    """Load a simulation snapshot.  Returns (config, sandbox_state, agent_states, tick)."""
    with gzip.open(str(path), "rb") as f:
        state = pickle.load(f)

    if state.get("version", 1) < 2:
        raise ValueError("Snapshot version too old; re-run and save again.")

    return (state["config"], state["sandbox"],
            state["agents"], state["tick"])


def restore_sandbox(sandbox, saved: dict) -> None:
    """Apply saved sandbox state onto an existing Sandbox instance."""
    sandbox.tick_count = saved["tick_count"]
    sandbox.day_cycle.current_tick = saved["day_cycle_tick"]
    # Crystals — recreate from serialised positions
    from genesis.environment.resources import EnergyCrystal
    sandbox.crystals.clear()
    for cd in saved["crystals"]:
        from genesis.environment.physics import Vec2
        c = EnergyCrystal(Vec2(cd["x"], cd["y"]), cd["energy"])
        sandbox.crystals.append(c)


def restore_agent(agent, saved: dict) -> None:
    """Apply saved agent state onto an existing ConsciousAgent."""
    # Neural weights
    agent.brain.weights = saved["weights"].copy()
    agent.brain.potentials = saved["potentials"].copy()
    agent.brain.refractory_timers = saved["refractory_timers"].copy()

    # Body state
    from genesis.environment.physics import Vec2
    agent.body.position = Vec2(saved["pos_x"], saved["pos_y"])
    agent.body.velocity = Vec2(saved["vel_x"], saved["vel_y"])
    agent.body.energy = saved["energy"]
    agent.body.integrity = saved["integrity"]
    agent.body.alive = saved["alive"]
    agent.body.ticks_alive = saved["ticks_alive"]

    # Self-model
    agent.self_model.model_accuracy = saved["self_model_accuracy"]
    agent.self_model.update_count = saved["self_model_update_count"]
    agent.self_model.position_estimate = saved["self_model_pos"].copy()
    agent.self_model.velocity_estimate = saved["self_model_vel"].copy()

    # Episodic memory (long-term episodes)
    agent.episodic_memory.episodes = saved.get("ltm_episodes", [])

    # Emotion state
    if "emotion_values" in saved:
        agent.emotions.state.values = saved["emotion_values"].copy()

    # Dream stats
    ds = saved.get("dream_stats", {})
    agent.dream_engine.stats.total_dream_cycles = ds.get("dream_cycles", 0)
    agent.dream_engine.stats.total_replayed_episodes = ds.get("replayed", 0)

    # Tick counter
    agent.tick_count = saved.get("agent_tick", 0)


# ── internal helpers ────────────────────────────────────────────

def _serialise_sandbox(sandbox) -> dict:
    return {
        "tick_count": sandbox.tick_count,
        "day_cycle_tick": sandbox.day_cycle.current_tick,
        "crystals": [
            {"x": c.position.x, "y": c.position.y, "energy": c.energy}
            for c in sandbox.crystals
        ],
    }


def _serialise_agent(agent) -> dict:
    return {
        "agent_id": agent.agent_id,
        # Neural
        "weights": agent.brain.weights.copy(),
        "potentials": agent.brain.potentials.copy(),
        "refractory_timers": agent.brain.refractory_timers.copy(),
        # Body
        "pos_x": agent.body.position.x,
        "pos_y": agent.body.position.y,
        "vel_x": agent.body.velocity.x,
        "vel_y": agent.body.velocity.y,
        "energy": agent.body.energy,
        "integrity": agent.body.integrity,
        "alive": agent.body.alive,
        "ticks_alive": agent.body.ticks_alive,
        # Self-model
        "self_model_accuracy": agent.self_model.model_accuracy,
        "self_model_update_count": agent.self_model.update_count,
        "self_model_pos": agent.self_model.position_estimate.copy(),
        "self_model_vel": agent.self_model.velocity_estimate.copy(),
        # Memory
        "ltm_episodes": list(agent.episodic_memory.episodes),
        # Emotions
        "emotion_values": agent.emotions.state.values.copy(),
        # Dream stats
        "dream_stats": agent.dream_engine.get_summary(),
        # Tick
        "agent_tick": agent.tick_count,
    }
