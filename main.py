"""Project Genesis — Main Simulation Loop.

Boots up the sandbox, spawns conscious agents, and runs the continuous
simulation with real-time analytics and visualization.

Usage:
    python -m genesis.main                  # default settings
    python -m genesis.main --ticks 100000   # run for 100k ticks
    python -m genesis.main --agents 3       # spawn 3 agents
    python -m genesis.main --fast           # no rendering, max speed
    python -m genesis.main --seed 123       # set random seed
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
import time

from genesis.agent.agent import ConsciousAgent
from genesis.analytics.dashboard import Dashboard
from genesis.analytics.logger import DataLogger
from genesis.analytics.tests import run_all as run_consciousness_tests, print_report as print_test_report
from genesis.config import SimulationConfig
from genesis.environment.sandbox import Sandbox
from genesis.visualization.renderer import Renderer


def create_simulation(config: SimulationConfig) -> tuple[Sandbox, list[ConsciousAgent]]:
    """Initialize the sandbox and spawn agents."""
    sandbox = Sandbox(config.world, seed=config.seed)
    agents = []
    for i in range(config.num_agents):
        agent = ConsciousAgent(agent_id=i, config=config, sandbox=sandbox)
        agents.append(agent)
    return sandbox, agents


def run_simulation(config: SimulationConfig, fast_mode: bool = False,
                   log_dir: str | None = None, gui: bool = False,
                   save_path: str | None = None,
                   load_path: str | None = None,
                   evolve: bool = False, gui_3d: bool = False) -> None:
    """Run the main simulation loop."""
    print("=" * 60)
    print("  PROJECT GENESIS — Initializing...")
    print("=" * 60)
    print(f"  World: {config.world.width}x{config.world.height}")
    print(f"  Agents: {config.num_agents}")
    print(f"  Neurons per agent: {config.neural.num_neurons}")
    print(f"  Max ticks: {config.max_ticks:,}")
    print(f"  Seed: {config.seed}")
    print()

    # Create world and agents
    sandbox, agents = create_simulation(config)

    # Restore from checkpoint if requested
    if load_path:
        from genesis.analytics.checkpoint import (
            load_simulation, restore_sandbox, restore_agent,
        )
        _, sb_state, ag_states, start_tick = load_simulation(load_path)
        restore_sandbox(sandbox, sb_state)
        for saved, agent in zip(ag_states, agents):
            restore_agent(agent, saved)
        print(f"  Restored checkpoint from {load_path} (tick {start_tick})")

    dashboard = Dashboard()
    renderer = Renderer(viewport_width=50, viewport_height=25)

    # Optional CSV logger
    logger: DataLogger | None = None
    if log_dir:
        logger = DataLogger(log_dir, config.num_agents)
        print(f"  Logging metrics to {log_dir}/")

    # Optional Pygame GUI
    gui_vis = None
    if gui_3d:
        from genesis.visualization.renderer_3d import Visualiser3D
        gui_vis = Visualiser3D(sandbox, config.num_agents)
        fast_mode = False
    elif gui:
        from genesis.visualization.pygame_vis import PygameVisualiser
        gui_vis = PygameVisualiser(sandbox, config.num_agents)
        fast_mode = False  # GUI implies non-fast

    print(f"  Sandbox generated: {len(sandbox.obstacles)} obstacles, "
          f"{len(sandbox.crystals)} crystals")
    print(f"  Features: {len(sandbox.berry_bushes)} berry bushes, "
          f"{len(sandbox.fungi)} fungi, {len(sandbox.ruins)} ruins, "
          f"{len(sandbox.rivers)} river cells")
    biome_dist = sandbox.biome_map.get_distribution()
    biome_parts = [f"{n}={c}" for n, c in biome_dist.items() if c > 0]
    print(f"  Biomes: {', '.join(biome_parts)}")
    for agent in agents:
        pos = agent.body.position
        conns = agent.brain.get_active_connections()
        print(f"  Agent {agent.agent_id} spawned at ({pos.x:.0f}, {pos.y:.0f}) "
              f"with {conns:,} synapses")
    print()

    if not fast_mode:
        print("  Press Ctrl+C to stop the simulation.")
        print()
        time.sleep(1.5)

    # Simulation loop
    start_time = time.time()

    # Evolution engine (optional)
    evo_engine = None
    if evolve:
        from genesis.cognition.evolution import EvolutionEngine
        evo_engine = EvolutionEngine()
        for a in agents:
            evo_engine.register_agent(a.agent_id)
        print("  Evolution enabled — dead agents will respawn with mutated weights")

    # Track which agents were alive last tick (for evolution respawn)
    prev_alive = {a.agent_id: a.alive for a in agents}

    try:
        for tick in range(1, config.max_ticks + 1):
            # Advance the world (with agent positions for predator AI)
            agent_positions = [(a.body.position, a.alive) for a in agents]
            sandbox.tick(agent_positions)

            # Apply predator damage to agents
            for agent_idx, energy_dmg, integrity_dmg in sandbox.predator_contacts:
                if 0 <= agent_idx < len(agents) and agents[agent_idx].alive:
                    a = agents[agent_idx]
                    a.body.energy -= energy_dmg
                    a.body.integrity -= integrity_dmg
                    a.body.pain_signal += (energy_dmg + integrity_dmg) / a.config.agent.max_energy
                    a.body.recent_pain_ticks = 5

            # Advance each agent
            for agent in agents:
                agent.tick(agents)
                if evo_engine and agent.alive:
                    evo_engine.record_tick(agent.agent_id,
                                           max(0.0, agent.body.pleasure_signal))

            # Evolution: respawn dead agents with inherited + mutated weights
            if evo_engine:
                for agent in agents:
                    if prev_alive.get(agent.agent_id, True) and not agent.alive:
                        evo_engine.agent_died(agent.agent_id)
                        alive_agents = [a for a in agents if a.alive]
                        parent = evo_engine.select_parent(alive_agents)
                        if parent is not None and hasattr(parent, 'brain'):
                            parent_weights = parent.brain.weights
                            child_weights, muts = evo_engine.inherit_weights(parent_weights)
                            # Reset the dead agent
                            agent.body.energy = agent.config.agent.max_energy
                            agent.body.integrity = agent.config.agent.max_integrity
                            agent.body.alive = True
                            agent.body.position = agent.body.sandbox.get_spawn_position()
                            agent.body.velocity.__class__(0.0, 0.0)
                            agent.brain.weights = child_weights
                            agent.brain.potentials[:] = 0.0
                            agent.brain.refractory_timers[:] = 0
                            # Inherit personality traits
                            child_traits = evo_engine.inherit_personality(
                                parent.personality.traits)
                            agent.personality.traits = child_traits
                            agent.emotions.state.baseline = \
                                agent.personality.apply_to_baseline(
                                    agent.emotions.state.baseline)
                            # Inherit cognitive map knowledge
                            inherited_cells = evo_engine.inherit_cognitive_map(
                                parent.cognitive_map, agent.cognitive_map)
                            evo_engine.register_agent(agent.agent_id,
                                                       parent_id=parent.agent_id)
                            if fast_mode and tick % 1000 < 2:
                                print(f"    ↻ Agent {agent.agent_id} respawned "
                                      f"(gen {evo_engine.generation_counter}, "
                                      f"{muts} mutations)")
                    prev_alive[agent.agent_id] = agent.alive

            # ── Population growth (civilisation) ──
            civ = getattr(sandbox, 'civilization', None)
            if civ and tick % 50 == 0:
                alive_agents = [a for a in agents if a.alive]
                alive_count = len(alive_agents)
                avg_energy = (sum(a.body.energy_ratio for a in alive_agents)
                              / max(1, alive_count))
                num_shelters = len(sandbox.shelters) + sum(
                    1 for s in civ.structures.values()
                    if s.building_type != 0  # non-shelter structures also count
                )
                if civ.should_spawn(alive_count, avg_energy, num_shelters, tick):
                    new_id = len(agents)
                    new_agent = ConsciousAgent(
                        agent_id=new_id, config=config, sandbox=sandbox)
                    # Inherit knowledge from a random living agent
                    if alive_agents:
                        parent = random.choice(alive_agents)
                        # Copy personalit traits with slight mutation
                        from genesis.cognition.evolution import EvolutionEngine
                        _evo = EvolutionEngine()
                        child_traits = _evo.inherit_personality(
                            parent.personality.traits)
                        new_agent.personality.traits = child_traits
                        new_agent.emotions.state.baseline = \
                            new_agent.personality.apply_to_baseline(
                                new_agent.emotions.state.baseline)
                        # Inherit partial cognitive map
                        _evo.inherit_cognitive_map(
                            parent.cognitive_map, new_agent.cognitive_map)
                    agents.append(new_agent)
                    if evo_engine:
                        evo_engine.register_agent(new_id)
                    prev_alive[new_id] = True
                    # Update 3D visualiser agent count
                    if gui_vis:
                        gui_vis.num_agents = len(agents)

            # Rendering and analytics
            _ts = gui_vis.time_scale if gui_vis else 1
            if not fast_mode and tick % config.analytics.dashboard_refresh == 0:
                if gui_vis:
                    if not gui_vis.handle_events():
                        break
                    gui_vis.render(sandbox, agents, tick)
                else:
                    _render_frame(tick, sandbox, agents, dashboard, renderer)
            elif gui_vis and tick % max(2, _ts) == 0:
                # Render 3D GUI; skip frames when time-lapse active
                if not gui_vis.handle_events():
                    break
                gui_vis.render(sandbox, agents, tick)

            # CSV logging
            if logger and tick % config.analytics.phi_sample_interval == 0:
                metrics = [a.compute_analytics() for a in agents]
                logger.log(tick, metrics)

            # Periodic log in fast mode
            if fast_mode and tick % 1000 == 0:
                elapsed = time.time() - start_time
                tps = tick / max(0.01, elapsed)
                alive = sum(1 for a in agents if a.alive)
                civ_info = ""
                civ = getattr(sandbox, 'civilization', None)
                if civ:
                    civ_info = (f"  |  Epoch: {civ.epoch.name}"
                                f"  |  Techs: {len(civ.discovered_techs)}")
                print(f"  Tick {tick:>8,}  |  {tps:,.0f} t/s  |  "
                      f"Alive: {alive}/{len(agents)}  |  "
                      f"Crystals: {len(sandbox.crystals)}{civ_info}")

            # Check if all agents are dead (only stop if no evolution)
            if not evolve and not any(a.alive for a in agents):
                print("\n  All agents have died.")
                break

    except KeyboardInterrupt:
        print("\n\n  Simulation interrupted by user.")

    # Final report
    elapsed = time.time() - start_time
    final_tick = min(tick, config.max_ticks)
    _print_final_report(final_tick, elapsed, agents)

    if logger:
        logger.close()
        print(f"\n  CSV logs saved to {log_dir}/")

    if gui_vis:
        gui_vis.quit()

    # Save checkpoint
    if save_path:
        from genesis.analytics.checkpoint import save_simulation
        save_simulation(save_path, sandbox, agents, final_tick, config)
        print(f"  Checkpoint saved to {save_path}")


def _render_frame(tick: int, sandbox: Sandbox, agents: list[ConsciousAgent],
                  dashboard: Dashboard, renderer: Renderer) -> None:
    """Render one frame of the visualization."""
    import os
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")

    # Render map
    print(renderer.render(sandbox, agents, follow_agent=0))
    print()

    # Gather metrics
    metrics = [a.compute_analytics() for a in agents]
    dashboard.update(tick, sandbox.get_state_summary(), metrics)
    print(dashboard.render())


def _print_final_report(total_ticks: int, elapsed: float,
                        agents: list[ConsciousAgent]) -> None:
    """Print the post-simulation analysis."""
    print()
    print("=" * 60)
    print("  SIMULATION COMPLETE — Final Analysis")
    print("=" * 60)
    print(f"  Total ticks: {total_ticks:,}")
    print(f"  Wall time: {elapsed:.1f}s ({total_ticks / max(0.01, elapsed):,.0f} ticks/sec)")
    print()

    for agent in agents:
        metrics = agent.compute_analytics()
        assessment = agent.phi_calculator.get_consciousness_assessment(
            self_model_accuracy=agent.self_model.model_accuracy,
            attention_accuracy=agent.attention_schema.schema_accuracy,
            metacognitive_confidence=agent.inner_speech.confidence,
            binding_coherence=agent.binding.coherence,
            empowerment=agent.empowerment.empowerment,
            narrative_identity=agent.narrative.identity_strength,
            curiosity_level=agent.curiosity.curiosity_level,
        )

        print(f"  -- Agent {agent.agent_id} --")
        print(f"  Status: {'ALIVE' if agent.alive else 'DEAD'} "
              f"(survived {agent.body.ticks_alive:,} ticks)")
        print(f"  Final Energy: {agent.body.energy:.1f} / "
              f"Integrity: {agent.body.integrity:.1f}")
        print(f"  Shelters Built: {len(agent.body.shelters)}")
        tools = agent.body.tools.get_summary()
        tool_names = ', '.join(tools['tools']) or 'none'
        print(f"  Tools Carried: {tools['tool_count']} ({tool_names})")
        coop = agent.cooperation.get_summary()
        print(f"  Cooperation: shared={coop['energy_shared']:.0f}e "
              f"received={coop['energy_received']:.0f}e "
              f"partners={coop['partners']}")
        emo = agent.emotions.get_summary()
        print(f"  Mood: valence={emo['mood_valence']:+.2f} "
              f"arousal={emo['mood_arousal']:.2f} "
              f"bonds={emo['bonds']}")
        print()
        print(f"  Consciousness Assessment:")
        print(f"    Phi:              {assessment['phi']:.4f} "
              f"[{assessment['phi_trend']}]")
        print(f"    Complexity:       {assessment['complexity']:.4f} "
              f"[{assessment['complexity_trend']}]")
        print(f"    Reverberation:    {assessment['reverberation']:.4f}")
        print(f"    COMPOSITE SCORE:  {assessment['composite_score']:.4f}")
        print(f"    Phase:            {assessment['phase']}")
        print()
        print(f"  Self-Model:")
        print(f"    Accuracy:         {agent.self_model.model_accuracy:.2%}")
        print(f"    Ego Emerged:      {'YES' if agent.self_model.has_ego else 'No'}")
        print(f"    Updates:          {agent.self_model.update_count:,}")
        print()

        att = agent.attention_schema.get_summary()
        print(f"  Attention Schema:")
        print(f"    Current Focus:    {att['current_focus']}")
        print(f"    Max Focus Streak: {att['max_focus_streak']}")
        print(f"    Schema Accuracy:  {att['schema_accuracy']:.2%}")
        print(f"    Predicted Next:   {att['predicted_next']}")
        print()
        print(f"  Neural Network:")
        print(f"    Active Synapses:  {agent.brain.get_active_connections():,}")
        print(f"    Prediction Error: {agent.prediction_engine.average_error:.4f}")
        print()
        print(f"  Memory:")
        print(f"    Episodic (LTM):   {agent.episodic_memory.long_term_count}")
        print(f"    Working Memory:   {len(agent.working_memory.buffer)} items")
        print()

        dream = agent.dream_engine.get_summary()
        if dream["dream_cycles"] > 0:
            print(f"  Dreaming:")
            print(f"    Dream Cycles:     {dream['dream_cycles']}")
            print(f"    Replayed Memories:{dream['replayed_episodes']}")
            print(f"    Synapses Pruned:  {dream['connections_pruned']:,}")
            print(f"    Synapses Boosted: {dream['connections_strengthened']:,}")
            print()

        emo = agent.emotions.get_summary()
        print(f"  Emotions:")
        print(f"    Dominant:         {emo['dominant']}")
        print(f"    Valence:          {emo['valence']:+.3f}")
        print(f"    Arousal:          {emo['arousal']:.3f}")
        for en in ["fear", "curiosity", "contentment",
                    "frustration", "loneliness", "surprise"]:
            bar = "#" * int(emo[en] * 20)
            print(f"    {en:14s}    {bar} {emo[en]:.2f}")
        print()

        social = agent.social_learning.get_summary()
        if social["total_observations"] > 0:
            print(f"  Social Learning:")
            print(f"    Observations:     {social['total_observations']}")
            print(f"    Imitations:       {social['total_imitations']}")
            print()

        # Curiosity
        cur = agent.curiosity.get_summary()
        print(f"  Curiosity:")
        print(f"    Curiosity Level:  {cur['curiosity_level']:.3f}")
        print(f"    Learning Progress:{cur['learning_progress']:+.4f}")
        print(f"    Novel Cells:      {cur['novel_cells_visited']}")
        print()

        # Inner Speech / Metacognition
        meta = agent.inner_speech.get_summary()
        print(f"  Inner Speech / Metacognition:")
        print(f"    Confidence:       {meta['confidence']:.2%}")
        print(f"    Distinct Symbols: {meta['distinct_symbols_used']}")
        print(f"    Reflections:      {meta['reflection_count']}")
        print()

        # Cognitive Map
        cmap = agent.cognitive_map.get_summary()
        print(f"  Cognitive Map:")
        print(f"    Coverage:         {cmap['coverage']:.1%} "
              f"({cmap['explored_cells']}/{cmap['total_cells']} cells)")
        print(f"    Best Crystal:     {cmap['best_crystal_score']:.2f}")
        print(f"    Worst Danger:     {cmap['worst_danger_score']:.2f}")
        print()

        # Hierarchical Goals
        goals = agent.goal_system.get_summary()
        print(f"  Hierarchical Goals:")
        print(f"    Active Goal:      {goals['active_goal']}")
        print(f"    Goal Switches:    {goals['switch_count']}")
        for gn in ["survive", "forage", "explore", "socialize", "learn", "create"]:
            bar = "#" * int(goals[gn + "_priority"] * 15)
            print(f"    {gn:14s}    {bar} p={goals[gn + '_priority']:.2f}  "
                  f"t={goals[gn + '_total_ticks']}")
        print()

        # Counterfactual Reasoning
        cf = agent.counterfactual.get_summary()
        if cf["total_replays"] > 0:
            print(f"  Counterfactual Reasoning:")
            print(f"    Replays:          {cf['total_replays']}")
            print(f"    Regrets:          {cf['total_regrets']}")
            print(f"    Reliefs:          {cf['total_reliefs']}")
            print(f"    Avg Regret:       {cf['average_regret']:.3f}")
            print()

        # Cultural Transmission
        cult = agent.culture.get_summary()
        if cult["teachings_given"] > 0 or cult["teachings_received"] > 0:
            print(f"  Cultural Transmission:")
            print(f"    Teachings Given:  {cult['teachings_given']}")
            print(f"    Teachings Recv'd: {cult['teachings_received']}")
            print(f"    Knowledge Base:   {cult['knowledge_base_size']} items")
            print()

        # Critical Periods
        cp = agent.critical_periods.get_summary(agent.tick_count)
        print(f"  Critical Periods (tick {agent.tick_count}):")
        print(f"    Effective LR:     {cp['effective_lr_multiplier']:.2f}x")
        for domain in ["sensory", "motor", "social", "language", "self_model"]:
            info = cp[domain]
            status = info["status"].upper()
            print(f"    {domain:14s}    {status:8s}  "
                  f"openness={info['openness']:.2f}  "
                  f"mult={info['multiplier']:.2f}")
        print()

        # Multi-Modal Binding
        bind = agent.binding.get_summary()
        print(f"  Multi-Modal Binding:")
        print(f"    Binding Strength: {bind['binding_strength']:.3f}")
        print(f"    Coherence:        {bind['coherence']:.3f}")
        print(f"    Avg Binding:      {bind['average_binding']:.3f}")
        print(f"    Weights — V={bind['vision_weight']:.2f}  "
              f"A={bind['audio_weight']:.2f}  "
              f"P={bind['proprio_weight']:.2f}")
        print()

        ws = agent.workspace.get_broadcast_summary()
        print(f"  Global Workspace:")
        print(f"    Total Broadcasts: {ws['total_broadcasts']:,}")
        if ws["broadcast_distribution"]:
            print(f"    Distribution:")
            for mod, frac in sorted(ws["broadcast_distribution"].items(),
                                     key=lambda x: x[1], reverse=True):
                bar = "#" * int(frac * 30)
                print(f"      {mod:20s} {bar} {frac:.1%}")
        print()

        # Communication
        signals = agent.communication.get_signal_associations()
        active_signals = [s for s in signals if s["times_heard"] > 0]
        if active_signals:
            print(f"  Proto-Language:")
            for sig in active_signals:
                v = sig["valence"]
                meaning_label = "positive" if v > 0.1 else ("negative" if v < -0.1 else "neutral")
                grounded = sig.get("meaning", "ungrounded")
                print(f"    {sig['symbol']}: heard {sig['times_heard']}x, "
                      f"valence={v:+.2f} ({meaning_label}), "
                      f"grounded={grounded}")
            if agent.communication.total_phrases_emitted > 0:
                print(f"    Phrases emitted: {agent.communication.total_phrases_emitted}")
            print()

        # Formal consciousness tests
        results = run_consciousness_tests(agent)
        print_test_report(agent.agent_id, results)
        print()

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Genesis — Digital Consciousness Simulation")
    parser.add_argument("--ticks", type=int, default=50000, help="Maximum simulation ticks")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true", help="Fast mode (no rendering)")
    parser.add_argument("--gui", action="store_true", help="Pygame 2D graphical visualisation")
    parser.add_argument("--3d", action="store_true", dest="gui_3d",
                        help="3D OpenGL visualisation")
    parser.add_argument("--log", type=str, default=None, metavar="DIR",
                        help="Directory to write CSV metric logs")
    parser.add_argument("--neurons", type=int, default=256, help="Neurons per agent")
    parser.add_argument("--no-dreaming", action="store_true",
                        help="Disable dreaming & memory consolidation")
    parser.add_argument("--save", type=str, default=None, metavar="FILE",
                        help="Save checkpoint at end (e.g. checkpoints/sim.gen)")
    parser.add_argument("--load", type=str, default=None, metavar="FILE",
                        help="Load checkpoint to resume from")
    parser.add_argument("--plot", type=str, default=None, metavar="DIR",
                        help="Generate plots from CSV logs in DIR after run")
    parser.add_argument("--evolve", action="store_true",
                        help="Enable evolution: dead agents respawn with mutated weights")
    args = parser.parse_args()

    config = SimulationConfig(
        num_agents=args.agents,
        max_ticks=args.ticks,
        seed=args.seed,
    )

    # Disable dreaming if requested
    if args.no_dreaming:
        from genesis.cognition.dreaming import DreamConfig
        config = dataclasses.replace(config,
                                     dream=DreamConfig(enabled=False))

    # Override neural config if specified
    if args.neurons != 256:
        from genesis.config import NeuralConfig
        motor = 16
        sensory = min(64, args.neurons // 4)
        inter = args.neurons - sensory - motor
        config = SimulationConfig(
            neural=NeuralConfig(
                num_neurons=args.neurons,
                sensory_neurons=sensory,
                motor_neurons=motor,
                interneurons=inter,
            ),
            num_agents=args.agents,
            max_ticks=args.ticks,
            seed=args.seed,
        )

    run_simulation(config, fast_mode=args.fast, log_dir=args.log,
                   gui=args.gui, save_path=args.save, load_path=args.load,
                   evolve=args.evolve, gui_3d=args.gui_3d)

    if args.plot:
        from genesis.analytics.plotter import main as plot_main
        plot_main(args.plot)


if __name__ == "__main__":
    main()
