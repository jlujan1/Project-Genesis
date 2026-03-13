"""The Conscious Agent — integrating all subsystems into a unified mind.

Wires together:
  Body (sensors/actuators) → Brain Modules (vision, audio, proprioception, prediction)
  → Global Workspace (competition & broadcast) → SNN (spiking neural network)
  → Motor output → Body (actions) → Environment

Plus: Self-model, homeostasis, memory, communication, and Φ calculation.
"""

from __future__ import annotations

import random

import numpy as np

from genesis.agent.body import (
    ACTION_COLLECT, ACTION_EMIT_SOUND, ACTION_MOVE_DOWN, ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT, ACTION_MOVE_UP, ACTION_NONE, ACTION_SPRINT,
    ACTION_EXAMINE, ACTION_BUILD, ACTION_REST, ACTION_STUDY,
    ACTION_CRAFT, ACTION_SHARE, ACTION_PLANT, ACTION_TEACH,
    NUM_ACTIONS, AgentBody,
)
from genesis.agent.communication import CommunicationSystem
from genesis.agent.cooperation import CooperationSystem
from genesis.agent.homeostasis import HomeostasisEngine
from genesis.analytics.dashboard import AgentMetrics
from genesis.analytics.phi import PhiCalculator
from genesis.cognition.binding import MultiModalBinding
from genesis.cognition.cognitive_map import CognitiveMap
from genesis.cognition.counterfactual import CounterfactualEngine
from genesis.cognition.critical_periods import CriticalPeriods
from genesis.cognition.culture import CulturalTransmission
from genesis.cognition.curiosity import CuriosityEngine
from genesis.cognition.dreaming import DreamEngine
from genesis.cognition.emotions import EmotionEngine, Personality
from genesis.cognition.attention_schema import AttentionSchema
from genesis.cognition.empowerment import EmpowermentEngine
from genesis.cognition.goals import HierarchicalGoals, GOAL_NAMES, SUBGOAL_NAMES
from genesis.cognition.inner_speech import InnerSpeech
from genesis.cognition.memory import EpisodicMemory, WorkingMemory
from genesis.cognition.narrative import NarrativeSelf
from genesis.cognition.prediction import PredictionEngine
from genesis.cognition.self_model import SelfModel
from genesis.cognition.social_learning import SocialLearning
from genesis.cognition.abstraction import SymbolicAbstraction
from genesis.cognition.theory_of_mind import TheoryOfMind
from genesis.cognition.workspace import GlobalWorkspace
from genesis.config import SimulationConfig
from genesis.environment.sandbox import Sandbox
from genesis.neural.modules import (
    AudioModule, PredictionModule, ProprioceptionModule, VisionModule,
    WorkspacePacket,
)
from genesis.neural.spiking import SpikingNeuralNetwork


class ConsciousAgent:
    """A digital agent with the full cognitive architecture.

    Each tick, the agent:
    1. Gathers sensory data from the environment (vision, audio, touch)
    2. Processes it through specialized brain modules (subconscious)
    3. Modules compete for the Global Workspace (consciousness bottleneck)
    4. The winner is broadcast to all modules via the SNN
    5. Motor neurons generate an action
    6. The action is executed in the environment
    7. Pain/pleasure signals modulate learning
    8. Self-model updates
    9. Memories are recorded
    """

    def __init__(self, agent_id: int, config: SimulationConfig,
                 sandbox: Sandbox) -> None:
        self.agent_id = agent_id
        self.config = config
        self.rng = random.Random(config.seed + agent_id)

        # === The Body ===
        self.body = AgentBody(agent_id, config.agent, sandbox)

        # === The Brain (SNN) ===
        self.brain = SpikingNeuralNetwork(config.neural, self.rng)
        self.brain.bootstrap_survival_wiring()

        # === Specialized Modules (The Subconscious) ===
        self.vision = VisionModule(encoding_size=32)
        self.audio = AudioModule(encoding_size=8)
        self.proprioception = ProprioceptionModule(encoding_size=12)
        self.prediction_module = PredictionModule(state_size=16, num_actions=NUM_ACTIONS)

        # === Global Workspace (The Stage) ===
        self.workspace = GlobalWorkspace(config.workspace)

        # === Higher Cognition ===
        self.self_model = SelfModel()
        self.prediction_engine = PredictionEngine(state_size=16, num_actions=NUM_ACTIONS)
        self.working_memory = WorkingMemory(capacity=config.memory.working_memory_size)
        self.episodic_memory = EpisodicMemory(config.memory)

        # === Homeostasis ===
        self.homeostasis = HomeostasisEngine(config.agent)

        # === Personality (unique per agent) ===
        self.personality = Personality()

        # === Emotions (personality-tinted) ===
        self.emotions = EmotionEngine(personality=self.personality)

        # === Attention Schema ===
        self.attention_schema = AttentionSchema()

        # === Communication ===
        self.communication = CommunicationSystem(agent_id)

        # === Cooperation ===
        self.cooperation = CooperationSystem()

        # === Theory of Mind ===
        self.theory_of_mind = TheoryOfMind(self.self_model)

        # === Social Learning ===
        self.social_learning = SocialLearning()

        # === Dreaming ===
        from genesis.cognition.dreaming import DreamConfig as _DC
        dream_cfg = config.dream if hasattr(config, 'dream') else _DC()
        self.dream_engine = DreamEngine(dream_cfg)

        # === Curiosity ===
        self.curiosity = CuriosityEngine()

        # === Inner Speech / Metacognition ===
        self.inner_speech = InnerSpeech()

        # === Spatial Memory / Cognitive Map ===
        self.cognitive_map = CognitiveMap(
            width=config.world.width, height=config.world.height
        )

        # === Hierarchical Goal System ===
        self.goal_system = HierarchicalGoals()

        # === Counterfactual Reasoning ===
        self.counterfactual = CounterfactualEngine()

        # === Cultural Transmission ===
        self.culture = CulturalTransmission()

        # === Critical Periods ===
        self.critical_periods = CriticalPeriods()

        # === Multi-Modal Binding ===
        self.binding = MultiModalBinding()

        # === Narrative Self (Level 23) ===
        self.narrative = NarrativeSelf()

        # === Empowerment (Level 24) ===
        self.empowerment = EmpowermentEngine(num_actions=NUM_ACTIONS)

        # === Symbolic Abstraction (Level 25) ===
        self.abstraction = SymbolicAbstraction()

        # === Civilisation ===
        # Shared state — attached to sandbox as singleton
        from genesis.cognition.civilization import CivilizationState, SpecProfile
        if not hasattr(sandbox, 'civilization'):
            sandbox.civilization = CivilizationState()
        self.civilization = sandbox.civilization
        self.spec_profile = SpecProfile()
        self.civilization.profiles[agent_id] = self.spec_profile

        # === Analytics ===
        self.phi_calculator = PhiCalculator(config.analytics)

        # State tracking
        self.previous_state: np.ndarray = np.zeros(16, dtype=np.float32)
        self.last_action = ACTION_NONE
        self.tick_count = 0

    @property
    def alive(self) -> bool:
        return self.body.alive

    @property
    def position(self):
        return self.body.position

    def tick(self, other_agents: list[ConsciousAgent]) -> None:
        """Run one full cognitive cycle."""
        if not self.alive:
            return

        self.tick_count += 1
        other_bodies = [a.body for a in other_agents if a.agent_id != self.agent_id]
        self._other_agents_cache = other_agents

        # Pre-compute distances to all other agents ONCE per tick
        _nearby_cache: list[tuple[ConsciousAgent, float]] = []
        for o in other_agents:
            if o.agent_id != self.agent_id and o.alive:
                d = self.body.position.distance_to(o.body.position)
                _nearby_cache.append((o, d))
        # Agents within vision range
        vis_range = self.config.agent.vision_range
        _visible = [(o, d) for o, d in _nearby_cache if d < vis_range]
        _visible_ids = [o.agent_id for o, _ in _visible]
        _nearby_count = len(_visible)

        # ─── 1. SENSE: Gather raw sensory data ───
        sensory_data = self.body.get_sensory_input(other_bodies)

        # ─── 2. PROCESS: Run through specialized modules (subconscious) ───
        packets: list[WorkspacePacket] = []

        # Vision processing
        vision_packet = self.vision.process(
            sensory_data["vision"],
            agent_positions=sensory_data["other_agents"],
            own_pos=sensory_data["own_pos"]
        )
        packets.append(vision_packet)

        # Audio processing
        audio_packet = self.audio.process(
            sensory_data["audio"],
            own_pos=sensory_data["own_pos"],
            own_id=self.agent_id
        )
        packets.append(audio_packet)

        # Proprioception (internal body state)
        proprio = sensory_data["proprioception"]
        proprio_packet = self.proprioception.process(
            energy=proprio["energy"],
            max_energy=proprio["max_energy"],
            integrity=proprio["integrity"],
            max_integrity=proprio["max_integrity"],
            velocity=proprio["velocity"],
            position=proprio["position"],
        )
        packets.append(proprio_packet)

        # Prediction error (action-conditional)
        current_state = self.body.get_state_vector()
        pred_packet = self.prediction_module.process(current_state,
                                                      last_action=self.last_action)
        packets.append(pred_packet)

        # Self-model packet (if ego has formed)
        self_encoding = self.self_model.get_encoding()
        self_packet = WorkspacePacket(
            source="self_model",
            data=self_encoding,
            relevance=0.15 + (0.3 if self.self_model.has_ego else 0.0),
            metadata={"has_ego": self.self_model.has_ego,
                      "accuracy": self.self_model.model_accuracy}
        )
        packets.append(self_packet)

        # Memory recall packet
        recalled = self.episodic_memory.recall(current_state, top_k=1)
        if recalled:
            mem_packet = WorkspacePacket(
                source="memory_recall",
                data=recalled[0].state,
                relevance=0.2 + abs(recalled[0].outcome_valence) * 0.3,
                metadata={"valence": recalled[0].outcome_valence}
            )
            packets.append(mem_packet)

        # Theory of Mind packet — observe other agents
        for other, dist in _visible:
            obs = self.theory_of_mind.observe(
                other.agent_id,
                np.array(other.body.position.as_tuple(), dtype=np.float32),
                np.array(other.body.velocity.as_tuple(), dtype=np.float32),
            )
            tom_enc = self.theory_of_mind.get_encoding(other.agent_id)
            tom_packet = WorkspacePacket(
                source="theory_of_mind",
                data=tom_enc,
                relevance=0.15 + obs["model_quality"] * 0.2,
                metadata=obs,
            )
            packets.append(tom_packet)

        # Homeostasis alarm (may override everything)
        alarm = self.homeostasis.process(self.body, self.workspace, self.brain)
        if alarm:
            packets.append(alarm)

        # Curiosity packet
        curiosity_enc = self.curiosity.get_encoding()
        curiosity_packet = WorkspacePacket(
            source="curiosity",
            data=curiosity_enc,
            relevance=self.curiosity.get_workspace_relevance(),
            metadata={"curiosity_level": self.curiosity.curiosity_level,
                      "learning_progress": self.curiosity.learning_progress}
        )
        packets.append(curiosity_packet)

        # Inner speech / metacognition packet
        inner_enc = self.inner_speech.get_encoding()
        inner_packet = WorkspacePacket(
            source="inner_speech",
            data=inner_enc,
            relevance=self.inner_speech.get_workspace_relevance(),
            metadata={"confidence": self.inner_speech.confidence,
                      "reflecting": self.inner_speech.should_reflect()}
        )
        packets.append(inner_packet)

        # Cognitive map / spatial memory packet
        pos_tuple = self.body.position.as_tuple()
        map_enc = self.cognitive_map.get_encoding(pos_tuple, self.tick_count)
        map_packet = WorkspacePacket(
            source="cognitive_map",
            data=map_enc,
            relevance=0.15 + min(0.3, len(self.cognitive_map.cells) / 200.0),
            metadata={"explored": len(self.cognitive_map.cells)}
        )
        packets.append(map_packet)

        # Hierarchical goal packet
        goal_enc = self.goal_system.get_encoding()
        goal_packet = WorkspacePacket(
            source="goals",
            data=goal_enc,
            relevance=0.2 + self.goal_system.goals[
                self.goal_system.active_goal].current_priority * 0.3,
            metadata={"active_goal": GOAL_NAMES[self.goal_system.active_goal]}
        )
        packets.append(goal_packet)

        # Multi-modal binding packet
        bound = self.binding.bind(vision_packet.data,
                                  audio_packet.data,
                                  proprio_packet.data)
        binding_packet = WorkspacePacket(
            source="binding",
            data=self.binding.get_encoding(),
            relevance=self.binding.get_workspace_relevance(),
            metadata={"coherence": self.binding.coherence,
                      "strength": self.binding.binding_strength}
        )
        packets.append(binding_packet)

        # Narrative self packet (Level 23)
        narrative_packet = WorkspacePacket(
            source="narrative",
            data=self.narrative.get_encoding(),
            relevance=self.narrative.get_workspace_relevance(),
            metadata={"identity_strength": self.narrative.identity_strength,
                      "coherence": self.narrative.narrative_coherence}
        )
        packets.append(narrative_packet)

        # Empowerment packet (Level 24)
        empowerment_packet = WorkspacePacket(
            source="empowerment",
            data=self.empowerment.get_encoding(),
            relevance=self.empowerment.get_workspace_relevance(),
            metadata={"empowerment": self.empowerment.empowerment,
                      "agency": self.empowerment.agency_score}
        )
        packets.append(empowerment_packet)

        # Symbolic abstraction packet (Level 25)
        abstraction_packet = WorkspacePacket(
            source="abstraction",
            data=self.abstraction.get_encoding(),
            relevance=self.abstraction.get_workspace_relevance(),
            metadata={"active_concepts": self.abstraction.active_concepts}
        )
        packets.append(abstraction_packet)

        # ─── 2c. ATTENTION SCHEMA: Boost workspace packet relevances ───
        relevance_boosts = self.attention_schema.get_workspace_relevance_boost()
        for pkt in packets:
            if pkt.source in relevance_boosts:
                pkt.relevance = min(1.0, pkt.relevance + relevance_boosts[pkt.source])

        # ─── 3. COMPETE & BROADCAST: Global Workspace selects winner ───
        winner = self.workspace.submit_and_broadcast(packets, self.tick_count)

        # ─── 3b. ATTENTION SCHEMA: Update model of own attention ───
        if winner:
            self.attention_schema.update(winner.source)

        # ─── 4. BROADCAST: Inject into SNN ───
        if winner:
            broadcast_vector = self.workspace.get_broadcast_vector()
            self.brain.inject_broadcast(broadcast_vector,
                                        self.config.workspace.broadcast_strength)

            # Store broadcast in working memory
            self.working_memory.push({
                "tick": self.tick_count,
                "source": winner.source,
                "data": winner.data.copy(),
                "relevance": winner.relevance,
            })

        # ─── 5. INJECT SENSORY DATA: Feed raw sensors into SNN ───
        sensory_input = self._encode_sensory_for_snn(vision_packet, audio_packet,
                                                      proprio_packet)
        self.brain.inject_sensory_input(sensory_input)

        # ─── 5b. CRITICAL PERIODS: Modulate SNN plasticity ───
        effective_lr = self.critical_periods.modulate_snn_learning_rate(
            self.config.neural.learning_rate, self.tick_count
        )
        # Study boost: temporarily increase learning rate when studying
        if self.body.study_boost_active:
            effective_lr *= 1.5
            self.body.study_boost_active = False
        self.brain.learning_rate = effective_lr

        # ─── 6. THINK: Run the SNN one step ───
        motor_output = self.brain.step()

        # ─── 7. DECIDE: Convert motor neuron output to action ───
        action = self._select_action(motor_output, current_state)

        # Handle communication
        if action == ACTION_EMIT_SOUND:
            ctx = current_state
            symbol = self.communication.choose_emission(ctx)
            self.body.emitted_sound_id = symbol
        elif self.body.energy_ratio < 0.3 and _nearby_count > 0:
            # Spontaneous help request when low on energy and others nearby
            request_symbol = self.communication.get_request_symbol()
            if request_symbol is not None:
                self.body.emitted_sound_id = request_symbol

        # Process heard signals from other agents
        heard_symbols: list[int] = []
        for other, dist in _nearby_cache:
            if other.body.emitted_sound_id >= 0 and dist < self.config.agent.hearing_range:
                self.communication.hear_signal(
                    other.body.emitted_sound_id, current_state, self.tick_count
                )
                heard_symbols.append(other.body.emitted_sound_id)

        # Compositional phrase processing — pair consecutive heard symbols
        if len(heard_symbols) >= 2:
            self.communication.hear_phrase(
                heard_symbols[0], heard_symbols[1],
                current_state, self.tick_count
            )

        # Trade response: if heard a "request"-grounded symbol, consider giving energy
        for other, dist in _visible:
            if other.body.emitted_sound_id >= 0 and dist < self.config.agent.hearing_range:
                sym_mem = self.communication.signal_memories[other.body.emitted_sound_id]
                if sym_mem.grounded_meaning == "request":
                    give_amount = self.cooperation.evaluate_request(
                        other.agent_id, self.body.energy_ratio
                    )
                    if give_amount > 0:
                        self.body.energy -= give_amount
                        other.body.energy = min(
                            other.config.agent.max_energy,
                            other.body.energy + give_amount
                        )
                        self.cooperation.share_energy(
                            other.agent_id, give_amount, self.tick_count
                        )
                        other.cooperation.receive_energy(
                            self.agent_id, give_amount, self.tick_count
                        )

        # ─── 8. ACT: Execute the chosen action ───
        self.body.execute_action(action, other_bodies=other_bodies)

        # ─── 8a. CIVILISATION: Record XP and check discoveries ───
        self.spec_profile.record_action(action)
        if self.tick_count % 10 == 0:
            gx, gy = self.body.position.grid_pos()
            sb = self.body.sandbox
            near_berry = gx is not None and (gx, gy) in sb._berry_grid
            near_river = (gx, gy) in sb.rivers
            near_wildlife = any(
                abs(int(w.position.x) - gx) + abs(int(w.position.y) - gy) < 5
                for w in sb.wildlife
            ) if hasattr(sb, 'wildlife') else False
            near_ruin = (gx, gy) in sb._ruin_grid
            near_fungus = (gx, gy) in sb._fungi_grid
            consciousness = getattr(self, '_last_composite', 0.0)
            alive_count = sum(1 for a in other_agents if a.alive) + (1 if self.alive else 0)
            discovered = self.civilization.try_discover(
                self.agent_id, self.tick_count,
                population=alive_count,
                is_night=sb.day_cycle.is_night,
                near_berry=near_berry,
                near_river=near_river,
                near_wildlife=near_wildlife,
                near_ruin=near_ruin,
                near_fungus=near_fungus,
                consciousness=consciousness,
            )
            if discovered is not None:
                from genesis.cognition.civilization import TECH_NAMES
                self.body.pleasure_signal += 0.8
                self.body.recent_reward_ticks = 5

        # ─── 8a2. TEACH resolution ───
        if action == ACTION_TEACH and _visible:
            nearest_other = min(_visible, key=lambda x: x[1])[0]
            # Transfer a random undiscovered tech to the other agent
            my_techs = self.civilization.discovered_techs
            if my_techs:
                other_profile = self.civilization.profiles.get(nearest_other.agent_id)
                if other_profile is not None:
                    # Boost their study/craft/build counts slightly
                    other_profile.study_count = max(
                        other_profile.study_count,
                        self.spec_profile.study_count // 2
                    )
                    other_profile.craft_count = max(
                        other_profile.craft_count,
                        self.spec_profile.craft_count // 2
                    )
                    self.spec_profile.teach_count += 1

        # ─── 8b. SOCIAL LEARNING: Observe other agents' outcomes ───
        for other, dist in _visible:
            self.social_learning.observe(
                tick=self.tick_count,
                other_id=other.agent_id,
                other_pos=np.array(other.body.position.as_tuple(),
                                   dtype=np.float32),
                other_action=other.last_action,
                other_gained_energy=other.body.recent_reward_ticks > 0,
                other_took_damage=other.body.recent_pain_ticks > 0,
            )
        # Apply imitation learning from buffered observations
        # Gate social learning by critical period
        sensory_input = self._encode_sensory_for_snn(vision_packet,
                                                      audio_packet,
                                                      proprio_packet)
        orig_social_lr = self.social_learning.learning_rate
        self.social_learning.learning_rate = self.critical_periods.gate_learning(
            "social", orig_social_lr, self.tick_count
        )
        self.social_learning.imitate(self.brain, sensory_input)
        self.social_learning.learning_rate = orig_social_lr

        # ─── 8c. COOPERATION: Resolve sharing + update bonds ───
        nearby_ids = _visible_ids

        # Resolve energy sharing
        if self.body.sharing_energy > 0 and nearby_ids:
            # Give to nearest agent
            nearest_other = None
            min_d = float("inf")
            for other, d in _visible:
                if d < min_d:
                    min_d = d
                    nearest_other = other
            if nearest_other is not None:
                nearest_other.body.energy = min(
                    nearest_other.config.agent.max_energy,
                    nearest_other.body.energy + self.body.sharing_energy,
                )
                self.cooperation.share_energy(
                    nearest_other.agent_id, self.body.sharing_energy, self.tick_count
                )
                nearest_other.cooperation.receive_energy(
                    self.agent_id, self.body.sharing_energy, self.tick_count
                )
            self.body.sharing_energy = 0.0

        # Update emotional bonds for nearby agents
        self.emotions.update_bonds(
            nearby_ids,
            shared_pleasure=self.body.pleasure_signal,
            shared_pain=self.body.pain_signal,
        )

        # Name nearby predators if visible
        visible_preds = self.body.sandbox.predators.get_visible_predators(
            self.body.position, self.config.agent.vision_range
        )
        for pinfo in visible_preds:
            self.communication.name_entity("predator", str(pinfo["id"]),
                                            current_state)

        # ─── 9. LEARN: Update from consequences ───
        new_state = self.body.get_state_vector()

        # Self-model update
        self_error = self.self_model.update(
            actual_pos=np.array(self.body.position.as_tuple(), dtype=np.float32),
            actual_vel=np.array(self.body.velocity.as_tuple(), dtype=np.float32),
            actual_energy=self.body.energy_ratio,
            actual_integrity=self.body.integrity_ratio,
            last_action=action,
            action_succeeded=self.body.last_action_succeeded
        )

        # Prediction engine update
        pred_error = self.prediction_engine.learn_transition(
            self.previous_state, action, new_state
        )

        # Emotion update
        nearby = _nearby_count
        self.emotions.update(
            pain=self.body.pain_signal,
            pleasure=self.body.pleasure_signal,
            prediction_error=pred_error,
            energy_ratio=self.body.energy_ratio,
            integrity_ratio=self.body.integrity_ratio,
            action_succeeded=self.body.last_action_succeeded,
            nearby_agents=nearby,
        )

        # Predator proximity boosts fear
        if visible_preds:
            closest = min(p["distance"] for p in visible_preds)
            fear_boost = max(0.0, 1.0 - closest / self.config.agent.vision_range) * 0.15
            self.emotions.state.values[0] = min(1.0,
                self.emotions.state.values[0] + fear_boost)

        # Curiosity update — notify of environmental changes
        pos_tuple = self.body.position.as_tuple()
        env_change = self.body.sandbox.environment_change
        if env_change > 0:
            self.curiosity.set_environment_change(env_change)
        # Examine discovery boosts curiosity signal
        if self.body.last_examine_discovery > 0:
            self.curiosity.set_environment_change(
                max(env_change, self.body.last_examine_discovery))
            self.body.last_examine_discovery = 0.0
        self.curiosity.update(pred_error, pos_tuple)

        # Inner speech / metacognition update
        broadcast_src = winner.source if winner else "none"
        self.inner_speech.update(
            broadcast_src, self.body.last_action_succeeded, pred_error
        )

        # Cognitive map update
        nearby_crystals = sum(
            1 for c in sensory_data.get("vision", [])
            if c.get("has_crystal", False) and c.get("distance", 999) < 3
        )
        self.cognitive_map.update(
            pos_tuple, self.tick_count,
            found_crystal=self.body.pleasure_signal > 0.1,
            pain=self.body.pain_signal,
            saw_agent=nearby > 0,
        )

        # Hierarchical goal update — now with spatial context for subgoals
        own_pos = np.array(self.body.position.as_tuple(), dtype=np.float32)
        best_crystal_dir = self.cognitive_map.get_navigation_signal(
            self.body.position.as_tuple(), self.tick_count
        )
        # Find nearest agent position for social subgoals
        nearest_agent_pos = None
        min_agent_dist = float("inf")
        for other, d in _nearby_cache:
            if d < min_agent_dist:
                min_agent_dist = d
                nearest_agent_pos = np.array(
                    other.body.position.as_tuple(), dtype=np.float32)
        # Threat direction from theory of mind or predators
        threat_dir = None
        pred_dir = self.body.sandbox.predators.get_nearest_predator_direction(
            self.body.position, self.config.agent.vision_range
        )
        if pred_dir is not None:
            threat_dir = np.array([pred_dir.x, pred_dir.y], dtype=np.float32)
        elif self.theory_of_mind.get_threat_level() > 0.2:
            for m in self.theory_of_mind.other_models.values():
                rel = m.position_estimate - own_pos
                if float(np.linalg.norm(rel)) > 0.1:
                    threat_dir = rel / float(np.linalg.norm(rel))
                    break

        # ToM influence on goal priorities
        tom_influence = self.theory_of_mind.get_goal_influence()

        # Predator proximity boosts survive goal
        predator_survive_boost = 0.0
        if visible_preds:
            predator_survive_boost = 0.5

        self.goal_system.update(
            energy_ratio=self.body.energy_ratio,
            integrity_ratio=self.body.integrity_ratio,
            pain=self.body.pain_signal,
            curiosity=self.curiosity.curiosity_level,
            loneliness=self.emotions.state["loneliness"],
            nearby_crystals=nearby_crystals,
            nearby_agents=nearby,
            own_position=own_pos,
            best_crystal_dir=best_crystal_dir,
            threat_direction=threat_dir,
            nearest_agent_pos=nearest_agent_pos,
            forage_boost=tom_influence["forage_boost"],
            socialize_boost=tom_influence["socialize_boost"],
            survive_boost=tom_influence["survive_boost"] + predator_survive_boost,
            map_coverage=min(1.0, len(self.cognitive_map.cells)
                            / max(1, self.cognitive_map.cols * self.cognitive_map.rows)),
            has_shelter_nearby=self.body.sandbox.is_sheltered(self.body.position),
            has_agriculture=self.civilization.has_tech(4),
            has_mature_crops=any(c.is_mature for c in self.civilization.crops),
        )

        # Counterfactual reasoning
        cf_results = self.counterfactual.maybe_replay(
            self.tick_count, self.episodic_memory,
            self.prediction_engine, self.self_model, NUM_ACTIONS
        )
        if cf_results:
            self.counterfactual.apply_learning(
                cf_results, self.brain, sensory_input
            )

        # Cultural transmission
        for other, dist in _visible:
                # If the other agent can teach, receive teaching
                if other.culture.can_teach(
                    other.body.ticks_alive, other.self_model.model_accuracy
                ):
                    teaching = other.culture.generate_teaching(
                        other.agent_id, other.cognitive_map,
                        other.body.position.as_tuple()
                    )
                    if teaching is not None:
                        self.culture.receive_teaching(
                            teaching, self.cognitive_map
                        )

        # Record to episodic memory
        valence = self.body.pleasure_signal - self.body.pain_signal
        source = winner.source if winner else "none"
        self.episodic_memory.record(
            tick=self.tick_count,
            state=new_state,
            action=action,
            valence=valence,
            source=source
        )

        # Communication outcome attribution — gated by language critical period
        orig_comm_lr = self.communication.emission_learning_rate
        self.communication.emission_learning_rate = self.critical_periods.gate_learning(
            "language", orig_comm_lr, self.tick_count
        )
        self.communication.attribute_outcome(
            self.body.pleasure_signal, self.body.pain_signal, self.tick_count
        )
        self.communication.emission_learning_rate = orig_comm_lr

        # Narrative self update (Level 23) — periodic autobiography review
        dominant_emotion = self.emotions.get_dominant()
        self.narrative.update(
            tick=self.tick_count,
            episodic_memory=self.episodic_memory,
            energy_ratio=self.body.energy_ratio,
            integrity_ratio=self.body.integrity_ratio,
            pain=self.body.pain_signal,
            pleasure=self.body.pleasure_signal,
            nearby_agents=nearby,
            self_model_accuracy=self.self_model.model_accuracy,
            dominant_emotion=dominant_emotion,
        )

        # Empowerment update (Level 24) — compute agency from prediction engine
        # Throttle to every 3 ticks (expensive: simulates all actions)
        if self.tick_count % 3 == 0:
            self.empowerment.compute(
                self.prediction_engine, new_state, self.self_model
            )

        # Symbolic abstraction update (Level 25) — learn + activate concepts
        self.abstraction.learn(
            state=new_state,
            valence=valence,
            pain=self.body.pain_signal,
            pleasure=self.body.pleasure_signal,
            prediction_error=pred_error,
            nearby_agents=nearby,
            empowerment=self.empowerment.empowerment,
            tick=self.tick_count,
        )
        self.abstraction.activate(new_state, self.tick_count)

        # Update state tracking
        self.previous_state = new_state.copy()
        self.last_action = action

        # Periodic Φ computation (runs inside tick, not just analytics)
        if self.tick_count % self.config.analytics.phi_sample_interval == 0:
            self.phi_calculator.compute_phi(self.brain)
            self.phi_calculator.compute_network_complexity(self.brain)
        if self.tick_count % (self.config.analytics.phi_sample_interval * 4) == 0:
            self.phi_calculator.zipping_test(self.brain)

        # ─── 10. DREAM: Memory consolidation during night ───
        day_phase = self.body.sandbox.day_cycle.phase
        if self.dream_engine.should_dream(
            day_phase, self.episodic_memory.long_term_count
        ):
            self.dream_engine.dream(
                self.episodic_memory, self.brain,
                self.working_memory, self.tick_count,
            )

    def _encode_sensory_for_snn(self, vision: WorkspacePacket,
                                 audio: WorkspacePacket,
                                 proprio: WorkspacePacket) -> np.ndarray:
        """Combine module outputs into sensory neuron input vector."""
        n_sensory = self.config.neural.sensory_neurons
        encoding = np.zeros(n_sensory, dtype=np.float32)

        # Divide sensory neurons among modalities
        v_end = min(32, n_sensory)
        encoding[:v_end] = vision.data[:v_end]

        a_start = 32
        a_end = min(a_start + 8, n_sensory)
        if a_start < n_sensory:
            n = a_end - a_start
            encoding[a_start:a_end] = audio.data[:n]

        p_start = 40
        p_end = min(p_start + 12, n_sensory)
        if p_start < n_sensory:
            n = p_end - p_start
            encoding[p_start:p_end] = proprio.data[:n]

        return encoding

    def _select_action(self, motor_output: np.ndarray,
                       current_state: np.ndarray) -> int:
        """Convert motor neuron activations to a discrete action.

        Uses mental simulation (prediction engine) to evaluate
        options when the motor output is ambiguous.  Now integrates:
        - Theory of Mind social bias
        - Inner Speech metacognitive modifiers
        - Empowerment bias
        - Symbolic abstraction advice
        - Subgoal-aware goal bias
        """
        # If motor output is clear, use it
        if len(motor_output) >= NUM_ACTIONS:
            action_scores = motor_output[:NUM_ACTIONS]
        else:
            action_scores = np.zeros(NUM_ACTIONS, dtype=np.float32)
            action_scores[:len(motor_output)] = motor_output

        # Inner speech decision modifiers
        meta = self.inner_speech.get_decision_modifiers()
        motor_damping = meta["motor_damping"]

        # Apply motor damping from inner speech
        action_scores = action_scores * motor_damping

        # Mental simulation: evaluate each action internally (1 step for speed)
        sim_scores = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a in range(NUM_ACTIONS):
            sim_scores[a] = self.prediction_engine.simulate_action(
                current_state, a, self.self_model, steps=1
            )

        # Combine motor output and simulation
        combined = action_scores * 0.4 + sim_scores * 0.6

        # Add exploration noise (decreases as self-model improves)
        exploration = max(0.12, 0.5 - self.self_model.model_accuracy * 0.4)
        # Curiosity boosts exploration noise
        exploration += self.curiosity.get_exploration_noise()
        # Inner speech uncertainty boosts exploration
        exploration += meta["exploration_boost"]
        noise = np.random.randn(NUM_ACTIONS).astype(np.float32) * exploration
        combined += noise

        # Goal system motor bias (now subgoal-aware)
        own_pos = np.array(self.body.position.as_tuple(), dtype=np.float32)
        combined += self.goal_system.get_motor_bias(NUM_ACTIONS,
                                                     own_position=own_pos)

        # Theory of Mind social bias
        combined += self.theory_of_mind.get_social_motor_bias(
            own_pos, NUM_ACTIONS)

        # Communication grounding bias
        combined += self.communication.get_communication_motor_bias(NUM_ACTIONS)

        # Counterfactual regret bias
        combined += self.counterfactual.get_regret_bias(NUM_ACTIONS)

        # Attention schema bias
        combined += self.attention_schema.get_action_bias(NUM_ACTIONS)

        # Empowerment bias (avoid getting stuck)
        combined += self.empowerment.get_exploration_bias(NUM_ACTIONS)

        # Inner speech strategy override during reflection
        override = self.inner_speech.get_strategy_override(
            self.goal_system.active_goal, NUM_ACTIONS)
        if override is not None:
            combined += override

        # Symbolic abstraction advice
        advice = self.abstraction.reason(current_state)
        if advice["urgency"] > 0.3:
            # Danger detected symbolically — boost movement
            n_move = min(4, NUM_ACTIONS)
            combined[:n_move] += advice["urgency"] * 0.15

        # Inactivity penalty: discourage NONE unless survival is urgent
        survival_priority = self.goal_system.goals[0].current_priority
        if survival_priority < 0.5:
            combined[ACTION_NONE] -= 0.15  # penalize doing nothing

        # Cooperation bias — trusted allies nearby boost social actions
        nearby_agent_ids = [
            o.agent_id for o in
            (getattr(self, '_other_agents_cache', None) or [])
            if o.agent_id != self.agent_id and o.alive
            and self.body.position.distance_to(o.body.position)
            < self.config.agent.vision_range
        ]
        combined += self.cooperation.get_cooperation_bias(
            NUM_ACTIONS, nearby_agent_ids)

        # Predator flee bias — if predator nearby, strongly bias fleeing
        pred_dir = self.body.sandbox.predators.get_nearest_predator_direction(
            self.body.position, self.config.agent.vision_range
        )
        if pred_dir is not None:
            # Flee opposite to predator
            if pred_dir.x > 0.3:
                combined[ACTION_MOVE_LEFT] += 0.4  # go opposite
            elif pred_dir.x < -0.3:
                combined[ACTION_MOVE_RIGHT] += 0.4
            if pred_dir.y > 0.3:
                combined[ACTION_MOVE_UP] += 0.4
            elif pred_dir.y < -0.3:
                combined[ACTION_MOVE_DOWN] += 0.4
            combined[ACTION_SPRINT] += 0.3  # sprint away

        action = int(np.argmax(combined))

        return action

    def compute_analytics(self) -> AgentMetrics:
        """Gather all metrics for the analytics dashboard."""
        assessment = self.phi_calculator.get_consciousness_assessment(
            self_model_accuracy=self.self_model.model_accuracy,
            attention_accuracy=self.attention_schema.schema_accuracy,
            metacognitive_confidence=self.inner_speech.confidence,
            binding_coherence=self.binding.coherence,
            empowerment=self.empowerment.empowerment,
            narrative_identity=self.narrative.identity_strength,
            curiosity_level=self.curiosity.curiosity_level,
        )
        ws = self.workspace.get_broadcast_summary()
        homeo = self.homeostasis.get_state()

        return AgentMetrics(
            agent_id=self.agent_id,
            ticks_alive=self.body.ticks_alive,
            energy=self.body.energy,
            integrity=self.body.integrity,
            phi=assessment["phi"],
            complexity=assessment["complexity"],
            reverberation=assessment["reverberation"],
            composite_consciousness=assessment["composite_score"],
            consciousness_phase=assessment["phase"],
            workspace_source=ws["current_source"],
            workspace_relevance=ws["current_relevance"],
            broadcast_distribution=ws["broadcast_distribution"],
            prediction_error=self.prediction_engine.average_error,
            self_model_accuracy=self.self_model.model_accuracy,
            has_ego=self.self_model.has_ego,
            active_connections=self.brain.get_active_connections(),
            pain=homeo["cumulative_pain"],
            pleasure=homeo["cumulative_pleasure"],
            survival_urgency=homeo["survival_urgency"],
            dream_cycles=self.dream_engine.stats.total_dream_cycles,
            is_dreaming=self.dream_engine.stats.is_dreaming,
            dominant_emotion=self.emotions.get_dominant(),
            emotional_valence=self.emotions.get_valence(),
            attention_focus=self.attention_schema.current_focus,
            attention_schema_accuracy=self.attention_schema.schema_accuracy,
            curiosity_level=self.curiosity.curiosity_level,
            metacognitive_confidence=self.inner_speech.confidence,
            active_goal=GOAL_NAMES[self.goal_system.active_goal],
            map_coverage=len(self.cognitive_map.cells) / max(
                1, self.cognitive_map.cols * self.cognitive_map.rows),
            binding_strength=self.binding.binding_strength,
            counterfactual_regrets=self.counterfactual.total_regrets,
            teachings_received=self.culture.teachings_received,
            critical_period_lr=self.critical_periods.modulate_snn_learning_rate(
                1.0, self.tick_count),
            empowerment=self.empowerment.empowerment,
            agency_score=self.empowerment.agency_score,
            identity_strength=self.narrative.identity_strength,
            narrative_coherence=self.narrative.narrative_coherence,
            active_concepts=len(self.abstraction.active_concepts),
            developmental_stage=self.critical_periods.get_developmental_stage(
                self.tick_count),
            active_subgoal=SUBGOAL_NAMES[self.goal_system.active_subgoal.subgoal_type],
            binding_conflicts=self.binding.conflict_count,
            grounded_symbols=self.communication.get_language_summary()[
                "grounded_symbols"],
            mood_valence=self.emotions.get_summary().get("mood_valence", 0.0),
            mood_arousal=self.emotions.get_summary().get("mood_arousal", 0.0),
            tool_count=self.body.tools.get_summary()["tool_count"],
            cooperation_partners=self.cooperation.get_summary()["partners"],
            vocabulary_size=self.communication.get_language_summary().get(
                "vocabulary_size", 0),
            named_entities=self.communication.get_language_summary().get(
                "named_entities", 0),
        )
