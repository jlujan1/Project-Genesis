"""Global configuration constants for Project Genesis."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class WorldConfig:
    """Configuration for the simulation sandbox."""
    width: int = 320
    height: int = 240
    tick_rate: float = 0.05  # seconds per simulation tick
    day_cycle_ticks: int = 3600  # ticks per full day/night cycle
    resource_spawn_rate: float = 0.55  # probability of new crystal per tick
    max_resources: int = 800
    decay_rate: float = 0.001  # rate at which things degrade
    obstacle_density: float = 0.035
    # Resource diversity
    toxic_crystal_ratio: float = 0.08   # fraction of crystals that are toxic
    rare_crystal_ratio: float = 0.12    # fraction that are rare/high-value
    rare_crystal_energy: float = 60.0
    toxic_crystal_damage: float = 15.0
    terrain_movement_cost: float = 0.05  # extra energy per unit elevation climbed


@dataclasses.dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent physiology."""
    max_energy: float = 100.0
    max_integrity: float = 100.0
    energy_drain_rate: float = 0.012  # per tick
    move_energy_cost: float = 0.02
    sprint_energy_multiplier: float = 2.0
    collision_damage: float = 0.2
    fall_damage: float = 1.0
    crystal_energy_value: float = 35.0
    pain_threshold: float = 30.0  # below this, pain signals fire
    critical_threshold: float = 10.0
    vision_range: int = 16
    hearing_range: int = 28
    touch_range: int = 1
    # Agent-agent physics
    agent_collision_damage: float = 0.5
    agent_push_force: float = 0.5
    # Homeostasis thresholds (moved from hardcoded values)
    pain_modulation_threshold: float = 0.1
    pleasure_modulation_threshold: float = 0.1
    survival_energy_threshold: float = 0.3
    survival_integrity_threshold: float = 0.3


@dataclasses.dataclass(frozen=True)
class NeuralConfig:
    """Configuration for the spiking neural network."""
    num_neurons: int = 320
    sensory_neurons: int = 64
    motor_neurons: int = 16
    interneurons: int = 240  # num_neurons - sensory - motor
    spike_threshold: float = 1.0
    resting_potential: float = 0.0
    reset_potential: float = -0.5
    leak_rate: float = 0.05
    refractory_period: int = 3  # ticks
    # Hebbian plasticity
    learning_rate: float = 0.01
    weight_decay: float = 0.00005
    max_weight: float = 2.0
    min_weight: float = -1.0
    initial_connectivity: float = 0.20  # denser connections for richer integration
    # STDP parameters
    stdp_a_plus: float = 0.01    # LTP amplitude
    stdp_a_minus: float = 0.006  # LTD amplitude
    stdp_tau: float = 20.0       # STDP time window (ticks)
    # Eligibility traces
    eligibility_decay: float = 0.95  # trace decay per tick


@dataclasses.dataclass(frozen=True)
class WorkspaceConfig:
    """Configuration for the Global Workspace."""
    broadcast_slots: int = 1  # only top packet gets broadcast
    relevance_decay: float = 0.1
    broadcast_strength: float = 0.8
    attention_momentum: float = 0.3  # how much previous focus carries over


@dataclasses.dataclass(frozen=True)
class MemoryConfig:
    """Configuration for memory systems."""
    working_memory_size: int = 16  # max items in working memory
    short_term_buffer_ticks: int = 600  # ~30 seconds at tick_rate
    long_term_capacity: int = 10000
    consolidation_threshold: float = 0.7  # salience needed for LTM storage
    retrieval_similarity_threshold: float = 0.5


@dataclasses.dataclass(frozen=True)
class DreamConfig:
    """Configuration for dreaming & memory consolidation."""
    enabled: bool = True
    replay_episodes_per_night: int = 5
    consolidation_strength: float = 0.02
    prune_threshold: float = 0.005
    dream_noise: float = 0.15
    night_phase_start: float = 0.5
    night_phase_end: float = 0.95
    min_episodes_for_dreaming: int = 3


@dataclasses.dataclass(frozen=True)
class AnalyticsConfig:
    """Configuration for the Phi calculator and dashboard."""
    phi_sample_interval: int = 20  # calculate phi every N ticks
    phi_partition_samples: int = 64  # subsets to test for MIP
    phi_convergence_threshold: float = 0.005  # stop sampling when phi stabilises
    phi_max_samples: int = 128  # upper bound on partition samples
    log_interval: int = 100  # ticks between log entries
    dashboard_refresh: int = 10  # ticks between display refresh
    # Consciousness phase thresholds
    phase_dormant: float = 0.05
    phase_reactive: float = 0.15
    phase_integrative: float = 0.30
    phase_aware: float = 0.50
    phase_self_modeling: float = 0.70


@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    """Master configuration combining all subsystems."""
    world: WorldConfig = dataclasses.field(default_factory=WorldConfig)
    agent: AgentConfig = dataclasses.field(default_factory=AgentConfig)
    neural: NeuralConfig = dataclasses.field(default_factory=NeuralConfig)
    workspace: WorkspaceConfig = dataclasses.field(default_factory=WorkspaceConfig)
    memory: MemoryConfig = dataclasses.field(default_factory=MemoryConfig)
    dream: DreamConfig = dataclasses.field(default_factory=DreamConfig)
    analytics: AnalyticsConfig = dataclasses.field(default_factory=AnalyticsConfig)
    num_agents: int = 4
    max_ticks: int = 50000
    seed: int = 42
