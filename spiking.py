"""Spiking Neural Network with STDP plasticity.

Implements a biologically-inspired network where neurons fire discrete spikes
when their membrane potential crosses a threshold. Connections strengthen
via Spike-Timing Dependent Plasticity (STDP) — causal pre→post timing
potentiates, anti-causal post→pre timing depresses.

All hot-path operations are vectorised with NumPy for performance.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from genesis.config import NeuralConfig


class SpikingNeuralNetwork:
    """A network of spiking neurons with plastic synapses.

    Architecture:
        - Sensory neurons receive external input from environment sensors
        - Interneurons process and integrate information
        - Motor neurons drive agent actions
        - All connections are weighted and subject to Hebbian plasticity

    Internal state is stored in contiguous NumPy arrays for
    vectorised computation.
    """

    def __init__(self, config: NeuralConfig, rng: random.Random | None = None) -> None:
        self.config = config
        self.rng = rng or random.Random()
        self.np_rng = np.random.RandomState(self.rng.randint(0, 2**31))

        self.num_neurons = config.num_neurons

        # Neuron state arrays (replaces per-object Neuron list)
        self.potentials = np.zeros(self.num_neurons, dtype=np.float32)
        self.refractory_timers = np.zeros(self.num_neurons, dtype=np.int32)

        # Weight matrix (from -> to). Sparse initialization.
        self.weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        self._initialize_connections()

        # Spike history — circular buffer for O(1) operations
        self.history_length = 50
        self.spike_history: deque[np.ndarray] = deque(maxlen=self.history_length)

        # STDP: track last spike time per neuron (for timing-dependent plasticity)
        self.last_spike_time = np.full(self.num_neurons, -1000, dtype=np.int32)
        self.global_tick: int = 0

        # Eligibility traces for reward-modulated learning
        self.eligibility_traces = np.zeros(
            (self.num_neurons, self.num_neurons), dtype=np.float32)

        # Reverb buffer — broadcasts echo for several ticks with decay
        self.reverb_buffer: deque[np.ndarray] = deque(maxlen=8)
        self.reverb_decay: float = 0.6

        # Index helpers
        self.sensory_indices = list(range(config.sensory_neurons))
        self.motor_start = config.sensory_neurons
        self.motor_indices = list(range(self.motor_start, self.motor_start + config.motor_neurons))
        self.inter_start = self.motor_start + config.motor_neurons
        self.inter_indices = list(range(self.inter_start, self.num_neurons))

        # Mutable learning rate override (defaults to config value)
        self._learning_rate_override: float | None = None

    @property
    def learning_rate(self) -> float:
        if self._learning_rate_override is not None:
            return self._learning_rate_override
        return self.config.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._learning_rate_override = value

    def _initialize_connections(self) -> None:
        """Create sparse initial connectivity (vectorised)."""
        n = self.num_neurons
        conn_prob = self.config.initial_connectivity

        # Random connectivity mask (no self-connections)
        mask = self.np_rng.random((n, n)).astype(np.float32) < conn_prob
        np.fill_diagonal(mask, False)

        # Random weights, mostly excitatory
        w = self.np_rng.uniform(0.01, 0.3, size=(n, n)).astype(np.float32)

        # ~20 % inhibitory
        inhibitory = self.np_rng.random((n, n)) < 0.2
        w[inhibitory] *= -0.5

        self.weights = w * mask

        # Boost recurrent inter→inter connectivity for reverberation
        inter_s = self.config.sensory_neurons + self.config.motor_neurons
        n_inter = n - inter_s
        # Add extra inter→inter connections (raise effective connectivity to ~45%)
        extra_mask = self.np_rng.random((n_inter, n_inter)).astype(np.float32) < 0.25
        np.fill_diagonal(extra_mask, False)
        extra_w = self.np_rng.uniform(0.05, 0.25, size=(n_inter, n_inter)).astype(np.float32)
        self.weights[inter_s:, inter_s:] += extra_mask * extra_w
        # Also boost existing inter connections
        inter_existing = np.abs(self.weights[inter_s:, inter_s:]) > 0.01
        self.weights[inter_s:, inter_s:] += inter_existing * 0.22

        # Bridge neurons — 8 hub neurons at end of interneuron pool
        # with high cross-region connectivity for information integration
        num_bridge = 8
        bridge_start = n - num_bridge
        sens_end = self.config.sensory_neurons
        motor_end = sens_end + self.config.motor_neurons

        # Vectorised bridge connectivity
        for b in range(bridge_start, n):
            # Bridge ↔ sensory (50%/30% connectivity)
            s2b_mask = self.np_rng.random(sens_end) < 0.50
            self.weights[:sens_end, b] += s2b_mask * self.np_rng.uniform(0.10, 0.25, sens_end)
            b2s_mask = self.np_rng.random(sens_end) < 0.30
            self.weights[b, :sens_end] += b2s_mask * self.np_rng.uniform(0.05, 0.15, sens_end)
            # Bridge ↔ motor (40% connectivity)
            n_motors = motor_end - sens_end
            m2b_mask = self.np_rng.random(n_motors) < 0.40
            self.weights[sens_end:motor_end, b] += m2b_mask * self.np_rng.uniform(0.08, 0.20, n_motors)
            b2m_mask = self.np_rng.random(n_motors) < 0.40
            self.weights[b, sens_end:motor_end] += b2m_mask * self.np_rng.uniform(0.08, 0.20, n_motors)
        # Bridge ↔ bridge (strong mutual connectivity)
        for b in range(bridge_start, n):
            bb_mask = self.np_rng.random(num_bridge) < 0.70
            bb_mask[b - bridge_start] = False  # no self-connection
            self.weights[b, bridge_start:n] += bb_mask * self.np_rng.uniform(0.15, 0.30, num_bridge)

    def bootstrap_survival_wiring(self) -> None:
        """Add initial sensory-to-motor biases so agents forage from tick 1.

        Vision encoding layout (sensory neurons 0-31):
          [11] crystal to right, [12] crystal to left,
          [13] crystal below,   [14] crystal above,
          [15] crystal proximity.
          [21] berry right, [22] berry left, [23] berry proximity.
          [24] fungus right, [25] fungus left, [26] fungus proximity.
          [27] ruin right, [28] ruin left, [29] ruin proximity.
        Motor neurons (starting at motor_start):
          0=NONE, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=COLLECT.
        """
        s = 0   # sensory offset
        m = self.motor_start
        bias = 0.35

        # Crystal direction -> matching movement motor
        self.weights[s + 11, m + 4] += bias   # crystal right -> MOVE_RIGHT
        self.weights[s + 12, m + 3] += bias   # crystal left  -> MOVE_LEFT
        self.weights[s + 13, m + 2] += bias   # crystal below -> MOVE_DOWN
        self.weights[s + 14, m + 1] += bias   # crystal above -> MOVE_UP

        # Crystal proximity -> COLLECT
        self.weights[s + 15, m + 5] += bias

        # Berry direction -> matching movement (weaker than crystal)
        berry_bias = bias * 0.7
        self.weights[s + 21, m + 4] += berry_bias   # berry right -> MOVE_RIGHT
        self.weights[s + 22, m + 3] += berry_bias   # berry left  -> MOVE_LEFT
        # Berry proximity -> COLLECT
        self.weights[s + 23, m + 5] += berry_bias

        # Fungus direction -> matching movement (weaker, night food)
        fungus_bias = bias * 0.5
        self.weights[s + 24, m + 4] += fungus_bias  # fungus right -> MOVE_RIGHT
        self.weights[s + 25, m + 3] += fungus_bias  # fungus left  -> MOVE_LEFT
        # Fungus proximity -> COLLECT
        self.weights[s + 26, m + 5] += fungus_bias

        # Ruin direction -> EXAMINE/STUDY (curiosity-driven)
        ruin_bias = bias * 0.4
        self.weights[s + 27, m + 4] += ruin_bias    # ruin right -> MOVE_RIGHT
        self.weights[s + 28, m + 3] += ruin_bias    # ruin left  -> MOVE_LEFT
        # Ruin proximity -> EXAMINE and STUDY
        self.weights[s + 29, m + 8] += ruin_bias    # ruin proximity -> EXAMINE
        self.weights[s + 29, m + 11] += ruin_bias   # ruin proximity -> STUDY

        # Obstacle proximity (sensor 1) -> inhibit movement motors
        for motor_offset in (1, 2, 3, 4, 7):  # UP, DOWN, LEFT, RIGHT, SPRINT
            self.weights[s + 1, m + motor_offset] -= bias * 0.3

        # --- New action bootstrap wiring ---
        # Curiosity (general sensory novelty) -> EXAMINE (motor 8)
        # Use agent-energy sensor region (sensor ~6 for energy ratio)
        self.weights[s + 15, m + 8] += bias * 0.3  # crystal proximity -> examine

        # High energy  -> BUILD (motor 9)
        # Sensor 6 = energy ratio in body_state encoding
        self.weights[s + 6, m + 9] += bias * 0.25  # energy -> build

        # Low integrity / pain -> REST (motor 10)
        # Sensor 7 = integrity ratio in body_state encoding
        # Low integrity means we want to rest — use inhibitory inverse logic
        # Wire self-model sensor to rest: sensor 8 = pain signal
        self.weights[s + 8, m + 10] += bias * 0.3  # pain -> rest

        # Self-model signal -> STUDY (motor 11)
        self.weights[s + 9, m + 11] += bias * 0.2  # self-awareness -> study

        # High energy -> CRAFT (motor 12) — craft tools when energy is good
        self.weights[s + 6, m + 12] += bias * 0.15  # energy -> craft

        # Social proximity -> SHARE (motor 13) — share when agents nearby
        # Use audio/agent detection sensors
        self.weights[s + 3, m + 13] += bias * 0.1  # nearby agent signal -> share

        # Berry proximity -> PLANT (motor 14) — plant near food sources
        self.weights[s + 23, m + 14] += bias * 0.2  # berry proximity -> plant

        # Social proximity -> TEACH (motor 15) — teach when agents nearby
        self.weights[s + 3, m + 15] += bias * 0.1  # nearby agent signal -> teach

    def inject_sensory_input(self, inputs: np.ndarray) -> None:
        """Inject external data into sensory neurons (vectorised)."""
        n_sens = min(len(inputs), self.config.sensory_neurons)
        active = self.refractory_timers[:n_sens] <= 0
        self.potentials[:n_sens] += inputs[:n_sens] * active

    def inject_broadcast(self, signal: np.ndarray, strength: float = 0.5) -> None:
        """Inject a Global-Workspace broadcast into interneurons (vectorised).

        Also stores the signal in a reverb buffer so it echoes for
        several subsequent ticks with exponential decay.
        """
        n = min(len(signal), len(self.inter_indices))
        s = self.inter_start
        active = self.refractory_timers[s:s + n] <= 0
        self.potentials[s:s + n] += signal[:n] * strength * active
        # Store in reverb buffer for later echo injection
        self.reverb_buffer.append(signal[:n].copy() * strength)

    def step(self) -> np.ndarray:
        """Run one tick of the network (vectorised). Returns motor neuron activations."""
        self.global_tick += 1
        spike_vector = np.zeros(self.num_neurons, dtype=np.float32)

        # Phase 0: Re-inject decaying broadcast echoes (reverberation)
        if self.reverb_buffer:
            s = self.inter_start
            for i, echo in enumerate(self.reverb_buffer):
                age = len(self.reverb_buffer) - i  # oldest = largest age
                decay = self.reverb_decay ** age
                n = min(len(echo), len(self.inter_indices))
                can_recv = self.refractory_timers[s:s + n] <= 0
                self.potentials[s:s + n] += echo[:n] * decay * can_recv

        # Phase 1: Refractory bookkeeping
        in_refractory = self.refractory_timers > 0
        self.refractory_timers[in_refractory] -= 1

        # Spontaneous noise — prevents SNN from collapsing to limit cycles
        noise = self.np_rng.randn(len(self.inter_indices)).astype(np.float32) * 0.05
        s_inter = self.inter_start
        avail = self.refractory_timers[s_inter:] <= 0
        self.potentials[s_inter:] += noise * avail

        # Neurons free to be evaluated
        active = ~in_refractory

        # Threshold check
        fires = active & (self.potentials >= self.config.spike_threshold)
        spike_vector[fires] = 1.0
        self.potentials[fires] = self.config.reset_potential
        self.refractory_timers[fires] = self.config.refractory_period

        # Update last spike times
        self.last_spike_time[fires] = self.global_tick

        # Leak for active neurons that did NOT fire
        leak_mask = active & ~fires
        self.potentials[leak_mask] *= 1.0 - self.config.leak_rate

        # Phase 2: Propagate spikes through synapses
        if spike_vector.any():
            incoming = spike_vector @ self.weights  # (N,)@(N,N)->(N,)
            can_receive = self.refractory_timers <= 0
            self.potentials[can_receive] += incoming[can_receive]

        # Phase 3: STDP plasticity
        self._apply_stdp(spike_vector)

        # Decay eligibility traces
        self.eligibility_traces *= self.config.eligibility_decay

        # Record spike history
        self.spike_history.append(spike_vector.copy())

        # Motor output (clipped to >= 0)
        m_start = self.motor_start
        m_end = m_start + len(self.motor_indices)
        return np.maximum(0.0, self.potentials[m_start:m_end]).copy()

    def _apply_stdp(self, spike_vector: np.ndarray) -> None:
        """Spike-Timing Dependent Plasticity (fully vectorised).

        Pre fires before post → potentiation (causal, LTP)
        Post fires before pre → depression (anti-causal, LTD)

        Also maintains eligibility traces for reward-modulated learning.
        """
        fired_mask = spike_vector > 0.5
        if not fired_mask.any():
            self.weights *= 1.0 - self.config.weight_decay
            np.clip(self.weights, self.config.min_weight, self.config.max_weight,
                    out=self.weights)
            return

        lr = self.learning_rate
        tau = self.config.stdp_tau
        a_plus = self.config.stdp_a_plus
        a_minus = self.config.stdp_a_minus

        # Time since last spike for all neurons
        dt = (self.global_tick - self.last_spike_time).astype(np.float32)

        # Vectorised LTP: for all fired post neurons at once
        # dt_pre[i] = time since neuron i last fired (pre-synaptic timing)
        # We need connections [:, fired] that are non-zero and dt in window
        has_conn = self.weights[:, fired_mask] != 0          # (N, n_fired)
        dt_col = np.broadcast_to(dt[:, None], has_conn.shape)  # (N, n_fired)
        in_window = (dt_col > 0) & (dt_col < tau) & has_conn
        # Zero out self-connections (diagonal entries for fired neurons)
        fired_indices = np.where(fired_mask)[0]
        for i, fi in enumerate(fired_indices):
            in_window[fi, i] = False
        dw_ltp = np.where(in_window, a_plus * np.exp(-dt_col / tau), 0.0).astype(np.float32)
        self.weights[:, fired_mask] += lr * dw_ltp
        self.eligibility_traces[:, fired_mask] += dw_ltp

        # Vectorised LTD: for all fired pre neurons at once
        has_conn_r = self.weights[fired_mask, :] != 0        # (n_fired, N)
        dt_row = np.broadcast_to(dt[None, :], has_conn_r.shape)  # (n_fired, N)
        in_window_r = (dt_row > 0) & (dt_row < tau) & has_conn_r
        for i, fi in enumerate(fired_indices):
            in_window_r[i, fi] = False
        dw_ltd = np.where(in_window_r, a_minus * np.exp(-dt_row / tau), 0.0).astype(np.float32)
        self.weights[fired_mask, :] -= lr * dw_ltd

        # Weight decay (pruning unused connections)
        self.weights *= 1.0 - self.config.weight_decay

        # Clip weights
        np.clip(self.weights, self.config.min_weight, self.config.max_weight,
                out=self.weights)

    def get_firing_rates(self) -> np.ndarray:
        """Calculate average firing rate per neuron over recent history."""
        if not self.spike_history:
            return np.zeros(self.num_neurons, dtype=np.float32)
        history = np.array(self.spike_history)
        return history.mean(axis=0)

    def get_active_connections(self) -> int:
        """Count non-negligible connections."""
        return int(np.count_nonzero(np.abs(self.weights) > 0.01))

    def get_connection_matrix(self) -> np.ndarray:
        """Return the weight matrix for analysis."""
        return self.weights.copy()

    def get_state_vector(self) -> np.ndarray:
        """Return the current membrane potentials of all neurons."""
        return self.potentials.copy()
