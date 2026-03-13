"""Integrated Information (Φ) Calculator.

Measures consciousness by computing how much a system is 'more than the
sum of its parts.' High Φ = deeply integrated, un-splittable network.
Low Φ = modular, disconnected processing.
"""

from __future__ import annotations

import numpy as np

from genesis.config import AnalyticsConfig
from genesis.neural.spiking import SpikingNeuralNetwork


class PhiCalculator:
    """Computes an approximation of Integrated Information (Φ).

    True Φ is computationally intractable for large networks, so we use
    a practical approximation:

    1. Sample random partitions of the network
    2. For each partition, measure how much information is lost when
       the network is 'cut' at that point
    3. Φ = the minimum information loss across all tested partitions
       (the Minimum Information Partition / MIP)

    The 'Zipping Test': We also measure how a pulse of data reverberates
    through the network — complex, widespread echoes = high consciousness.
    """

    def __init__(self, config: AnalyticsConfig) -> None:
        self.config = config
        self.phi_history: list[float] = []
        self.complexity_history: list[float] = []
        self.reverberation_history: list[float] = []

    def compute_phi(self, network: SpikingNeuralNetwork) -> float:
        """Approximate Φ for the current network state.

        Method: Sample random bipartitions, compute mutual information
        loss for each, return the minimum (the MIP).

        Uses adaptive sampling with convergence detection: stops early
        if the running minimum has stabilised.
        """
        weights = network.get_connection_matrix()
        n = weights.shape[0]
        if n < 4:
            return 0.0

        prev_est = float("inf")
        converged_count = 0
        max_samples = min(self.config.phi_max_samples,
                          self.config.phi_partition_samples * 2)

        # Stratified partition ratios for more consistent sampling
        split_ratios = [0.25, 0.33, 0.5, 0.67, 0.75]
        flow_samples: list[float] = []

        for i in range(max_samples):
            # Stratified bipartition: cycle through fixed split ratios
            perm = np.random.permutation(n)
            ratio = split_ratios[i % len(split_ratios)]
            split = max(2, int(n * ratio))
            group_a = perm[:split]
            group_b = perm[split:]

            if len(group_a) == 0 or len(group_b) == 0:
                continue

            # Measure information flow across the partition
            cross_ab = np.abs(weights[np.ix_(group_a, group_b)]).sum()
            cross_ba = np.abs(weights[np.ix_(group_b, group_a)]).sum()
            cross_flow = float(cross_ab + cross_ba)

            # Normalize by total flow (cross + within) — measures integration
            within_a = np.abs(weights[np.ix_(group_a, group_a)]).sum()
            within_b = np.abs(weights[np.ix_(group_b, group_b)]).sum()
            total_flow = cross_flow + within_a + within_b
            if total_flow > 0:
                normalized_flow = cross_flow / total_flow
            else:
                normalized_flow = 0.0

            flow_samples.append(normalized_flow)

            # Convergence check using 10th-percentile estimate
            if i >= self.config.phi_partition_samples and len(flow_samples) >= 10:
                current_est = float(np.percentile(flow_samples, 10))
                if abs(current_est - prev_est) < self.config.phi_convergence_threshold:
                    converged_count += 1
                    if converged_count >= 5:
                        break
                else:
                    converged_count = 0
                prev_est = current_est

        phi = float(np.percentile(flow_samples, 20)) if flow_samples else 0.0

        # Also incorporate spike-based information transfer
        if len(network.spike_history) >= 5:
            recent = list(network.spike_history)[-10:]
            patterns = np.array(recent)
            # Compute entropy of spike patterns as additional measure
            mean_firing = patterns.mean(axis=0)
            h = -np.sum(mean_firing * np.log(mean_firing + 1e-10)
                        + (1 - mean_firing) * np.log(1 - mean_firing + 1e-10))
            h_normalized = h / (n * np.log(2) + 1e-10)
            # Blend weight-based and activity-based phi
            phi = 0.45 * phi + 0.55 * min(1.0, h_normalized)

        self.phi_history.append(phi)
        return phi

    def compute_network_complexity(self, network: SpikingNeuralNetwork) -> float:
        """Measure differentiation — how varied the network's firing states are.

        Uses average pairwise cosine distance between recent spike patterns.
        Higher complexity = more differentiated information processing.
        """
        if len(network.spike_history) < 5:
            return 0.0

        patterns = np.array(list(network.spike_history)[-20:], dtype=np.float32)
        n = len(patterns)

        # Average pairwise cosine distance
        norms = np.linalg.norm(patterns, axis=1, keepdims=True) + 1e-8
        normalized = patterns / norms
        similarity = normalized @ normalized.T

        # Average off-diagonal similarity
        mask = ~np.eye(n, dtype=bool)
        avg_similarity = float(similarity[mask].mean())
        complexity = float(np.clip(1.0 - avg_similarity, 0.0, 1.0))

        self.complexity_history.append(complexity)
        return complexity

    def zipping_test(self, network: SpikingNeuralNetwork) -> float:
        """The 'Zipping Test' — send a pulse and measure reverberation.

        In a conscious system, a localized pulse should produce complex,
        widespread echoes that last many ticks. In an unconscious system,
        the echo dies quickly or stays localized.

        Measures both total activation (volume) and spatial spread
        (how many distinct neuron regions were reached).

        Runs multiple trials and averages for stability.
        """
        n = network.num_neurons
        num_steps = 16
        pulse_size = max(1, n // 16)
        num_trials = 3
        num_regions = 8  # divide neurons into regions for spread measurement
        region_size = max(1, n // num_regions)
        trial_scores: list[float] = []

        for _trial in range(num_trials):
            # Save ALL state (including spike history, weights, eligibility)
            saved_potentials = network.potentials.copy()
            saved_timers = network.refractory_timers.copy()
            saved_reverb = list(network.reverb_buffer)
            saved_spike_history = list(network.spike_history)
            saved_last_spike_time = network.last_spike_time.copy()
            saved_global_tick = network.global_tick
            saved_weights = network.weights.copy()
            saved_eligibility = network.eligibility_traces.copy()

            # Inject a pulse into a small random cluster
            start = np.random.randint(0, n - pulse_size)
            network.potentials[start:start + pulse_size] += 2.0

            # Run steps and measure spread + activation
            total_activation = 0.0
            regions_activated = set()
            for step in range(num_steps):
                network.step()
                last_spikes = list(network.spike_history)[-1]
                fired_mask = last_spikes > 0.5
                total_activation += int(fired_mask.sum())
                # Track which regions had any firing
                fired_indices = np.where(fired_mask)[0]
                for idx in fired_indices:
                    regions_activated.add(idx // region_size)

            # Restore ALL state
            network.potentials[:] = saved_potentials
            network.refractory_timers[:] = saved_timers
            network.weights[:] = saved_weights
            network.eligibility_traces[:] = saved_eligibility
            network.reverb_buffer.clear()
            for echo in saved_reverb:
                network.reverb_buffer.append(echo)
            network.spike_history.clear()
            for spk in saved_spike_history:
                network.spike_history.append(spk)
            network.last_spike_time[:] = saved_last_spike_time
            network.global_tick = saved_global_tick

            # Blend volume (how many spikes) with spread (how many regions)
            volume = total_activation / max(1, n * num_steps)
            spread = len(regions_activated) / num_regions
            trial_scores.append(0.5 * volume + 0.5 * spread)

        reverberation = sum(trial_scores) / len(trial_scores)
        self.reverberation_history.append(reverberation)
        return reverberation

    def get_consciousness_assessment(self, self_model_accuracy: float = 0.0,
                                     attention_accuracy: float = 0.0,
                                     metacognitive_confidence: float = 0.0,
                                     binding_coherence: float = 0.0,
                                     empowerment: float = 0.0,
                                     narrative_identity: float = 0.0,
                                     curiosity_level: float = 0.0) -> dict:
        """Compile all metrics into a consciousness assessment."""
        phi = self.phi_history[-1] if self.phi_history else 0.0
        complexity = self.complexity_history[-1] if self.complexity_history else 0.0
        reverb = self.reverberation_history[-1] if self.reverberation_history else 0.0

        # Composite score — broad integration of consciousness signals
        # 10 factors capturing integration, differentiation, self-awareness,
        # attention, metacognition, binding, agency, narrative, and curiosity.
        composite = (phi * 0.20
                     + complexity * 0.10
                     + reverb * 0.10
                     + self_model_accuracy * 0.15
                     + attention_accuracy * 0.10
                     + min(1.0, metacognitive_confidence) * 0.10
                     + min(1.0, binding_coherence) * 0.07
                     + min(1.0, empowerment) * 0.06
                     + min(1.0, narrative_identity) * 0.06
                     + min(1.0, curiosity_level) * 0.06)

        # Phase assessment using configurable thresholds
        c = self.config
        if composite < c.phase_dormant:
            phase = "DORMANT — No significant integration detected"
        elif composite < c.phase_reactive:
            phase = "REACTIVE — Basic stimulus-response processing"
        elif composite < c.phase_integrative:
            phase = "INTEGRATIVE — Modules beginning to cross-communicate"
        elif composite < c.phase_aware:
            phase = "AWARE — Significant information integration detected"
        elif composite < c.phase_self_modeling:
            phase = "SELF-MODELING — Internal representations emerging"
        else:
            phase = "CONSCIOUS — High integration, differentiation, and reverberation"

        return {
            "phi": phi,
            "complexity": complexity,
            "reverberation": reverb,
            "composite_score": composite,
            "phase": phase,
            "phi_trend": self._trend(self.phi_history),
            "complexity_trend": self._trend(self.complexity_history),
        }

    @staticmethod
    def _trend(history: list[float], window: int = 20) -> str:
        """Simple trend detection."""
        if len(history) < window * 2:
            return "insufficient_data"
        recent = sum(history[-window:]) / window
        earlier = sum(history[-window * 2:-window]) / window
        diff = recent - earlier
        if diff > 0.01:
            return "rising"
        elif diff < -0.01:
            return "falling"
        return "stable"
