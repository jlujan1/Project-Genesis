"""Reward-modulated plasticity with eligibility traces.

Provides reward and pain modulated plasticity that uses STDP eligibility
traces for temporal credit assignment. Instead of immediately reinforcing
all co-active pairs, the eligibility trace remembers which synapses were
recently active (via STDP events) and modulates them when reward/pain arrives.
"""

from __future__ import annotations

import numpy as np

from genesis.neural.spiking import SpikingNeuralNetwork


def apply_reward_signal(network: SpikingNeuralNetwork, reward: float) -> None:
    """Strengthen recently-active pathways proportional to reward.

    Uses eligibility traces for temporal credit assignment:
    synapses that participated in recent STDP events get modulated
    proportionally to their trace strength, giving proper credit
    assignment even when reward is delayed.
    """
    if abs(reward) < 0.01:
        return

    scale = reward * network.config.learning_rate * 2.0

    # Use eligibility traces if available (preferred)
    if np.any(network.eligibility_traces != 0):
        network.weights += scale * network.eligibility_traces
    else:
        # Fallback: look at recent spike correlations
        if len(network.spike_history) < 2:
            return
        start = max(0, len(network.spike_history) - 5)
        for t in range(start, len(network.spike_history)):
            spikes = network.spike_history[t]
            fired = spikes > 0.5
            co_active = np.outer(fired, fired)
            np.fill_diagonal(co_active, False)
            network.weights += scale * 0.1 * co_active

    np.clip(network.weights, network.config.min_weight, network.config.max_weight,
            out=network.weights)


def apply_pain_signal(network: SpikingNeuralNetwork, pain: float) -> None:
    """Inject high-priority alarm signal and weaken pathways via eligibility traces.

    Pain floods the network, weakening pathways that were recently strengthened
    (as indicated by positive eligibility traces) and boosting avoidance circuitry.
    """
    if abs(pain) < 0.01:
        return

    # Inject alarm into interneurons (vectorised)
    s = network.inter_start
    n = len(network.inter_indices)
    active = network.refractory_timers[s:s + n] <= 0
    network.potentials[s:s + n] += pain * 0.3 * active

    pain_scale = pain * network.config.learning_rate

    # Use eligibility traces for targeted anti-Hebbian learning
    if np.any(network.eligibility_traces != 0):
        network.weights -= pain_scale * network.eligibility_traces
    else:
        # Fallback: weaken recently-active pathways
        if len(network.spike_history) >= 2:
            recent = network.spike_history[-1]
            prev = network.spike_history[-2]
            fired_now = recent > 0.5
            fired_prev = prev > 0.5
            anti_hebbian = np.outer(fired_prev, fired_now)
            np.fill_diagonal(anti_hebbian, False)
            network.weights -= pain_scale * 0.05 * anti_hebbian

    np.clip(network.weights, network.config.min_weight, network.config.max_weight,
            out=network.weights)
