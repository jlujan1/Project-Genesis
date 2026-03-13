"""Homeostasis engine — pain, pleasure, and survival drives.

Monitors the agent's internal variables and generates cascading
signals that force the Global Workspace to attend to survival needs.
"""

from __future__ import annotations

import numpy as np

from genesis.agent.body import AgentBody
from genesis.cognition.workspace import GlobalWorkspace
from genesis.config import AgentConfig
from genesis.neural.modules import WorkspacePacket
from genesis.neural.plasticity import apply_pain_signal, apply_reward_signal
from genesis.neural.spiking import SpikingNeuralNetwork


class HomeostasisEngine:
    """Monitors internal state and generates survival-priority signals.

    Pain and pleasure are not emotions — they are priority overrides
    that forcibly redirect the Global Workspace's attention and
    modulate neural plasticity to reinforce or punish behaviors.
    """

    def __init__(self, cfg: AgentConfig | None = None) -> None:
        self.cfg = cfg or AgentConfig()
        self.cumulative_pain = 0.0
        self.cumulative_pleasure = 0.0
        self.survival_urgency = 0.0  # 0 = safe, 1 = critical

    def process(self, body: AgentBody, workspace: GlobalWorkspace,
                network: SpikingNeuralNetwork) -> WorkspacePacket | None:
        """Generate homeostasis signals and modulate the nervous system.

        Returns a high-priority workspace packet if survival is threatened.
        """
        pain = body.pain_signal
        pleasure = body.pleasure_signal

        # Track cumulative signals
        self.cumulative_pain = 0.95 * self.cumulative_pain + pain
        self.cumulative_pleasure = 0.95 * self.cumulative_pleasure + pleasure

        # Calculate urgency using configurable thresholds
        e_thresh = self.cfg.survival_energy_threshold
        i_thresh = self.cfg.survival_integrity_threshold
        self.survival_urgency = 0.0
        if body.energy_ratio < e_thresh:
            self.survival_urgency += (e_thresh - body.energy_ratio) * 3.0
        if body.integrity_ratio < i_thresh:
            self.survival_urgency += (i_thresh - body.integrity_ratio) * 3.0
        self.survival_urgency = min(1.0, self.survival_urgency)

        # Apply neural modulation using configurable thresholds
        if pleasure > self.cfg.pleasure_modulation_threshold:
            apply_reward_signal(network, pleasure)

        if pain > self.cfg.pain_modulation_threshold:
            apply_pain_signal(network, pain)
            # Force workspace attention to proprioception
            workspace.boost_attention("proprioception", pain * 0.5)

        # Generate alarm packet if critical
        if body.is_critical or pain > 0.5:
            alarm_data = np.array([
                pain, pleasure, body.energy_ratio, body.integrity_ratio,
                self.survival_urgency, self.cumulative_pain,
                float(body.is_critical), float(body.in_pain),
            ], dtype=np.float32)

            return WorkspacePacket(
                source="homeostasis_alarm",
                data=alarm_data,
                relevance=0.9 + self.survival_urgency * 0.1,  # near-max priority
                metadata={
                    "pain": pain,
                    "pleasure": pleasure,
                    "urgency": self.survival_urgency,
                }
            )

        return None

    def get_state(self) -> dict:
        return {
            "cumulative_pain": self.cumulative_pain,
            "cumulative_pleasure": self.cumulative_pleasure,
            "survival_urgency": self.survival_urgency,
        }
