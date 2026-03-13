"""CSV Data Logger — records time-series metrics to disk.

Produces one CSV per agent, plus a summary CSV, for post-simulation
analysis and plotting.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from genesis.analytics.dashboard import AgentMetrics


class DataLogger:
    """Writes per-tick agent metrics to CSV files in an output directory."""

    FIELDS = [
        "tick",
        "energy",
        "integrity",
        "phi",
        "complexity",
        "reverberation",
        "composite_consciousness",
        "consciousness_phase",
        "self_model_accuracy",
        "has_ego",
        "prediction_error",
        "active_connections",
        "pain",
        "pleasure",
        "survival_urgency",
        "workspace_source",
        "workspace_relevance",
        "dream_cycles",
        "empowerment",
        "agency_score",
        "identity_strength",
        "active_concepts",
        "developmental_stage",
        "grounded_symbols",
    ]

    def __init__(self, output_dir: str, num_agents: int) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._writers: list[csv.DictWriter] = []
        self._files: list = []

        for i in range(num_agents):
            fp = open(self.output_dir / f"agent_{i}.csv", "w", newline="")
            writer = csv.DictWriter(fp, fieldnames=self.FIELDS)
            writer.writeheader()
            self._writers.append(writer)
            self._files.append(fp)

    def log(self, tick: int, metrics: list[AgentMetrics]) -> None:
        """Write one row per agent for this tick."""
        for m in metrics:
            if m.agent_id >= len(self._writers):
                continue
            row = {
                "tick": tick,
                "energy": f"{m.energy:.2f}",
                "integrity": f"{m.integrity:.2f}",
                "phi": f"{m.phi:.6f}",
                "complexity": f"{m.complexity:.6f}",
                "reverberation": f"{m.reverberation:.6f}",
                "composite_consciousness": f"{m.composite_consciousness:.6f}",
                "consciousness_phase": m.consciousness_phase,
                "self_model_accuracy": f"{m.self_model_accuracy:.6f}",
                "has_ego": int(m.has_ego),
                "prediction_error": f"{m.prediction_error:.6f}",
                "active_connections": m.active_connections,
                "pain": f"{m.pain:.4f}",
                "pleasure": f"{m.pleasure:.4f}",
                "survival_urgency": f"{m.survival_urgency:.4f}",
                "workspace_source": m.workspace_source,
                "workspace_relevance": f"{m.workspace_relevance:.4f}",
                "dream_cycles": m.dream_cycles,
                "empowerment": f"{m.empowerment:.4f}",
                "agency_score": f"{m.agency_score:.4f}",
                "identity_strength": f"{m.identity_strength:.4f}",
                "active_concepts": m.active_concepts,
                "developmental_stage": m.developmental_stage,
                "grounded_symbols": m.grounded_symbols,
            }
            self._writers[m.agent_id].writerow(row)

    def close(self) -> None:
        """Flush and close all CSV files."""
        for fp in self._files:
            fp.close()
        self._files.clear()
