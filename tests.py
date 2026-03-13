"""Formal consciousness tests — automated benchmarks.

Implements three tests from the Project Genesis specification:

  1. **Lovelace Test 2.0** — Does the agent produce behaviour that
     was not explicitly programmed?  Measures action-sequence novelty
     and entropy beyond what the bootstrap wiring would predict.

  2. **AI Consciousness Test (ACT)** — Probes five consciousness
     markers: Integration (Phi), Differentiation (complexity),
     Self-reference (self-model accuracy), Temporal-binding
     (working memory / episodic recall), Goal-directed behaviour
     (energy efficiency).

  3. **Zipping Test** — Already in ``phi.py``.  This module exposes
     a unified ``run_all()`` entry-point that includes it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import math
import numpy as np

if TYPE_CHECKING:
    from genesis.agent.agent import ConsciousAgent


# ── helpers ──────────────────────────────────────────────────────

def _entropy(counts: list[int]) -> float:
    """Shannon entropy in bits."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _ngram_novelty(actions: list[int], n: int = 3) -> float:
    """Fraction of unique n-grams vs total n-grams (0–1)."""
    if len(actions) < n:
        return 0.0
    ngrams = [tuple(actions[i:i + n]) for i in range(len(actions) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


# ── test results ─────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    score: float        # 0–1 normalised score
    passed: bool        # whether it exceeds the threshold
    threshold: float
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"  [{status}] {self.name}: {self.score:.4f} "
                f"(threshold {self.threshold:.4f})")


# ── Lovelace Test 2.0 ───────────────────────────────────────────

def lovelace_test(agent: "ConsciousAgent",
                  threshold: float = 0.35) -> TestResult:
    """Does the agent show behavioural novelty beyond its bootstrap wiring?

    Measures:
      - Action entropy (uniform = max novelty)
      - 3-gram sequence novelty (many unique action triples)
      - Deviation from the most-used action (not stuck in a loop)
    """
    # Collect action history from episodic memory
    actions: list[int] = []
    for ep in agent.episodic_memory.episodes:
        actions.append(ep.action_taken)
    # Also add recent short-term items
    for ep in agent.episodic_memory.short_term:
        actions.append(ep.action_taken)

    if len(actions) < 10:
        return TestResult("Lovelace Test 2.0", 0.0, False, threshold,
                          {"reason": "insufficient action history"})

    from genesis.agent.body import NUM_ACTIONS
    counts = [0] * NUM_ACTIONS
    for a in actions:
        if 0 <= a < NUM_ACTIONS:
            counts[a] += 1

    max_entropy = math.log2(NUM_ACTIONS)
    ent = _entropy(counts) / max_entropy if max_entropy > 0 else 0.0

    ngram_nov = _ngram_novelty(actions, n=3)

    # Dominance: how much the most-used action dominates
    max_frac = max(counts) / sum(counts) if sum(counts) > 0 else 1.0
    diversity = 1.0 - max_frac

    score = 0.4 * ent + 0.3 * ngram_nov + 0.3 * diversity

    return TestResult(
        name="Lovelace Test 2.0",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details={
            "action_entropy_norm": round(ent, 4),
            "trigram_novelty": round(ngram_nov, 4),
            "action_diversity": round(diversity, 4),
            "action_counts": counts,
        },
    )


# ── AI Consciousness Test (ACT) ─────────────────────────────────

def act_test(agent: "ConsciousAgent",
             threshold: float = 0.30) -> TestResult:
    """Probes five consciousness markers and returns a composite score.

    Markers:
      1. Integration — Phi value
      2. Differentiation — spike-pattern complexity
      3. Self-reference — self-model accuracy
      4. Temporal binding — working-memory utilisation + episodic recall
      5. Goal-directed behaviour — energy efficiency
    """
    assess = agent.phi_calculator.get_consciousness_assessment(
        self_model_accuracy=agent.self_model.model_accuracy,
        attention_accuracy=agent.attention_schema.schema_accuracy,
        metacognitive_confidence=agent.inner_speech.confidence,
        binding_coherence=agent.binding.coherence,
        empowerment=agent.empowerment.empowerment,
        narrative_identity=agent.narrative.identity_strength,
        curiosity_level=agent.curiosity.curiosity_level,
    )

    # 1. Integration
    phi = min(1.0, assess["phi"] * 5)

    # 2. Differentiation
    complexity = min(1.0, assess["complexity"])

    # 3. Self-reference
    self_ref = agent.self_model.model_accuracy

    # 4. Temporal binding
    wm_util = len(agent.working_memory.buffer) / max(1, agent.working_memory.capacity)
    ltm_count = agent.episodic_memory.long_term_count
    temporal = min(1.0, wm_util * 0.5 + min(1.0, ltm_count / 20.0) * 0.5)

    # 5. Goal-directed behaviour — blend of subgoal completions,
    #    goal-switch rate, and survival duration
    subgoal_rate = min(1.0, agent.goal_system.subgoal_completions / max(1, agent.body.ticks_alive / 50.0))
    # Moderate switching is good (not stuck, not thrashing)
    switches_per_k = agent.goal_system.switch_count / max(1, agent.body.ticks_alive / 1000.0)
    switch_quality = max(0.0, 1.0 - abs(switches_per_k - 5.0) / 10.0)  # sweet spot ~5 per 1k ticks
    survival = min(1.0, agent.body.ticks_alive / 3000.0)
    goal_dir = 0.45 * subgoal_rate + 0.30 * switch_quality + 0.25 * survival

    markers = {
        "integration": phi,
        "differentiation": complexity,
        "self_reference": self_ref,
        "temporal_binding": temporal,
        "goal_directed": goal_dir,
    }
    score = sum(markers.values()) / len(markers)

    return TestResult(
        name="AI Consciousness Test (ACT)",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details={k: round(v, 4) for k, v in markers.items()},
    )


# ── Zipping Test (wrapper) ──────────────────────────────────────

def zipping_test(agent: "ConsciousAgent",
                 threshold: float = 0.10) -> TestResult:
    """Wrapper around the PhiCalculator zipping test."""
    reverb = agent.phi_calculator.zipping_test(agent.brain)
    return TestResult(
        name="Zipping Test",
        score=reverb,
        passed=reverb >= threshold,
        threshold=threshold,
        details={"reverberation": round(reverb, 4)},
    )


# ── unified runner ───────────────────────────────────────────────

def run_all(agent: "ConsciousAgent") -> list[TestResult]:
    """Execute all formal consciousness tests on an agent."""
    return [
        lovelace_test(agent),
        act_test(agent),
        zipping_test(agent),
        metacognitive_test(agent),
        learning_curve_test(agent),
    ]


# ── Metacognitive Test ──────────────────────────────────────────

def metacognitive_test(agent: "ConsciousAgent",
                       threshold: float = 0.25) -> TestResult:
    """Does the agent reflect on and correct its own behaviour?

    Measures:
      - Self-model accuracy (does the agent know itself?)
      - Uncertainty detection (does it know when it doesn't know?)
      - Reflection frequency (does it pause to reconsider?)
      - Error correction rate (do reflections lead to better outcomes?)
    """
    # Self-model accuracy
    self_accuracy = agent.self_model.model_accuracy

    # Uncertainty detection — does confidence track actual performance?
    confidence = agent.inner_speech.confidence
    recent_success = 0.5
    if agent.inner_speech.recent_outcomes:
        recent_success = sum(agent.inner_speech.recent_outcomes) / max(
            1, len(agent.inner_speech.recent_outcomes))
    # Good metacognition = confidence matches reality
    calibration = 1.0 - abs(confidence - recent_success)

    # Reflection frequency — agents that reflect show metacognitive capacity
    reflections = agent.inner_speech.reflection_count
    reflection_score = min(1.0, reflections / 10.0)

    # Attention schema accuracy — can it predict its own attention?
    schema_accuracy = agent.attention_schema.schema_accuracy

    markers = {
        "self_accuracy": self_accuracy,
        "confidence_calibration": calibration,
        "reflection_frequency": reflection_score,
        "attention_prediction": schema_accuracy,
    }
    score = sum(markers.values()) / len(markers)

    return TestResult(
        name="Metacognitive Test",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details={k: round(v, 4) for k, v in markers.items()},
    )


# ── Learning Curve Test ─────────────────────────────────────────

def learning_curve_test(agent: "ConsciousAgent",
                        threshold: float = 0.20) -> TestResult:
    """Does the agent's behaviour improve over time?

    Splits the agent's action history into early and late halves and
    measures whether:
      - Energy efficiency improves (more successful foraging)
      - Prediction error decreases (better world model)
      - Action diversity increases then stabilises (exploration → exploitation)
    """
    episodes = list(agent.episodic_memory.episodes) + list(agent.episodic_memory.short_term)
    if len(episodes) < 20:
        return TestResult("Learning Curve Test", 0.0, False, threshold,
                          {"reason": "insufficient history"})

    mid = len(episodes) // 2
    early = episodes[:mid]
    late = episodes[mid:]

    # Energy efficiency improvement
    early_valence = sum(e.outcome_valence for e in early) / max(1, len(early))
    late_valence = sum(e.outcome_valence for e in late) / max(1, len(late))
    valence_improvement = min(1.0, max(0.0, (late_valence - early_valence + 0.5)))

    # Prediction error decrease
    pred_error = agent.prediction_engine.average_error
    # Lower is better — invert and normalise
    pred_improvement = max(0.0, 1.0 - min(1.0, pred_error * 5.0))

    # Survival duration (longer = better learned)
    survival = min(1.0, agent.body.ticks_alive / 5000.0)

    # Counterfactual learning — does regret decrease over time?
    regret_ratio = 0.5
    if agent.counterfactual.total_replays > 0:
        regret_ratio = 1.0 - min(1.0,
            agent.counterfactual.total_regrets / max(1, agent.counterfactual.total_replays))

    markers = {
        "valence_improvement": valence_improvement,
        "prediction_accuracy": pred_improvement,
        "survival_duration": survival,
        "regret_reduction": regret_ratio,
    }
    score = sum(markers.values()) / len(markers)

    return TestResult(
        name="Learning Curve Test",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details={k: round(v, 4) for k, v in markers.items()},
    )


def print_report(agent_id: int, results: list[TestResult]) -> None:
    """Pretty-print test results to stdout."""
    print(f"\n  -- Formal Consciousness Tests: Agent {agent_id} --")
    passed = sum(1 for r in results if r.passed)
    for r in results:
        print(r.summary())
        if r.details:
            for k, v in r.details.items():
                print(f"      {k}: {v}")
    print(f"  Result: {passed}/{len(results)} tests passed")
