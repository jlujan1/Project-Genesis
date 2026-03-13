"""Inner Speech & Metacognition — the agent thinking about its own thinking.

Routes proto-language symbols back into the agent's own workspace,
creating an internal monologue loop.  The metacognitive monitor tracks
confidence in recent decisions, detects uncertainty, and can trigger
reflective re-evaluation.

Inspired by Vygotsky's inner-speech theory and Higher-Order Thought
(HOT) theories of consciousness.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class InnerSpeech:
    """Generates internal narration from workspace broadcasts.

    Each tick, the module observes the winning workspace source and
    maps it to one of the agent's proto-language symbols.  This
    'inner voice' is fed back as a workspace packet on the next tick,
    creating a re-entrant loop — consciousness reflecting on itself.
    """

    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size
        # Mapping: workspace source → preferred symbol index
        self.source_symbol_map: dict[str, int] = {}
        self.current_word: int = -1
        self.monologue: deque[int] = deque(maxlen=50)

        # Metacognitive monitor
        self.confidence: float = 0.8
        self.recent_outcomes: deque[bool] = deque(maxlen=30)
        self.uncertainty_streak: int = 0
        self.reflection_count: int = 0

    def update(self, broadcast_source: str, action_succeeded: bool,
               prediction_error: float) -> int:
        """Process the latest broadcast and update inner speech.

        Returns the current inner-speech symbol index (or -1 if silent).
        """
        # Map source to a symbol (learn mapping over time)
        if broadcast_source not in self.source_symbol_map:
            # Assign the next available symbol slot
            idx = len(self.source_symbol_map) % self.vocab_size
            self.source_symbol_map[broadcast_source] = idx

        self.current_word = self.source_symbol_map[broadcast_source]
        self.monologue.append(self.current_word)

        # Metacognitive confidence update
        self.recent_outcomes.append(action_succeeded)
        if len(self.recent_outcomes) >= 5:
            recent_success = sum(self.recent_outcomes) / len(self.recent_outcomes)
            self.confidence = 0.85 * self.confidence + 0.15 * recent_success
        else:
            self.confidence = 0.8

        # High prediction error → moderate confidence reduction
        # Tolerate normal operational errors (< 0.1) to avoid persistent erosion
        if prediction_error > 0.1:
            self.confidence *= max(0.85, 1.0 - (prediction_error - 0.1) * 0.2)

        # Uncertainty detection — only flag genuine uncertainty
        uncertain = self.confidence < 0.5 or prediction_error > 0.2
        if uncertain:
            self.uncertainty_streak += 1
        else:
            self.uncertainty_streak = max(0, self.uncertainty_streak - 1)

        # Trigger reflection when uncertainty is sustained
        if self.uncertainty_streak >= 5:
            self.reflection_count += 1
            self.uncertainty_streak = 0  # reset after reflection

        return self.current_word

    def should_reflect(self) -> bool:
        """Whether metacognition is triggering a reflective re-evaluation."""
        return self.uncertainty_streak >= 3

    def get_encoding(self) -> np.ndarray:
        """Encode inner-speech state for a workspace packet."""
        enc = np.zeros(8, dtype=np.float32)
        enc[0] = self.current_word / max(1, self.vocab_size)
        enc[1] = self.confidence
        enc[2] = min(1.0, self.uncertainty_streak / 15.0)
        enc[3] = 1.0 if self.should_reflect() else 0.0
        # Monologue diversity (how many distinct symbols used recently)
        if self.monologue:
            recent = list(self.monologue)[-10:]
            enc[4] = len(set(recent)) / self.vocab_size
        enc[5] = min(1.0, self.reflection_count / 20.0)
        return enc

    def get_workspace_relevance(self) -> float:
        """Inner speech becomes more relevant when uncertain."""
        base = 0.1
        if self.should_reflect():
            base += 0.4
        if self.confidence < 0.4:
            base += 0.2
        return min(1.0, base)

    def get_decision_modifiers(self) -> dict:
        """Return modifiers that other systems should apply to their decisions.

        When confidence is low and reflection is active, the agent should:
        - Increase exploration noise (try new things)
        - Reduce reliance on motor output (don't trust habitual responses)
        - Boost memory recall relevance (consult past experience)
        - Switch goals more readily
        """
        reflecting = self.should_reflect()
        return {
            "exploration_boost": 0.15 if self.confidence < 0.3 else 0.0,
            "motor_damping": 0.5 if reflecting else 1.0,
            "memory_relevance_boost": 0.3 if reflecting else 0.0,
            "goal_switch_threshold": 0.5 if reflecting else 0.0,
            "reflecting": reflecting,
            "confidence": self.confidence,
        }

    def get_strategy_override(self, current_goal: int,
                              num_actions: int) -> np.ndarray | None:
        """When in prolonged reflection, suggest a strategy shift.

        Returns an action bias that favours exploration and information-
        gathering over exploitation, or None if no override needed.
        """
        if not self.should_reflect():
            return None

        # Reflection mode: bias toward exploratory actions
        bias = np.zeros(num_actions, dtype=np.float32)
        n_move = min(4, num_actions)
        # Favour random movement (explore)
        bias[:n_move] = 0.12
        # Boost sound emission (seek social information)
        if num_actions > 6:
            bias[6] += 0.08
        return bias

    def get_summary(self) -> dict:
        distinct = len(set(self.monologue)) if self.monologue else 0
        return {
            "confidence": self.confidence,
            "current_word": self.current_word,
            "monologue_length": len(self.monologue),
            "distinct_symbols_used": distinct,
            "reflection_count": self.reflection_count,
            "uncertainty_streak": self.uncertainty_streak,
        }
