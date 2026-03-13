"""Communication system — proto-language emergence between agents.

Agents can emit audio signals. Through Hebbian reinforcement, specific
signals become associated with specific meanings (crystals, danger, etc).
This is 'the first word.'

Extended with compositional utterances: agents can emit *two-symbol*
phrases (e.g. call_2 + call_5 = "crystals nearby") and develop grounded
meanings linked to sensory contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Vocabulary slots — agents develop associations for these symbols
NUM_SYMBOLS = 8
SYMBOL_NAMES = ["call_0", "call_1", "call_2", "call_3",
                "call_4", "call_5", "call_6", "call_7"]

# Grounded categories — the "topics" a symbol can refer to
GROUNDING_CATEGORIES = ["food", "danger", "agent", "direction", "state",
                        "predator", "weather", "tool", "request", "trade", "other"]
NUM_CATEGORIES = len(GROUNDING_CATEGORIES)


@dataclass
class SignalMemory:
    """Tracks what happened after hearing a specific signal."""
    symbol_id: int
    times_heard: int = 0
    total_reward_after: float = 0.0
    total_pain_after: float = 0.0
    # Context when the signal was last heard
    last_context: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32))
    # Grounding: which category this symbol is most associated with
    grounding_weights: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_CATEGORIES, dtype=np.float32))

    @property
    def average_valence(self) -> float:
        """Positive = good things follow, negative = bad things follow."""
        if self.times_heard == 0:
            return 0.0
        return (self.total_reward_after - self.total_pain_after) / self.times_heard

    @property
    def grounded_meaning(self) -> str:
        """The category this symbol is most associated with."""
        idx = int(np.argmax(np.abs(self.grounding_weights)))
        if np.abs(self.grounding_weights[idx]) < 0.1:
            return "ungrounded"
        return GROUNDING_CATEGORIES[idx]


class CommunicationSystem:
    """Manages an agent's proto-language capabilities.

    Each agent independently develops associations between audio
    symbols and environmental contexts. When two agents develop
    overlapping associations, they can communicate.
    """

    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id
        self.signal_memories: list[SignalMemory] = [
            SignalMemory(symbol_id=i) for i in range(NUM_SYMBOLS)
        ]

        # Emission tendencies (which symbol to emit in which context)
        self.emission_weights = np.random.randn(NUM_SYMBOLS, 8).astype(np.float32) * 0.1
        self.emission_learning_rate = 0.01

        # Recently heard signals (for delayed reward attribution)
        self.recent_heard: list[tuple[int, int]] = []  # (symbol_id, tick)
        self.attribution_window = 50  # ticks to attribute outcomes to heard signals

        # Compositional phrases: second-word emission weights
        self.phrase_weights = np.random.randn(NUM_SYMBOLS, NUM_SYMBOLS, 8).astype(np.float32) * 0.05
        self.phrase_buffer: list[tuple[int, int, int]] = []  # (word1, word2, tick)
        self.total_phrases_emitted: int = 0

        # Naming system — maps "type:key" → symbol_id
        self.names: dict[str, int] = {}
        self.vocabulary_size: int = 0

    def choose_emission(self, context: np.ndarray) -> int:
        """Choose which symbol to emit based on current context."""
        ctx = context[:8]
        if len(ctx) < 8:
            ctx = np.pad(ctx, (0, 8 - len(ctx)))

        scores = self.emission_weights @ ctx
        # Softmax-like selection
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / (exp_scores.sum() + 1e-9)

        return int(np.random.choice(NUM_SYMBOLS, p=probs))

    def choose_phrase(self, context: np.ndarray) -> tuple[int, int]:
        """Choose a two-word compositional phrase.

        First word is selected via emission_weights; second word is
        conditioned on the first via phrase_weights.
        """
        word1 = self.choose_emission(context)
        ctx = context[:8]
        if len(ctx) < 8:
            ctx = np.pad(ctx, (0, 8 - len(ctx)))
        scores_w2 = self.phrase_weights[word1] @ ctx
        exp_s = np.exp(scores_w2 - np.max(scores_w2))
        probs = exp_s / (exp_s.sum() + 1e-9)
        word2 = int(np.random.choice(NUM_SYMBOLS, p=probs))
        self.total_phrases_emitted += 1
        return word1, word2

    def hear_signal(self, symbol_id: int, context: np.ndarray, tick: int) -> None:
        """Register hearing a signal from another agent."""
        if 0 <= symbol_id < NUM_SYMBOLS:
            mem = self.signal_memories[symbol_id]
            mem.times_heard += 1
            ctx = context[:8]
            if len(ctx) < 8:
                ctx = np.pad(ctx, (0, 8 - len(ctx)))
            mem.last_context = ctx.copy()
            self.recent_heard.append((symbol_id, tick))

            # Grounding update — associate symbol with sensory context
            self._update_grounding(mem, ctx)

            # Trim old entries
            self.recent_heard = [
                (s, t) for s, t in self.recent_heard
                if tick - t < self.attribution_window
            ]

    def _update_grounding(self, mem: SignalMemory, ctx: np.ndarray) -> None:
        """Update grounding weights based on sensory context when signal heard.

        Context layout (first 8 of state vector):
          0: pos_x  1: pos_y  2: vel_x  3: vel_y
          4: energy  5: integrity  6: pain  7: crystals_nearby
        We map these to grounding categories heuristically.
        """
        lr = 0.02
        # food: high crystal proximity or energy gain
        mem.grounding_weights[0] += lr * max(0, ctx[7] if len(ctx) > 7 else 0)
        # danger: pain present
        mem.grounding_weights[1] += lr * max(0, ctx[6] if len(ctx) > 6 else 0)
        # state: energy/integrity context
        mem.grounding_weights[4] += lr * abs(ctx[4]) if len(ctx) > 4 else 0
        # predator: high pain + low integrity often = predator
        if len(ctx) > 6 and len(ctx) > 5:
            if ctx[6] > 0.3 and ctx[5] < 0.5:
                mem.grounding_weights[5] += lr * 0.5

    def attribute_outcome(self, reward: float, pain: float, tick: int) -> None:
        """After receiving reward/pain, attribute it to recently-heard signals."""
        for symbol_id, heard_tick in self.recent_heard:
            if tick - heard_tick < self.attribution_window:
                recency = 1.0 - (tick - heard_tick) / self.attribution_window
                mem = self.signal_memories[symbol_id]
                mem.total_reward_after += reward * recency
                mem.total_pain_after += pain * recency

    def reinforce_emission(self, symbol_id: int, context: np.ndarray,
                           reward: float) -> None:
        """Reinforce emitting a symbol when it led to reward (cooperation)."""
        if 0 <= symbol_id < NUM_SYMBOLS:
            ctx = context[:8]
            if len(ctx) < 8:
                ctx = np.pad(ctx, (0, 8 - len(ctx)))
            self.emission_weights[symbol_id] += \
                self.emission_learning_rate * reward * ctx

    def get_signal_associations(self) -> list[dict]:
        """Return what each symbol 'means' to this agent."""
        return [
            {
                "symbol": SYMBOL_NAMES[m.symbol_id],
                "times_heard": m.times_heard,
                "valence": m.average_valence,
                "meaning": m.grounded_meaning,
            }
            for m in self.signal_memories
        ]

    def is_deceptive_opportunity(self, own_energy: float,
                                  other_agent_nearby: bool) -> bool:
        """Detect if conditions are right for deceptive signaling.

        Deception emerges naturally when:
        - Agent's energy is critically low
        - Another agent is nearby and could compete for a crystal
        - Agent has learned that emitting 'danger' signals causes others to flee
        """
        if not other_agent_nearby or own_energy > 0.5:
            return False
        # Check if any signal has been associated with negative outcomes for others
        return any(m.average_valence < -0.3 and m.times_heard > 5
                   for m in self.signal_memories)

    # ── Compositional language extensions ──────────────────────

    def hear_phrase(self, word1: int, word2: int,
                    context: np.ndarray, tick: int) -> dict | None:
        """Process a heard two-word phrase with compositional grounding.

        Returns parsed meaning if the phrase has developed grounding.
        """
        if not (0 <= word1 < NUM_SYMBOLS and 0 <= word2 < NUM_SYMBOLS):
            return None

        # Record both individual symbols
        self.hear_signal(word1, context, tick)
        self.hear_signal(word2, context, tick)

        # Compositional interpretation:
        # word1 = topic (what category), word2 = modifier (spatial/quality)
        topic_mem = self.signal_memories[word1]
        mod_mem = self.signal_memories[word2]

        topic_meaning = topic_mem.grounded_meaning
        mod_meaning = mod_mem.grounded_meaning

        # Strengthen cross-word grounding associations
        ctx = context[:8]
        if len(ctx) < 8:
            ctx = np.pad(ctx, (0, 8 - len(ctx)))
        self._update_grounding(topic_mem, ctx)
        self._update_grounding(mod_mem, ctx)

        # Update phrase-level context weights
        self.phrase_weights[word1, word2] += (
            0.005 * np.outer(np.ones(NUM_SYMBOLS), ctx)[:NUM_SYMBOLS, :8]
        )

        return {
            "word1": word1,
            "word2": word2,
            "topic": topic_meaning,
            "modifier": mod_meaning,
            "phrase_valence": (topic_mem.average_valence + mod_mem.average_valence) / 2,
        }

    def get_grounding_matrix(self) -> np.ndarray:
        """Return the full symbol × category grounding weight matrix."""
        matrix = np.zeros((NUM_SYMBOLS, NUM_CATEGORIES), dtype=np.float32)
        for i, mem in enumerate(self.signal_memories):
            matrix[i] = mem.grounding_weights
        return matrix

    def get_language_summary(self) -> dict:
        """Summary of language development for analytics."""
        grounded_count = sum(
            1 for m in self.signal_memories if m.grounded_meaning != "ungrounded"
        )
        total_heard = sum(m.times_heard for m in self.signal_memories)
        unique_meanings = len(set(
            m.grounded_meaning for m in self.signal_memories
            if m.grounded_meaning != "ungrounded"
        ))
        return {
            "grounded_symbols": grounded_count,
            "total_symbols": NUM_SYMBOLS,
            "unique_meanings": unique_meanings,
            "total_heard": total_heard,
            "phrases_emitted": self.total_phrases_emitted,
            "named_entities": len(self.names),
            "vocabulary_size": self.vocabulary_size,
        }

    def get_communication_motor_bias(self, num_actions: int) -> np.ndarray:
        """Return motor bias based on recently heard grounded symbols.

        Symbols that have developed grounded meanings now actually
        influence the agent's decisions:
          - 'food' grounding → bias toward COLLECT and movement
          - 'danger' grounding → bias toward fleeing
          - 'agent' grounding → bias toward social actions
        """
        bias = np.zeros(num_actions, dtype=np.float32)
        if not self.recent_heard:
            return bias

        for sym_id, _tick in self.recent_heard:
            mem = self.signal_memories[sym_id]
            meaning = mem.grounded_meaning
            if meaning == "ungrounded":
                continue

            confidence = min(1.0, mem.times_heard / 20.0)
            strength = confidence * 0.12

            if meaning == "food":
                if num_actions > 5:
                    bias[5] += strength       # COLLECT
                n_move = min(4, num_actions)
                bias[:n_move] += strength * 0.3  # move to find food
            elif meaning == "danger":
                n_move = min(4, num_actions)
                bias[:n_move] += strength * 0.5  # flee
                if num_actions > 5:
                    bias[5] -= strength * 0.3    # don't stop to collect
            elif meaning == "agent":
                if num_actions > 6:
                    bias[6] += strength * 0.4    # respond with sound
            elif meaning == "direction":
                n_move = min(4, num_actions)
                bias[:n_move] += strength * 0.2  # follow directional cue
            elif meaning == "predator":
                n_move = min(4, num_actions)
                bias[:n_move] += strength * 0.6  # flee urgently
                if num_actions > 5:
                    bias[5] -= strength * 0.3
            elif meaning == "weather":
                # Storm warning — seek shelter
                if num_actions > 9:
                    bias[9] += strength * 0.2  # BUILD shelter
                if num_actions > 10:
                    bias[10] += strength * 0.15  # REST
            elif meaning == "request":
                # Someone is asking for help — bias toward sharing
                if num_actions > 13:
                    bias[13] += strength * 0.4  # SHARE
            elif meaning == "trade":
                # Trading opportunity — approach and cooperate
                if num_actions > 13:
                    bias[13] += strength * 0.3  # SHARE

        return bias

    # ── Naming system — agents assign names to things ─────────

    def name_entity(self, entity_type: str, entity_key: str,
                    context: np.ndarray) -> int:
        """Assign or retrieve a symbol name for an entity.

        entity_type: 'biome', 'predator', 'agent', 'location'
        entity_key: unique identifier
        Returns the symbol_id associated with this entity.
        """
        full_key = f"{entity_type}:{entity_key}"
        if full_key in self.names:
            return self.names[full_key]

        # Find least-used symbol to assign
        usage = np.array([m.times_heard for m in self.signal_memories],
                         dtype=np.float32)
        # Prefer symbols not yet used as names
        named_symbols = set(self.names.values())
        for sid in named_symbols:
            usage[sid] += 1000  # discourage reuse
        chosen = int(np.argmin(usage))
        self.names[full_key] = chosen

        # Ground this symbol in the appropriate category
        cat_idx = {"biome": 3, "predator": 5, "agent": 2,
                   "location": 3, "weather": 6, "tool": 7}.get(entity_type, 8)
        self.signal_memories[chosen].grounding_weights[cat_idx] += 0.3
        self.vocabulary_size = len(self.names) + sum(
            1 for m in self.signal_memories if m.grounded_meaning != "ungrounded"
        )
        return chosen

    def get_name_for(self, entity_type: str, entity_key: str) -> int | None:
        """Look up the symbol for a named entity, or None."""
        return self.names.get(f"{entity_type}:{entity_key}")

    def get_request_symbol(self) -> int | None:
        """Find which symbol is most closely grounded to 'request'.

        Returns the symbol_id if one is sufficiently grounded, else None.
        If no symbol is grounded to 'request', picks the least-used symbol
        and starts grounding it.
        """
        request_idx = GROUNDING_CATEGORIES.index("request")
        best_sym = -1
        best_weight = 0.0
        for i, mem in enumerate(self.signal_memories):
            w = mem.grounding_weights[request_idx]
            if w > best_weight:
                best_weight = w
                best_sym = i
        if best_sym >= 0 and best_weight > 0.05:
            return best_sym
        # Bootstrap: assign the least-used symbol to "request"
        usage = [m.times_heard for m in self.signal_memories]
        named_symbols = set(self.names.values())
        for sid in named_symbols:
            usage[sid] += 1000
        chosen = int(np.argmin(usage))
        self.signal_memories[chosen].grounding_weights[request_idx] += 0.15
        return chosen
