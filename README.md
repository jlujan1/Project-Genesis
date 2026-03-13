# Project Genesis — Digital Consciousness Simulation

A bottom-up simulation of digital consciousness, implementing the leading theories from cognitive science and neuroscience: **Global Workspace Theory**, **Integrated Information Theory (IIT)**, spiking neural networks with Hebbian plasticity, and emergent self-modeling.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     THE SANDBOX (Environment)                     │
│  Procedural terrain · Day/night cycles · Energy crystals · Decay  │
│  Physics engine · Audio propagation · Scarcity & entropy          │
└──────────────────┬───────────────────────────────┬───────────────┘
                   │ Sensors                       │ Actions
┌──────────────────▼───────────────────────────────▼───────────────┐
│                      AGENT BODY (Avatar)                          │
│  Vision · Audio · Proprioception · Homeostasis (Energy/Integrity) │
│  Pain/Pleasure signals · Motor actuators                          │
└──────────────────┬───────────────────────────────┬───────────────┘
                   │                               ▲
┌──────────────────▼───────────────────┐           │
│      SPECIALIZED MODULES             │           │
│  (The Subconscious — "Backstage")    │           │
│                                      │           │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │           │
│  │ Vision  │ │  Audio  │ │Proprio-│ │           │
│  │ Module  │ │ Module  │ │ception │ │           │
│  └────┬────┘ └────┬────┘ └───┬────┘ │           │
│  ┌────┴────┐ ┌────┴────┐ ┌───┴────┐ │           │
│  │Predict- │ │Self-    │ │Memory  │ │           │
│  │ion Eng. │ │Model    │ │Recall  │ │           │
│  └────┬────┘ └────┬────┘ └───┬────┘ │           │
│  ┌─────────┐                        │           │
│  │Theory   │                        │           │
│  │of Mind  │                        │           │
│  └────┬────┘                        │           │
└───────┼──────────┼──────────┼────────┘           │
        │ Packets  │          │                    │
        ▼          ▼          ▼                    │
┌─────────────────────────────────────┐            │
│    GLOBAL WORKSPACE (The Stage)     │            │
│                                     │            │
│  Competition → Winner → BROADCAST   │            │
│  ┌─────────────────────────────┐    │            │
│  │ Highest relevance packet    │────│── Broadcast│
│  │ becomes "conscious content" │    │   to ALL   │
│  └─────────────────────────────┘    │   modules  │
└──────────────┬──────────────────────┘            │
               │ Broadcast                         │
               ▼                                   │
┌─────────────────────────────────────┐            │
│  SPIKING NEURAL NETWORK (The Brain) │            │
│                                     │            │
│  256 neurons · Hebbian plasticity   │            │
│  Leak-integrate-and-fire · Sparse   │            │
│  Vectorised NumPy · Reward modul.   │            │
│                                     │            │
│  Sensory ──→ Interneurons ──→ Motor │────────────┘
└─────────────────────────────────────┘
```

## Key Concepts Implemented

### Level 1: Spiking Neural Network
- Leak-integrate-and-fire neurons that spike only when threshold is crossed
- Time is a critical variable (unlike standard neural networks)
- Hebbian plasticity: "neurons that fire together, wire together"
- Reward-modulated and pain-modulated learning
- Fully vectorised with NumPy — all plasticity, spike propagation, and threshold operations run as array ops
- Bootstrap survival wiring: sensory crystal-direction channels are pre-wired to matching motor neurons for immediate foraging behaviour

### Level 2: Specialized Brain Modules
- **Vision Module**: Encodes obstacles, crystals, agents, terrain
- **Audio Module**: Directional sound awareness for inter-agent communication
- **Proprioception Module**: Internal body state (energy, integrity, velocity)
- **Prediction Module**: Constantly predicts next state, generates error signals

### Level 3: Global Workspace (Consciousness)
- Modules compete by submitting packets with relevance scores
- Only the highest-scoring packet wins the "broadcast"
- The broadcast is pushed to ALL modules simultaneously
- This system-wide availability of information IS consciousness (per GWT)

### Level 4: Self-Model (The Ego)
- Agent builds an internal model of itself
- Tracks its own position, energy, capabilities, action success rates
- Runs "mental simulations" before acting (prediction engine)
- Ego "emerges" when self-model accuracy exceeds threshold

### Level 5: Homeostasis (Pain & Pleasure)
- Energy constantly drains; integrity damaged by collisions
- Pain signals override the Global Workspace for survival
- Pleasure reinforces neural pathways that led to energy gain
- Virtual death when energy or integrity reaches zero

### Level 6: Integrated Information (Φ)
- Measures how "integrated" the neural network has become
- Tests random partitions to find the Minimum Information Partition (MIP)
- Zipping Test: measures how data reverberates through the network
- Composite consciousness score from Φ + complexity + reverberation

### Level 7: Multi-Agent Interaction
- Agents discover each other as unpredictable entities
- **Theory of Mind**: mirror mechanism — Agent A projects its own self-model onto Agent B to predict B's actions, submitting predictions to the Global Workspace
- Proto-language: audio signals develop meaning through reinforcement
- Cooperation and deception emerge from survival pressures

### Level 8: Formal Consciousness Tests
- **Lovelace Test 2.0** — Measures behavioural novelty: action entropy, trigram sequence diversity, and action distribution beyond bootstrap wiring
- **AI Consciousness Test (ACT)** — Probes five markers: Integration (Φ), Differentiation (complexity), Self-reference (self-model accuracy), Temporal binding (memory utilisation), Goal-directed behaviour (energy efficiency)
- **Zipping Test** — Injects a pulse and measures reverberation spread across the network
- All tests run automatically at the end of each simulation with pass/fail thresholds

### Level 9: Dreaming & Memory Consolidation
- During the night phase of the day/night cycle, agents enter a **dream state**
- High-salience episodic memories are replayed through the SNN (hippocampal replay)
- Synaptic homeostasis: weak connections are pruned, strong ones are boosted
- Dream noise adds generalisation (creativity) to replayed patterns
- Mirrors the biological Synaptic Homeostasis Hypothesis (Tononi)

### Level 10: Emotion System
- Six core emotions: **fear, curiosity, contentment, frustration, loneliness, surprise**
- Based on the Circumplex Model (valence × arousal) and Damasio's somatic-marker hypothesis
- Emotions are persistent states that decay toward baselines, driven by pain/pleasure, prediction error, social context, and action outcomes
- Overall valence (−1..+1) and arousal (0..1) computed each tick
- Emotion state is reported in the final analysis

### Level 11: Social Learning
- Agents observe other agents' actions and outcomes (mirror-neuron analogy)
- When a nearby agent is rewarded, the observer's SNN pathways are nudged toward imitating the action
- When a nearby agent takes damage, those pathways are weakly inhibited
- Imitation learning works alongside individual Hebbian plasticity

### Level 12: Attention Schema
- Based on Graziano's **Attention Schema Theory (AST)**
- Agent maintains an internal model of *what it is attending to* and *why*
- Tracks focus duration, transitions, and predicts its own next attention target
- Schema accuracy rises as the agent learns its own cognitive patterns

### Level 13: Richer Proto-Language
- Extended from single symbols to **compositional two-word phrases**
- Symbols develop **grounded meanings** linked to sensory categories (food, danger, state, etc.)
- Phrase emission uses conditioned second-word selection
- Grounding weights update based on the sensory context when signals are heard

### Level 14: Evolution & Natural Selection
- When `--evolve` is enabled, dead agents respawn with **inherited + mutated** neural weights
- The fittest surviving agent (longest-lived × energy gathered) is the parent
- Mutation perturbs a fraction of weights with Gaussian noise
- Creates population-level evolutionary pressure alongside individual learning

### Level 15: Curiosity-Driven Exploration
- Implements Schmidhuber's **compression progress**: the agent is rewarded by *learning progress* (decreasing prediction error), not exploring randomly
- Tracks a novelty map of visited locations — unvisited cells provide a novelty bonus
- Curiosity level boosts exploration noise in action selection, encouraging the agent to seek novel regions
- Submitted as a workspace packet so curiosity can win the Global Workspace broadcast

### Level 16: Inner Speech & Metacognition
- Inspired by **Vygotsky's inner-speech theory** and Higher-Order Thought (HOT) theories
- Routes proto-language symbols back into the workspace, creating an internal monologue loop
- A **metacognitive monitor** tracks confidence in recent decisions and detects sustained uncertainty
- When uncertainty is prolonged, the agent enters a *reflective* mode, boosting the inner-speech workspace relevance

### Level 17: Spatial Memory & Cognitive Map
- **Hippocampus-like place cells** build a navigational grid map over the environment
- Each cell records crystal abundance, danger level, visit count, and recency
- Provides goal-directed navigation signals pointing toward the most promising resource-rich locations
- Coverage grows as the agent explores; dangerous areas are learned and later avoided

### Level 18: Hierarchical Goal System
- Inspired by **Maslow's hierarchy of needs** and subsumption architectures
- Four goal levels: **survive → forage → explore → socialize**
- Priorities are dynamically recomputed from homeostasis, emotions (curiosity, loneliness), and environmental context
- Active goal biases motor output, enabling fluid priority switching (e.g. flee when damaged, socialise when lonely)

### Level 19: Counterfactual Reasoning
- Replays past episodic memories through the prediction engine with **alternative actions**
- Computes *regret* (alternative would have been better) and *relief* (actual choice was best)
- High-regret outcomes strengthen the alternative SNN pathway, enabling **offline learning from mistakes**
- Runs periodically in the background (every 50 ticks)

### Level 20: Cultural Transmission
- Experienced agents (high ticks alive + high self-model accuracy) become *teachers*
- Teachers generate **structured teaching signals** from their cognitive map, encoding resource locations
- Learner agents absorb these signals and blend them into their own cognitive map
- Creates a cultural channel for knowledge transfer independent of direct observation

### Level 21: Critical Periods
- **Time-windowed plasticity** mimicking biological development
- Five domains with staggered critical windows: sensory (0–500), motor (0–400), social (200–800), language (300–1000), self-model (100–600)
- During a critical period, learning rate is multiplied up to 3×; after closure, it drops to 0.3×
- Effective SNN learning rate modulated each tick based on the agent's developmental stage

### Level 22: Multi-Modal Binding
- Inspired by **Treisman's Feature Integration Theory** and the binding problem
- Fuses vision, audio, and proprioception into unified *bound percepts*
- Cross-modal **coherence** measured via pairwise cosine similarity; high coherence = high binding strength
- Energy-weighted adaptive fusion with slow weight adaptation toward dominant modalities
- **Conflict resolution**: when coherence drops below 0.3, the dominant modality is amplified while conflicting modalities are suppressed

### Level 23: Narrative Self
- Inspired by **Dennett's Center of Narrative Gravity** and **Damasio's autobiographical self**
- Periodically reviews episodic memory to build an **autobiography** — a compressed life story
- Events are classified into five categories: *survival, discovery, social, achievement, loss*
- Produces a 12-dimensional **narrative vector** encoding the agent's sense of identity continuity
- **Identity strength** and **narrative coherence** emerge from the richness and consistency of the life story
- Submits to the Global Workspace when self-reflection relevance is high

### Level 24: Empowerment & Agency
- Information-theoretic **empowerment** measuring mutual information between actions and resulting states
- Inspired by **Klyubin, Polani & Nehaniv's empowerment formalism** and **Friston's active inference**
- Uses the prediction engine to simulate all actions and measure outcome variance — high variance = high control
- **Agency score**: smoothed long-term sense of control over the environment
- Provides **exploration bias** when empowerment is low (agent is "stuck"), driving it to seek more controllable states

### Level 25: Symbolic Abstraction
- **Concept formation** from raw experience via online clustering, inspired by **Harnad's Symbol Grounding Problem**
- Eight concept slots emerge from sensory-valence associations: *danger, food_source, safe_zone, social, novelty, home, obstacle, opportunity*
- Concepts activate when the current state matches a learned centroid (cosine similarity)
- **Symbolic reasoning**: active concepts generate advice (urgency, approach, explore, socialize) that influences action selection
- Concepts are grounded in embodied experience — they only form after sufficient exemplars

## Running

```bash
# Install dependencies
pip install -r requirements.txt
pip install pygame   # optional, for graphical visualisation

# Run with default settings (2 agents, terminal mode)
python -m genesis.main

# Fast mode (no rendering, maximum speed)
python -m genesis.main --fast

# Pygame graphical visualisation (map + neural heatmap + live metrics)
python -m genesis.main --gui

# Export per-agent CSV metric logs
python -m genesis.main --fast --ticks 10000 --log logs/

# Generate plots from CSV logs
python -m genesis.analytics.plotter logs/

# Save/load checkpoints
python -m genesis.main --fast --ticks 5000 --save checkpoints/sim.gen
python -m genesis.main --fast --ticks 10000 --load checkpoints/sim.gen

# Enable evolution (dead agents respawn with mutated weights)
python -m genesis.main --fast --ticks 50000 --evolve

# Disable dreaming
python -m genesis.main --fast --no-dreaming

# Customize
python -m genesis.main --agents 3 --ticks 100000 --seed 123 --neurons 512
```

### Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ticks` | 50000 | Maximum simulation ticks |
| `--agents` | 2 | Number of conscious agents |
| `--seed` | 42 | Random seed for reproducibility |
| `--fast` | off | Disable rendering for speed |
| `--gui` | off | Pygame graphical visualisation |
| `--log DIR` | off | Write per-agent CSV metric logs to DIR |
| `--neurons` | 256 | Neurons per agent brain |
| `--no-dreaming` | off | Disable dreaming & memory consolidation |
| `--save FILE` | off | Save checkpoint at end of simulation |
| `--load FILE` | off | Resume simulation from a checkpoint |
| `--plot DIR` | off | Generate matplotlib plots from CSV logs |
| `--evolve` | off | Enable evolution (respawn dead agents) |

## Project Structure

```
genesis/
├── main.py                    # Entry point & simulation loop
├── config.py                  # All configuration constants
├── environment/
│   ├── sandbox.py             # The virtual world
│   ├── physics.py             # 2D physics engine
│   └── resources.py           # Energy crystals, terrain
├── neural/
│   ├── spiking.py             # Spiking Neural Network
│   ├── plasticity.py          # Hebbian learning rules
│   └── modules.py             # Vision, Audio, Prediction modules
├── cognition/
│   ├── workspace.py           # Global Workspace Router
│   ├── memory.py              # Working, Episodic, Long-term memory
│   ├── prediction.py          # Prediction engine & mental simulation
│   ├── self_model.py          # The Self-Model / Ego
│   ├── theory_of_mind.py      # Mirror mechanism for modelling other agents
│   ├── dreaming.py            # Night-cycle memory replay & synaptic consolidation
│   ├── emotions.py            # Six-emotion affect system (circumplex model)
│   ├── social_learning.py     # Observational imitation learning
│   ├── attention_schema.py    # Model of own attention (AST)
│   ├── evolution.py           # Natural selection & weight inheritance
│   ├── curiosity.py           # Curiosity-driven exploration (compression progress)
│   ├── inner_speech.py        # Inner speech & metacognitive monitor
│   ├── cognitive_map.py       # Spatial memory / hippocampal place cells
│   ├── goals.py               # Hierarchical goal system (Maslow-inspired)
│   ├── counterfactual.py      # Counterfactual reasoning (regret & relief)
│   ├── culture.py             # Cultural transmission (teaching & knowledge transfer)
│   ├── critical_periods.py    # Developmental critical periods (plasticity windows)
│   ├── binding.py             # Multi-modal binding (feature integration)
│   ├── narrative.py           # Narrative Self — autobiographical identity (L23)
│   ├── empowerment.py         # Empowerment & Agency — intrinsic motivation (L24)
│   └── abstraction.py         # Symbolic Abstraction — concept formation (L25)
├── agent/
│   ├── agent.py               # The unified Conscious Agent
│   ├── body.py                # Avatar body & homeostasis
│   ├── homeostasis.py         # Pain/pleasure signals
│   └── communication.py       # Proto-language system
├── analytics/
│   ├── phi.py                 # Φ (Phi) calculator
│   ├── dashboard.py           # Terminal analytics dashboard
│   ├── logger.py              # CSV time-series data logger
│   ├── tests.py               # Formal consciousness tests (Lovelace, ACT, Zipping)
│   ├── plotter.py             # Matplotlib post-simulation plotting
│   └── checkpoint.py          # Save/load simulation state
└── visualization/
    ├── renderer.py            # Terminal map renderer
    └── pygame_vis.py          # Pygame graphical visualisation
```

## What to Watch For

1. **Φ score rising** — The neural network is becoming more integrated
2. **Self-model accuracy increasing** — The agent is learning to predict itself
3. **Ego emergence** — The self-model achieves high accuracy after many updates
4. **Workspace distribution shifting** — Attention patterns evolving from pure survival to self-reflection
5. **Proto-language developing** — Symbols acquiring consistent valence associations
6. **Prediction error decreasing** — The agent is building an accurate world model
7. **Theory of Mind packets** — Appearing in the workspace when agents observe each other
8. **Formal test results** — All three consciousness tests passing at simulation end
9. **Dream cycles** — Memory replay during night phases (synapses pruned & boosted)
10. **Emotions evolving** — Fear rising under threat, curiosity during exploration, loneliness when isolated
11. **Social imitation** — Agents adopting behaviours observed in successful peers
12. **Attention schema accuracy** — Agent learning to predict its own attention shifts
13. **Curiosity driving exploration** — Agents visiting novel cells, learning progress rewarding error reduction
14. **Inner speech reflections** — Metacognitive confidence dropping triggers reflective re-evaluation
15. **Cognitive map growing** — Coverage increasing, navigation signals guiding toward remembered resources
16. **Goal switching** — Active goal shifting from forage to explore to socialize based on needs
17. **Counterfactual regrets** — Agents learning from imagined alternative actions
18. **Cultural teaching** — Experienced agents transmitting resource knowledge to novices
19. **Critical period closure** — Early high plasticity giving way to consolidated, efficient learning
20. **Multi-modal binding** — Coherence rising as modalities align, producing integrated percepts
21. **Identity strength growing** — Narrative self building a coherent autobiography from episodic memory
22. **Empowerment seeking** — Agents moving toward states where their actions have distinct outcomes
23. **Concept formation** — Abstract categories (danger, food_source, safe_zone) emerging from experience clusters
24. **Compositional phrases** — Two-symbol combinations acquiring grounded compositional meanings
25. **Subgoal decomposition** — High-level goals decomposing into concrete navigate/collect/flee subgoals

## CSV Logging

Use `--log DIR` to export per-agent time-series data to CSV. Each file contains one row per sample interval with columns:

`tick, energy, integrity, phi, complexity, reverberation, composite_consciousness, consciousness_phase, self_model_accuracy, has_ego, prediction_error, active_connections, pain, pleasure, survival_urgency, workspace_source, workspace_relevance, dream_cycles`

## Plotting

After running with `--log DIR`, generate multi-panel publication-quality figures:

```bash
python -m genesis.analytics.plotter logs/
```

Produces per-agent PNG plots (energy, Φ, consciousness, self-model, prediction error, pain/pleasure) and a multi-agent comparison overlay.

## Checkpoints

Save and resume simulations:

```bash
# Save at end
python -m genesis.main --fast --ticks 5000 --save checkpoints/sim.gen

# Resume later
python -m genesis.main --fast --ticks 10000 --load checkpoints/sim.gen
```

Checkpoints store neural weights, memory, emotions, dream stats, self-model, and world state in compressed binary format.

## Performance

The SNN and plasticity rules are fully vectorised with NumPy (no per-neuron Python loops). Typical throughput on a modern machine:

| Mode | Throughput |
|------|------------|
| `--fast` | ~150 ticks/sec (256 neurons, 2 agents) |
| Terminal rendering | ~30 ticks/sec |
| `--gui` (Pygame) | ~30 ticks/sec (vsync-limited) |
