# Agent8 Climber v1 - Fear-Based Exploration Simulation

A simulation exploring how people with different risk tolerances navigate uncertain territories, using rock climbing as a metaphor.

## What is This?

Imagine two people exploring the same landscape of challenges:

**Bold Explorer (low θ)**: Needs real stakes to feel engaged. Safe territory feels underwhelming. They venture into dangerous zones, which can accelerate learning - but also cuts many lifecycles short.

**Cautious Explorer (high θ)**: Feels engaged even with modest stakes. They stay in safer territory, which limits initial options but allows for longer lifecycles and gradual skill building.

Both seek the same thing: **novelty** - the satisfaction of exploring new territory, developing skills, and operating at the edge of their abilities. The simulation tracks whether different personality types can achieve similar outcomes despite taking very different paths.

### What Is Novelty?

Novelty represents the value gained from exploration. It is the same reward for everyone - unlocking new routes, completing challenges, improving skills. What differs is the pacing:

- Bold agents get larger bursts from high-stakes routes, but their lifecycles often end early
- Cautious agents accumulate novelty more gradually, but over longer lifecycles

Novelty is highest when you are operating at your edge - not too safe (boring), not too scary (overwhelming). This "sweet spot" differs by personality but exists for everyone.

### Two Types of Skill

Agents develop two capabilities through experience:

| Skill | Developed By | What It Does |
|-------|--------------|--------------|
| **Physical** | Attempting difficult routes | Reduces fear of difficulty, improves success rate, unlocks harder routes over time |
| **Mental** | Exposure to consequence (risk) | Reduces "choking under pressure" - the tendency to fail when consequences are high |

Everyone starts at zero for both skills. The difference is exposure:

- Bold agents regularly face high consequences and develop mental skill faster
- Cautious agents encounter high stakes less often, so mental skill develops more slowly, though it still matters when they venture beyond their usual territory

**Choking**: When the stakes exceed your mental training, you may fail a route you would normally complete. This matters for any agent who faces consequences beyond what they have trained for.

### The Core Question

> Can different personality types find similar fulfillment despite opposite approaches?

The simulation suggests yes, but with important caveats. Strategy helps across personality types, and survival strongly influences lifetime outcomes. Bold explorers who survive can match cautious ones - but many do not survive.

### Methodology

This is a **Monte Carlo** simulation. For each configuration, we run hundreds of stochastic agent lifecycles (typically 50-800) and aggregate the statistics. Each run involves random outcomes for route attempts, falls, and deaths based on the probability models described below.

We also use **Evolutionary optimization** to discover optimal strategy weights for each personality type - evolving populations of strategies over multiple generations and selecting for fitness (novelty + skill).


---

## Table of Contents

1. [Quick Start](#quick-start)
2. [The Model](#the-model)
3. [Strategic Route Selection](#strategic-route-selection)
4. [Evolutionary Optimization](#evolutionary-optimization)
5. [Landscapes](#landscapes)
6. [Running Experiments](#running-experiments)
7. [Key Findings](#key-findings)
8. [File Structure](#file-structure)

---

## Quick Start

```bash
# 1. Basic exploration (greedy agents)
python run_exploration.py --quick --landscape compressed --plot

# 2. Strategic exploration with default weights
python run_strategic_exploration.py --quick --landscape compressed --thetas 2,4,6,8 --plot --heatmap

# 3. Evolve optimal strategies and save them
python run_evolution.py --landscape compressed --thetas 2,4,6,8 --plot --save output/optimal_weights.json

# 4. Run strategic exploration with evolved weights
python run_strategic_exploration.py --landscape compressed --thetas 2,4,6,8 --load-weights output/optimal_weights.json --plot --heatmap
```

---

## The Model

### Fear Sensitivity (θ)

Each agent has a fear sensitivity parameter θ (theta) on a scale of 1-8. θ determines the agent's "Lauda line" - the maximum consequence (death probability) they will accept:

| θ Range | Profile | Approximate Maximum Acceptable Death Risk |
|---------|---------|-------------------------------------------|
| 1-2 | Bold | 12-27% per attempt |
| 3-4 | Moderate | 4-7% |
| 5-6 | Cautious | 2-3% |
| 7-8 | Very Cautious | 1-1.5% |

The Lauda line formula (named after Niki Lauda's famous risk calculation):
```python
lauda_line = 0.35 * exp(-0.6 * θ) + 0.01
```

This is not a judgment - it is a parameter. Some people genuinely perform best with higher stakes; others thrive when consequences are contained.

### The Fear Calculation

An agent evaluates each route by computing a fear score combining two factors:

```python
fear = consequence_fear + difficulty_fear
```

**Consequence fear** depends on personality (θ). It reaches 1.0 when the route's consequence equals the agent's Lauda line:

```python
consequence_fear = route.consequence / lauda_line
```

**Difficulty fear** depends on physical skill. As agents develop ability, hard routes feel less intimidating. It also habituates with repeated exposure:

```python
confidence = physical_skill / (2.0 + physical_skill)
difficulty_fear = 0.5 * difficulty * (1 - confidence) * habituation * boost
```

A route feels acceptable when total fear is 1.0 or less. Agents only consider routes where fear is between 0.5 and 1.0 - below 0.5 feels boring and disengaging.

### Route Selection (How Agents Choose)

Agents evaluate routes using a value function that combines three factors:

```python
value = fear_sweetness * stakes_engagement * novelty_potential
```

**Fear sweetness**: A bell curve centered at optimal fear (0.75). Value peaks when fear is in the sweet spot, drops for "too boring" or "too scary":

```python
fear_sweetness = exp(-((fear - 0.75) / 0.25)^2)
```

**Stakes engagement**: Bold agents need real consequence to feel engaged. A hard boulder with low death risk might hit the right fear level but feel hollow without real stakes:

```python
required_consequence = 0.025 / θ
# θ=2: needs ~1.25% death rate for full engagement
# θ=8: needs ~0.3% (easily satisfied)
stakes_engagement = min(1.0, route.consequence / required_consequence)
```

**Novelty potential**: Based on engagement (peaks at 80% of max fear), freshness (decays with repetition), and individual talent:

```python
engagement = exp(-((relative_fear - 0.80) / 0.30)^2)
freshness = exp(-0.5 * experience)
novelty = 10.0 * engagement * freshness * talent * noise
```

### Success and Failure

Success probability uses a two-skill model:

```python
# Physical success - based on difficulty and physical skill
phys_factor = physical_skill / (1 + physical_skill)
base_success = 1 - route.difficulty
p_physical = min(0.99, max(0.05, base_success * (1 + phys_factor)))

# Choking - high stakes + low mental skill = trouble
mental_capacity = mental_skill / (0.5 + mental_skill)
p_choke = min(0.5, route.consequence * (1 - mental_capacity) * 5)

# Combined: must physically succeed AND not choke
p_success = p_physical * (1 - p_choke)
```

On failure, the agent falls. Death probability equals the route's consequence rating directly.

### Skill Development

Skills are gained from every attempt:

```python
# On success:
physical_gained = 0.05 * route.difficulty
mental_gained = 0.05 * route.consequence * 10  # scaled up since consequence values are small

# On failure (survived):
physical_gained = 0.05 * route.difficulty * 0.5
mental_gained = 0.05 * route.consequence * 10 * 0.5
```

### Lifecycle Outcomes

Each agent's lifecycle ends one of three ways:

| Outcome | Meaning |
|---------|---------|
| **Completed** | Finished maximum attempts (full lifecycle) |
| **Stagnated** | No acceptable routes remaining (alive but stuck) |
| **Dead** | Fatal accident |

---

## Strategic Route Selection

### Greedy vs Strategic Agents

**Greedy**: Picks the route with highest immediate value (fear_sweetness * stakes_engagement * novelty_potential).

**Strategic**: Adds three planning bonuses to immediate value:

```python
strategic_value = immediate_value + unlock_bonus + freshness_bonus + diversity_bonus
```

### Strategy Behaviors

| Behavior | Formula | Effect |
|----------|---------|--------|
| **Unlock** | `unlock_weight * min(n_almost, 20) * difficulty` | When routes are "almost accessible" (fear 1.0-1.5), prefer harder routes that build skill toward unlocking them |
| **Freshness** | `freshness_weight * 50 * exp(-0.5 * experience)` | Prefer routes not recently attempted |
| **Diversity** | `diversity_weight * 100` (if new grid cell) | Prefer routes in unvisited areas of the difficulty/consequence grid |

Default weights: unlock=0.2, freshness=0.1, diversity=0.1

### Why Strategy Helps

Without strategy, greedy agents tend to:
- Deplete favorite routes quickly (leading to stagnation)
- Miss opportunities to unlock new territory through skill building
- Cluster in one area of the landscape

Strategic agents spread their exploration, avoid exhausting options, and unlock more routes over time.

---

## Evolutionary Optimization

### Purpose

Hand-tuned strategy weights work reasonably well, but are they optimal? Evolution discovers better weights for each personality type.

### What Gets Evolved

The three behavior weights: `[unlock_weight, freshness_weight, diversity_weight]`

These control HOW agents pick routes. They are separate from the fitness function which controls WHAT outcomes we optimize for.

### Method



<img width="2385" height="1624" alt="evolution_results" src="https://github.com/user-attachments/assets/f4b0b471-12c8-49c0-831e-df434b601053" />



*Evolved strategy weights by personality type. The stacked bars show direction (which behaviors matter), while the black line shows magnitude (how much strategy matters overall).*

```
GENERATION 0: Random population of weight combinations
─────────────────────────────────────────────────────────┐
│  Individual    Weights [u,f,d]      Fitness             │
├─────────────────────────────────────────────────────────┤
│  A             [0.1, 0.8, 0.3]    → 30 runs → 1200      │
│  B             [0.5, 0.1, 0.7]    → 30 runs → 1350      │
│  C             [0.3, 0.2, 0.4]    → 30 runs → 1450 ★    │
│  D             [0.9, 0.2, 0.1]    → 30 runs → 1100      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
              SELECTION (keep top 50%)
              CROSSOVER (blend parent genes)
              MUTATION (add random noise)
                         │
                         ▼
              ... repeat for 30 generations ...
                         │
                         ▼
FINAL: Optimized Weights for this θ
┌─────────────────────────────────────────────────────────┐
│  Best: [0.45, 0.22, 0.38]  Fitness: 1620                │
└─────────────────────────────────────────────────────────┘
```

### Fitness Function

What we optimize for (the goals):

```python
fitness = novelty_weight * avg_novelty + survival_weight * avg_survival * 1000 + skill_weight * avg_skill * 100
```

Default weights: novelty=1.0, survival=0.0, skill=0.5

### Interpreting Results

Strategy weights are decomposed into magnitude and direction:

```python
magnitude = sum(weights)           # How much strategy matters overall
direction = weights / magnitude    # Which behaviors matter most (sums to 100%)
```

**Magnitude interpretation:**

| Range | Meaning |
|-------|---------|
| > 1.5 | Strategy provides meaningful benefit |
| 1.0 - 1.5 | Strategy provides moderate benefit |
| < 1.0 | Natural instincts may suffice |

In practice, evolved magnitudes cluster in the 1.3-1.8 range across personality types, suggesting strategy helps everyone to a similar degree. What varies more is the type of strategy that helps.

### Saving and Loading Weights

```bash
# Evolve and save
python run_evolution.py --landscape compressed --thetas 2,4,6,8 --save output/optimal_weights.json

# Use evolved weights
python run_strategic_exploration.py --load-weights output/optimal_weights.json --thetas 2,4,6,8
```

---

## Landscapes

### Compressed Landscape (Recommended)

Tight consequence bands (0.1% to 3.5%) designed so all θ values can explore most of the 5x5 grid:

```python
landscape = Landscape.create_compressed(seed=42)
```

Consequence bands:
- C1: 0.10% - 0.40% (accessible to all θ)
- C2: 0.40% - 0.70% (accessible to θ <= 7)
- C3: 0.70% - 1.20% (accessible to θ <= 6)
- C4: 1.20% - 2.00% (accessible to θ <= 5)
- C5: 2.00% - 3.50% (accessible to θ <= 4)

```
                        CONSEQUENCE
              C1        C2        C3        C4        C5
           (0.1%)    (0.4%)    (0.7%)    (1.2%)    (2.0%)
         ┌─────────┬─────────┬─────────┬─────────┬─────────┐
    D1   │   500   │   500   │   500   │   500   │   500   │  Easy
         ├─────────┼─────────┼─────────┼─────────┼─────────┤
    D2   │   500   │   500   │   500   │   500   │   500   │
         ├─────────┼─────────┼─────────┼─────────┼─────────┤
    D3   │   500   │   500   │   500   │   500   │   500   │
         ├─────────┼─────────┼─────────┼─────────┼─────────┤
    D4   │   500   │   500   │   500   │   500   │   500   │
         ├─────────┼─────────┼─────────┼─────────┼─────────┤
    D5   │   500   │   500   │   500   │   500   │   500   │  Hard
         └─────────┴─────────┴─────────┴─────────┴─────────┘
                                                    = 12,500 routes
```

**Why compressed?** 
- All θ values can access most of the grid
- Enables fair comparison of exploration patterns
- Reveals territorial differentiation by personality

### Other Options

```python
# Fair: Routes calibrated per-θ for equal sweet-spot access
landscape = Landscape.create_fair(thetas=[2, 4, 6, 8])

# Stratified: Uniform 5x5 grid with wider consequence range
landscape = Landscape.create_stratified_coverage(seed=42)
```

---

## Running Experiments

### Basic Exploration (Greedy)

```bash
python run_exploration.py --quick --landscape compressed --plot --heatmap
python run_exploration.py --full --landscape compressed --thetas 1,2,4,6,8 --plot --heatmap
```

### Strategic Exploration

```bash
# With default weights
python run_strategic_exploration.py --quick --landscape compressed --thetas 2,4,6,8 --plot --heatmap

# With evolved weights
python run_strategic_exploration.py --landscape compressed --thetas 2,4,6,8 --load-weights output/optimal_weights.json --plot --heatmap
```

### Evolutionary Optimization

```bash
# Quick test (~5 min)
python run_evolution.py --quick --landscape compressed --thetas 4,8 --plot

# Standard run (~20 min)
python run_evolution.py --landscape compressed --thetas 2,4,6,8 --plot --save output/optimal_weights.json

# Full optimization (~1 hour)
python run_evolution.py --full --landscape compressed --thetas 1,2,3,4,5,6,7,8 --plot --heatmap --save output/optimal_weights.json
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--quick` | Fewer runs/generations (fast) |
| `--full` | More runs/generations (precise) |
| `--thetas` | Comma-separated θ values |
| `--plot` | Generate learning curves |
| `--heatmap` | Generate exploration heatmaps |
| `--landscape` | `compressed`, `fair`, or `stratified` |
| `--seed` | Random seed |
| `--max-attempts` | Max attempts per agent |
| `--save` | Save evolved weights to JSON |
| `--load-weights` | Load evolved weights from JSON |

---

## Key Findings

### Exploration Patterns

![Exploration Heatmap](output/exploration_heatmap.png)

*Bold agents (low θ) explore vertically into high-consequence territory. Cautious agents (high θ) explore horizontally, pushing difficulty within safer zones.*

### Learning Trajectories


<img width="2084" height="1475" alt="strategic_learning_curves" src="https://github.com/user-attachments/assets/7e49b8ba-5bc7-467b-af1b-355708108597" />



*Novelty and skill accumulation over time. Cautious agents (θ=7-8) often accumulate more total novelty because their lifecycles last longer. All agents tend to converge toward operating at their personal edge (fear around 0.8).*

### What the Simulations Suggest

**1. Similar novelty at personal edges**

Agents across θ values tend to achieve comparable novelty per attempt when operating at their sweet spot (fear around 0.75-0.80). The absolute difficulty matters less than being appropriately challenged for your own threshold.

**2. Different exploration territories**

Bold and cautious agents explore different regions:
- Bold agents explore vertically (consequence dimension) - they access high-risk routes
- Cautious agents explore horizontally (difficulty dimension) - they push technical skill within safer zones

**3. Survival shapes lifetime outcomes**

Cautious agents often accumulate more lifetime novelty - not because they find more per attempt, but because they survive longer. Bold agents can match this, but many lifecycles end early.

**4. Strategy helps across personality types**

Evolution suggests strategy provides moderate benefit for all personality types (magnitude roughly 1.3-1.8 across θ values). What varies more is which behaviors matter:
- Unlock focus (building toward inaccessible routes) tends to dominate
- Freshness (avoiding repetition) provides secondary benefit
- Diversity (exploring new areas) appears more relevant at the extremes (very bold or very cautious)

**5. Strategic agents avoid stagnation**

Greedy agents frequently exhaust their preferred routes and stagnate (20-40% of lifecycles). Strategic agents rarely stagnate because freshness and diversity bonuses spread exploration.

### Caveats

This is a simplified simulation, not a predictive model of human behavior. The findings suggest patterns worth investigating, not definitive conclusions about how people actually behave.

---

## File Structure

```
agent8_climber_v1/
|-- run_exploration.py              # Greedy agent exploration
|-- run_strategic_exploration.py    # Strategic exploration
|-- run_evolution.py                # Evolutionary optimization
|
|-- src/
|   |-- agent.py                    # Agent with fear model and two-skill system
|   |-- strategic_agent.py          # Strategic planning behaviors
|   |-- evolution.py                # Evolutionary optimizer
|   |-- landscape.py                # Route generation
|   |-- exploration.py              # Simulation loop
|   |-- visualizations.py           # Plotting
|
|-- output/                         # Generated figures and data
|-- requirements.txt
|-- README.md
```

---

## Requirements

```
Python 3.8+
numpy
matplotlib
```

Install with: `pip install -r requirements.txt`

---

## Example Workflow

```bash
# 1. Quick exploration to verify setup
python run_exploration.py --quick --landscape compressed --thetas 2,4,6,8 --plot

# 2. Evolve optimal strategies for each θ
python run_evolution.py --landscape compressed --thetas 2,4,6,8 --plot --save output/optimal_weights.json

# 3. Run strategic exploration with evolved weights
python run_strategic_exploration.py --landscape compressed --thetas 2,4,6,8 --load-weights output/optimal_weights.json --plot --heatmap

# 4. Full production run (all θ values)
python run_evolution.py --full --landscape compressed --thetas 1,2,3,4,5,6,7,8 --plot --heatmap --save output/optimal_weights.json
```

---

## Acknowledgments

Thanks to Sophie Herzog for ideas and feedback.

---

## License

MIT
