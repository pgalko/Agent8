# Agent8 Climber v1 - Fear-Based Exploration with Evolutionary Strategy Optimization

A simulation exploring how different fear sensitivities (θ) lead to different exploration trajectories, with **evolutionary optimization** to discover optimal strategies for each personality type.

## Core Thesis

> "Everyone finds meaning at THEIR edge"

Bold climbers (low θ) and cautious climbers (high θ) achieve similar novelty when operating at their personal edge—but **strategy matters differently** depending on personality, and we can **evolve optimal strategies** for each type.

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

### Agent Fear Sensitivity (θ)

```
θ = 1-2: Bold (like Alex Honnold) - accepts high death risk
θ = 3-4: Moderate
θ = 6-8: Cautious - accepts only low death risk
```

### The Fear Function

```python
fear = consequence_fear + difficulty_fear

# Consequence fear (personality-dependent)
lauda_line = 0.35 × exp(-0.6 × θ) + 0.01  # Max acceptable death risk
consequence_fear = route.consequence / lauda_line

# Difficulty fear (skill-dependent)
confidence = physical_skill / (2.0 + physical_skill)
difficulty_fear = 0.5 × difficulty × (1 - confidence)
```

A route is **accessible** if `fear ≤ 1.0`.

### Sweet Spot

Agents seek routes where `fear ≈ 0.75` (their personal edge):
- Too low (< 0.5): boring, disengaged
- Too high (> 1.0): terrifying, won't attempt
- Just right (0.5-0.9): maximum engagement and novelty

### Two-Skill System

| Skill | Trained By | Effect |
|-------|------------|--------|
| **Physical** | Route difficulty | Reduces difficulty fear, improves success rate |
| **Mental** | Consequence exposure | Reduces choking under pressure |

### Outcome Tracking

Each agent's career ends one of three ways:

| Outcome | Meaning |
|---------|---------|
| **Completed** | Finished all max_attempts (full career) |
| **Stagnated** | Ran out of acceptable routes (alive but stuck) |
| **Dead** | Fatal accident |

---

## Strategic Route Selection

### Greedy vs Strategic

**Greedy Agent**: Picks route with highest immediate value
```python
value = novelty × (1 - fear)
```

**Strategic Agent**: Considers three additional behaviors
```python
strategic_value = (
    immediate_value                              # Novelty now
    + unlock_weight × n_almost × difficulty      # Routes almost accessible
    + freshness_weight × exp(-0.5 × experience)  # Prefer unexplored routes
    + diversity_weight × (new_cell_bonus)        # Explore different grid areas
)
```

### Strategy Weights (Behaviors)

These control **HOW** agents pick routes:

| Weight | Behavior | Effect |
|--------|----------|--------|
| `unlock_weight` | Seek routes near fear threshold | Expands accessible territory |
| `freshness_weight` | Prefer routes not recently attempted | Prevents route exhaustion |
| `diversity_weight` | Explore different grid cells | Spreads exploration across landscape |

### Why Strategy Matters

Without strategy, greedy agents:
- Deplete their favorite routes quickly (stagnation)
- Miss opportunities to unlock new territory
- Cluster in one corner of the landscape

Strategic agents spread exploration, avoid exhaustion, and unlock more routes over time.

---

## Evolutionary Optimization

### The Goal

Find the optimal `[unlock_weight, freshness_weight, diversity_weight]` for each θ.

### How It Works

```
GENERATION 0: Random Population
┌─────────────────────────────────────────────────────────┐
│  Individual    Weights [u,f,d]      Fitness (Novelty)   │
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

### Fitness Function (Goals)

These control **WHAT** outcomes we optimize for:

| Weight | Goal | Default |
|--------|------|---------|
| `novelty_weight` | Total novelty accumulated | 1.0 |
| `survival_weight` | Complete full career | 0.0 |
| `skill_weight` | Total skill developed | 0.5 |

### Normalized Weights: Magnitude + Direction

Raw weights are decomposed into interpretable components:

```
magnitude = sum(weights)           # How much strategy matters
direction = weights / magnitude    # Which behaviors matter most (sums to 100%)
```

**Magnitude Interpretation:**

| Range | Label | Meaning |
|-------|-------|---------|
| > 2.0 | CRITICAL | Strategy is essential (bold climbers) |
| 1.5 - 2.0 | IMPORTANT | Strategy significantly helps |
| 1.0 - 1.5 | MODERATE | Strategy provides benefit |
| < 1.0 | OPTIONAL | Natural fear is sufficient (cautious climbers) |

### Save and Load Evolved Weights

```bash
# Step 1: Run evolution and save weights
python run_evolution.py --landscape compressed --thetas 1,2,3,4,5,6,7,8 --plot --save output/optimal_weights.json

# Step 2: Use evolved weights in strategic exploration
python run_strategic_exploration.py --landscape compressed --thetas 1,2,3,4,5,6,7,8 --load-weights output/optimal_weights.json --plot --heatmap
```

The JSON file contains per-θ optimal weights:

```json
{
  "4.0": {
    "theta": 4.0,
    "raw_weights": [0.54, 0.22, 0.38],
    "magnitude": 1.14,
    "direction": [0.47, 0.19, 0.33],
    "strategy_type": "Unlock-focused",
    "performance": {
      "avg_novelty": 1620,
      "avg_survival": 0.85,
      "avg_skill": 6.2,
      "avg_diversity": 18.5
    }
  }
}
```

---

## Landscapes

### Compressed Landscape (Recommended)

Ultra-tight consequence bands designed so ALL θ values can explore the full grid:

```python
landscape = Landscape.create_compressed(seed=42)
```

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
- Shows territorial differentiation by personality

### Other Landscapes

```python
# Fair: Routes calibrated per-θ (equal access by design)
landscape = Landscape.create_fair(thetas=[2, 4, 6, 8])

# Stratified: Uniform grid, wider consequence range
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

### 1. Territorial Differentiation

Bold and cautious climbers explore different regions of the landscape:
- **Bold (low θ)**: Vertical exploration (consequence dimension)
- **Cautious (high θ)**: Horizontal exploration (difficulty dimension)

### 2. Strategy Magnitude Varies by Personality

| θ | Magnitude | Interpretation |
|---|-----------|----------------|
| 1-2 | ~2.0+ | CRITICAL - strategy is essential for survival |
| 3-4 | ~1.5 | IMPORTANT - strategy helps significantly |
| 5-6 | ~1.0 | MODERATE - strategy provides benefit |
| 7-8 | ~0.7 | OPTIONAL - natural fear is sufficient |

### 3. Strategic Agents Avoid Stagnation

| Agent Type | Stagnation Rate |
|------------|-----------------|
| Greedy | 20-40% (exhaust routes) |
| Strategic | ~0% (freshness/diversity spread exploration) |

### 4. Death Timing Matters More Than Survival Rate

- θ=1 may have higher survival % but lower expected novelty (deaths occur early)
- θ=8 may have lower survival % but higher expected novelty (deaths occur late)

---

## File Structure

```
agent_climber_v1/
├── run_exploration.py              # Greedy agent exploration
├── run_strategic_exploration.py    # Strategic exploration (with --load-weights)
├── run_evolution.py                # Evolutionary optimization (with --save)
│
├── src/
│   ├── agent.py                    # Alex agent with fear model
│   ├── strategic_agent.py          # StrategicAlex with planning behaviors
│   ├── evolution.py                # Evolutionary optimizer
│   ├── landscape.py                # Route generation (incl. compressed)
│   ├── exploration.py              # Simulation loop
│   └── visualizations.py           # Plotting utilities
│
├── output/
│   ├── learning_curves.png
│   ├── exploration_heatmap.png
│   ├── progression_heatmap.png
│   ├── evolution_results.png
│   └── optimal_weights.json        # Evolved weights (from --save)
│
└── README.md
```

---

## Requirements

```
Python 3.8+
numpy
matplotlib
```

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

## License

MIT