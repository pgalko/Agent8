"""
Strategic Agent - Planning-Based Route Selection

Instead of greedy route selection (pick highest immediate value),
this agent uses three strategic behaviors:

1. UNLOCK: Seek routes near the fear threshold to expand territory
2. FRESHNESS: Prefer routes not recently attempted
3. DIVERSITY: Explore different areas of the difficulty/consequence grid

Key insight for fearful people (high θ):
- You can reach the same peaks as bold people
- It just requires smarter sequencing and preparation
- Your fear isn't a barrier, it's a constraint to optimize around

Usage:
    from strategic_agent import StrategicAlex, StrategyConfig
    
    config = StrategyConfig(unlock_weight=0.2, freshness_weight=0.1, diversity_weight=0.1)
    alex = StrategicAlex(AgentConfig(theta=6.0), strategy=config)
"""

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Set
from collections import defaultdict

from agent import Alex, AgentConfig, Route


# Grid cell computation for diversity tracking
# Matches the compressed landscape structure (5x5 grid)
DIFFICULTY_EDGES = [0.1, 1.08, 2.06, 3.04, 4.02, 5.0]  # D1-D5
CONSEQUENCE_EDGES = [0.001, 0.004, 0.007, 0.012, 0.020, 0.035]  # C1-C5


def get_grid_cell(difficulty: float, consequence: float) -> Tuple[int, int]:
    """
    Map a route's difficulty/consequence to grid cell (d_idx, c_idx).
    
    Returns (0-4, 0-4) for D1-D5, C1-C5.
    """
    # Find difficulty band
    d_idx = 0
    for i in range(len(DIFFICULTY_EDGES) - 1):
        if difficulty >= DIFFICULTY_EDGES[i]:
            d_idx = i
    d_idx = min(d_idx, 4)
    
    # Find consequence band
    c_idx = 0
    for i in range(len(CONSEQUENCE_EDGES) - 1):
        if consequence >= CONSEQUENCE_EDGES[i]:
            c_idx = i
    c_idx = min(c_idx, 4)
    
    return (d_idx, c_idx)


@dataclass
class StrategyConfig:
    """Configuration for strategic planning.
    
    Strategy weights control HOW agents pick routes:
    - unlock: seek routes that expand accessible territory
    - freshness: prefer routes not recently attempted  
    - diversity: explore different areas of the grid
    """
    
    # Planning weights (behaviors)
    unlock_weight: float = 0.2         # Weight for unlock potential
    freshness_weight: float = 0.1      # Weight for exploring fresh routes
    diversity_weight: float = 0.1      # Weight for exploring new grid cells


class StrategicAlex(Alex):
    """
    Alex with strategic planning capabilities.
    
    Extends the base Alex with lookahead planning for route selection.
    Uses three strategic behaviors: unlock, freshness, diversity.
    """
    
    def __init__(self, config: AgentConfig, strategy: StrategyConfig = None):
        super().__init__(config)
        self.strategy = strategy or StrategyConfig()
        
        # Track strategic metrics
        self.visited_cells: Set[Tuple[int, int]] = set()  # Grid cells visited
        
        # Caching for performance
        self._route_cache = {}
        self._cache_skill = -1  # Track skill level when cache was made
        
    def _get_route_accessibility(self, all_routes: list) -> dict:
        """Get cached route accessibility analysis."""
        # Invalidate cache if skill changed significantly
        skill_now = self.physical_skill + self.mental_skill
        if abs(skill_now - self._cache_skill) > 0.1 or not self._route_cache:
            accessible = []
            blocked = []
            boring = []
            almost = []  # Fear 1.0-1.5 (just out of reach)
            
            for r in all_routes:
                fear = self.compute_fear(r)
                if fear > 1.5:
                    blocked.append(r)
                elif fear > 1.0:
                    almost.append(r)
                elif fear < 0.5:
                    boring.append(r)
                else:
                    accessible.append(r)
            
            self._route_cache = {
                'accessible': accessible,
                'blocked': blocked,
                'boring': boring,
                'almost': almost,
                'n_almost': len(almost),
            }
            self._cache_skill = skill_now
            
        return self._route_cache
        
    def choose_route(self, available_routes: list) -> Optional[Route]:
        """
        Choose route using strategic planning.
        """
        if not available_routes:
            return None
        
        # Get accessibility analysis
        access = self._get_route_accessibility(available_routes)
        
        # Primary candidates: accessible routes (fear 0.5-1.0)
        candidates = access['accessible']
        
        if not candidates:
            # Strategic fallback: consider "boring" routes
            # But only if there are routes we could unlock
            if access['boring'] and access['n_almost'] > 0:
                candidates = access['boring']
        
        if not candidates:
            return None
        
        # Fast evaluation - sample if too many candidates
        if len(candidates) > 30:
            candidates = random.sample(candidates, 30)
        
        # Compute strategic value for each candidate
        values = [self._strategic_value(r, access) for r in candidates]
        
        # Select best (with small noise for exploration)
        if sum(values) == 0:
            chosen = random.choice(candidates)
        else:
            max_val = max(values)
            noisy_values = [v + random.uniform(0, 0.05 * max_val) for v in values]
            best_idx = noisy_values.index(max(noisy_values))
            chosen = candidates[best_idx]
        
        # Track visited grid cell
        cell = get_grid_cell(chosen.difficulty, chosen.consequence)
        self.visited_cells.add(cell)
        
        return chosen
    
    def _strategic_value(self, route: Route, access: dict) -> float:
        """
        Compute strategic value using three strategic behaviors.
        
        Value = immediate_value + unlock_bonus + freshness_bonus + diversity_bonus
        """
        # 1. IMMEDIATE VALUE (same as greedy)
        immediate = self.compute_route_value(route)
        
        # 2. UNLOCK BONUS
        # If there are routes almost accessible, value skill-building routes more
        n_almost = access['n_almost']
        if n_almost > 0:
            unlock_bonus = self.strategy.unlock_weight * min(n_almost, 20) * route.difficulty
        else:
            unlock_bonus = 0
        
        # 3. FRESHNESS BONUS
        # Prefer routes not recently attempted
        exp = self.experience.get(route.name, 0)
        freshness_bonus = self.strategy.freshness_weight * 50 * math.exp(-0.5 * exp)
        
        # 4. DIVERSITY BONUS
        # Prefer routes in grid cells not yet visited
        cell = get_grid_cell(route.difficulty, route.consequence)
        if cell not in self.visited_cells:
            diversity_bonus = self.strategy.diversity_weight * 100  # Strong bonus for new cells
        else:
            diversity_bonus = 0
        
        total = immediate + unlock_bonus + freshness_bonus + diversity_bonus
        
        return max(0, total)


def compare_strategies(landscape, theta: float, n_runs: int = 30, max_attempts: int = 100, seed: int = 42):
    """
    Compare greedy vs strategic agent for a given θ.
    """
    from exploration import run_exploration
    
    # Run greedy agent
    greedy_results = {'novelty': [], 'survival': [], 'attempts': [], 'skill': []}
    for run in range(n_runs):
        random.seed(seed + run)
        agent = Alex(AgentConfig(theta=theta))
        trace = run_exploration(agent, landscape, max_attempts=max_attempts)
        greedy_results['novelty'].append(trace.total_novelty)
        greedy_results['survival'].append(1 if trace.alive else 0)
        greedy_results['attempts'].append(trace.attempts)
        greedy_results['skill'].append(trace.total_skill)
    
    # Run strategic agent  
    strategic_results = {'novelty': [], 'survival': [], 'attempts': [], 'skill': [], 'diversity': []}
    for run in range(n_runs):
        random.seed(seed + run)
        config = AgentConfig(theta=theta)
        strategy = StrategyConfig(unlock_weight=0.2, freshness_weight=0.1, diversity_weight=0.1)
        agent = StrategicAlex(config, strategy)
        trace = run_exploration(agent, landscape, max_attempts=max_attempts)
        strategic_results['novelty'].append(trace.total_novelty)
        strategic_results['survival'].append(1 if trace.alive else 0)
        strategic_results['attempts'].append(trace.attempts)
        strategic_results['skill'].append(trace.total_skill)
        strategic_results['diversity'].append(len(agent.visited_cells))
    
    return {
        'theta': theta,
        'greedy': {
            'novelty_mean': sum(greedy_results['novelty']) / n_runs,
            'survival_rate': sum(greedy_results['survival']) / n_runs,
            'attempts_mean': sum(greedy_results['attempts']) / n_runs,
            'skill_mean': sum(greedy_results['skill']) / n_runs,
        },
        'strategic': {
            'novelty_mean': sum(strategic_results['novelty']) / n_runs,
            'survival_rate': sum(strategic_results['survival']) / n_runs,
            'attempts_mean': sum(strategic_results['attempts']) / n_runs,
            'skill_mean': sum(strategic_results['skill']) / n_runs,
            'diversity_mean': sum(strategic_results['diversity']) / n_runs,
        },
    }


def print_comparison(comparisons: list):
    """Print comparison table for greedy vs strategic across θ values."""
    print("\n" + "=" * 95)
    print("GREEDY vs STRATEGIC COMPARISON")
    print("=" * 95)
    print()
    
    print(f"{'θ':>5} | {'Mode':>10} | {'Survival':>10} | {'Novelty':>10} | {'Skill':>8} | {'Attempts':>10} | {'Cells':>10}")
    print("-" * 95)
    
    for comp in comparisons:
        theta = comp['theta']
        g = comp['greedy']
        s = comp['strategic']
        
        print(f"{theta:>5.1f} | {'Greedy':>10} | {g['survival_rate']:>9.1%} | {g['novelty_mean']:>10.0f} | {g['skill_mean']:>8.1f} | {g['attempts_mean']:>10.0f} | {'N/A':>10}")
        print(f"{'':>5} | {'Strategic':>10} | {s['survival_rate']:>9.1%} | {s['novelty_mean']:>10.0f} | {s['skill_mean']:>8.1f} | {s['attempts_mean']:>10.0f} | {s['diversity_mean']:>7.1f}/25")
        
        # Show improvement
        novelty_diff = s['novelty_mean'] - g['novelty_mean']
        survival_diff = s['survival_rate'] - g['survival_rate']
        novelty_pct = (novelty_diff / g['novelty_mean'] * 100) if g['novelty_mean'] > 0 else 0
        print(f"{'':>5} | {'Δ':>10} | {survival_diff:>+9.1%} | {novelty_diff:>+10.0f} ({novelty_pct:+.1f}%) |")
        print("-" * 95)
    
    print()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    from landscape import Landscape
    
    print("=" * 60)
    print("STRATEGIC vs GREEDY AGENT COMPARISON")
    print("=" * 60)
    
    # Create landscape
    landscape = Landscape.create_stratified_coverage(seed=42)
    print(f"\nLandscape: {landscape.name}")
    print(f"Total routes: {len(landscape.routes)}")
    print()
    
    # Compare for different θ values
    print("Running comparisons (30 runs each, 100 attempts)...")
    print()
    
    comparisons = []
    for theta in [2.0, 4.0, 6.0, 8.0]:
        print(f"  θ={theta}...", end=" ", flush=True)
        comp = compare_strategies(landscape, theta, n_runs=30, max_attempts=100)
        comparisons.append(comp)
        print("done")
    
    print_comparison(comparisons)
    
    # Key insight
    print("KEY INSIGHT FOR FEARFUL PEOPLE (high θ):")
    print("-" * 60)
    
    fearful = [c for c in comparisons if c['theta'] >= 6.0]
    if fearful:
        for c in fearful:
            g = c['greedy']
            s = c['strategic']
            novelty_gain = (s['novelty_mean'] - g['novelty_mean']) / g['novelty_mean'] * 100 if g['novelty_mean'] > 0 else 0
            survival_gain = (s['survival_rate'] - g['survival_rate']) * 100
            
            print(f"\nθ={c['theta']} (fearful):")
            print(f"  Strategic planning improves novelty by {novelty_gain:+.1f}%")
            print(f"  Survival rate changes by {survival_gain:+.1f} percentage points")
            print(f"  Average {s['investments_mean']:.1f} 'boring but useful' skill investments")
            
            if novelty_gain > 5:
                print(f"  → Strategy HELPS: Patient skill-building pays off!")
            elif novelty_gain < -5:
                print(f"  → Strategy HURTS: Being too cautious reduces exploration")
            else:
                print(f"  → Strategy is NEUTRAL: Similar outcomes either way")