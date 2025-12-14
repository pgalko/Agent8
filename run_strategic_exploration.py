#!/usr/bin/env python3
"""
Alex Climber v2 - Strategic Agent Exploration

Runs the STRATEGIC agent (with unlock/freshness/diversity weighting) 
instead of the base greedy agent.

Output format matches run_exploration.py exactly.

Usage:
    python run_strategic_exploration.py --quick --landscape compressed --thetas 1,2,4,6,8 --plot --heatmap
    python run_strategic_exploration.py --full --landscape compressed --max-attempts 800 --plot --heatmap
"""

import sys
import os
import argparse
import random
import time
import json
from typing import List, Dict
import numpy as np

sys.path.insert(0, 'src')

from agent import AgentConfig
from landscape import Landscape
from exploration import run_exploration, ExplorationTrace
from strategic_agent import StrategicAlex, StrategyConfig
from visualizations import create_learning_curves, create_exploration_heatmap, create_progression_heatmap


# =============================================================================
# EVOLVED WEIGHTS LOADING
# =============================================================================

def load_evolved_weights(filepath: str) -> Dict[float, StrategyConfig]:
    """
    Load evolved weights from JSON file created by run_evolution.py --save.
    
    Returns a dict mapping theta -> StrategyConfig
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    strategies = {}
    for theta_str, info in data.items():
        theta = float(theta_str)
        weights = info['raw_weights']
        strategies[theta] = StrategyConfig(
            unlock_weight=weights[0],
            freshness_weight=weights[1],
            diversity_weight=weights[2],
        )
    
    return strategies


# =============================================================================
# SAFE STATISTICS
# =============================================================================

def safe_mean(data: List[float]) -> float:
    return np.mean(data) if data else float('nan')

def safe_std(data: List[float], ddof: int = 1) -> float:
    if len(data) < 2:
        return float('nan')
    return np.std(data, ddof=ddof)

def safe_min(data: List[float]) -> float:
    return np.min(data) if data else float('nan')

def safe_max(data: List[float]) -> float:
    return np.max(data) if data else float('nan')

def fmt(value: float, fmt_spec: str = ".2f") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:{fmt_spec}}"


# =============================================================================
# BATCH RUNNER FOR STRATEGIC AGENT
# =============================================================================

def run_strategic_batch(
    thetas: List[float],
    landscape: Landscape,
    n_runs: int = 100,
    max_attempts: int = 500,
    seed_base: int = 0,
    strategy_config: StrategyConfig = None,
    strategy_configs: Dict[float, StrategyConfig] = None,
) -> Dict[float, List[ExplorationTrace]]:
    """
    Run multiple strategic agent explorations for each theta.
    
    Args:
        strategy_config: Single config used for all thetas (if strategy_configs not provided)
        strategy_configs: Dict mapping theta -> StrategyConfig (from evolved weights)
    """
    # Default strategy if neither provided
    default_strategy = strategy_config or StrategyConfig(
        unlock_weight=0.2,
        freshness_weight=0.1,
        diversity_weight=0.1,
    )
    
    results = {}
    
    for theta in thetas:
        # Use per-theta config if available, else use default
        if strategy_configs and theta in strategy_configs:
            strategy = strategy_configs[theta]
        else:
            strategy = default_strategy
            
        traces = []
        for run in range(n_runs):
            random.seed(seed_base + run)
            agent = StrategicAlex(AgentConfig(theta=theta), strategy)
            trace = run_exploration(agent, landscape, max_attempts=max_attempts)
            traces.append(trace)
        results[theta] = traces
    
    return results


# =============================================================================
# METRICS COMPUTATION (matches run_exploration.py)
# =============================================================================

def compute_metrics(results: Dict[float, List[ExplorationTrace]]) -> Dict:
    """Compute comprehensive metrics for all thetas."""
    metrics = {}
    
    for theta in sorted(results.keys()):
        traces = results[theta]
        # Real survivors = alive AND completed full career
        survivors = [t for t in traces if t.alive and t.completed]
        # Stagnated = alive but stopped early (ran out of routes)
        stagnated = [t for t in traces if t.alive and not t.completed]
        dead = [t for t in traces if not t.alive]
        
        # === SURVIVAL ===
        survival_rate = len(survivors) / len(traces)
        stagnation_rate = len(stagnated) / len(traces)
        death_rate = len(dead) / len(traces)
        
        # === NOVELTY ===
        all_novelty = [t.total_novelty for t in traces]
        surv_novelty = [t.total_novelty for t in survivors]
        dead_novelty = [t.total_novelty for t in dead]
        
        # === SKILL (TWO-SKILL MODEL) ===
        all_skill = [t.total_skill for t in traces]
        surv_skill = [t.total_skill for t in survivors]
        
        # Physical skill (trained by difficulty)
        all_phys_skill = [t.physical_skill for t in traces]
        surv_phys_skill = [t.physical_skill for t in survivors]
        
        # Mental skill (trained by consequence exposure)
        all_mental_skill = [t.mental_skill for t in traces]
        surv_mental_skill = [t.mental_skill for t in survivors]
        
        # === DIFFICULTY ===
        all_eff_diff = [t.max_difficulty_reached for t in traces]
        all_phys_achieved = [t.max_physical_achieved for t in traces]
        all_phys_attempted = [t.max_physical_attempted for t in traces]
        surv_eff_diff = [t.max_difficulty_reached for t in survivors]
        surv_phys_achieved = [t.max_physical_achieved for t in survivors]
        
        # === ATTEMPTS & SUCCESS ===
        all_attempts = [t.attempts for t in traces]
        surv_attempts = [t.attempts for t in survivors]
        all_successes = [t.successes for t in traces]
        success_rates = [t.successes / t.attempts if t.attempts > 0 else 0 for t in traces]
        
        # === ROUTES EXPLORED ===
        routes_tried = [t.routes_tried for t in traces]
        surv_routes = [t.routes_tried for t in survivors]
        
        # === FEAR (average across trajectory) ===
        avg_fears = []
        for t in traces:
            if t.fear_over_time:
                avg_fears.append(np.mean(t.fear_over_time))
        
        surv_avg_fears = []
        for t in survivors:
            if t.fear_over_time:
                surv_avg_fears.append(np.mean(t.fear_over_time))
        
        # === CONSEQUENCE ===
        avg_cons = []
        for t in traces:
            if t.consequence_over_time:
                avg_cons.append(np.mean(t.consequence_over_time))
        
        surv_avg_cons = []
        for t in survivors:
            if t.consequence_over_time:
                surv_avg_cons.append(np.mean(t.consequence_over_time))
        
        # === DEATH TIMING ===
        death_attempts = [t.attempts for t in dead] if dead else []
        
        metrics[theta] = {
            'n_runs': len(traces),
            'n_survivors': len(survivors),
            'n_stagnated': len(stagnated),
            'n_dead': len(dead),
            'survival_rate': survival_rate,
            'stagnation_rate': stagnation_rate,
            'death_rate': death_rate,
            
            # Novelty
            'novelty_all_mean': np.mean(all_novelty),
            'novelty_all_std': np.std(all_novelty),
            'novelty_surv_mean': safe_mean(surv_novelty),
            'novelty_surv_std': safe_std(surv_novelty),
            'novelty_surv_min': safe_min(surv_novelty),
            'novelty_surv_max': safe_max(surv_novelty),
            'novelty_dead_mean': safe_mean(dead_novelty),
            
            # Skill (Total = Physical + Mental)
            'skill_all_mean': np.mean(all_skill),
            'skill_all_std': np.std(all_skill),
            'skill_surv_mean': safe_mean(surv_skill),
            'skill_surv_std': safe_std(surv_skill),
            
            # Physical Skill (trained by difficulty)
            'phys_skill_all_mean': np.mean(all_phys_skill),
            'phys_skill_surv_mean': safe_mean(surv_phys_skill),
            'phys_skill_surv_std': safe_std(surv_phys_skill),
            
            # Mental Skill (trained by consequence exposure)
            'mental_skill_all_mean': np.mean(all_mental_skill),
            'mental_skill_surv_mean': safe_mean(surv_mental_skill),
            'mental_skill_surv_std': safe_std(surv_mental_skill),
            
            # Difficulty (all refer to ACHIEVED/SUCCESS routes now)
            'eff_diff_all_mean': np.mean(all_eff_diff),
            'eff_diff_all_max': np.max(all_eff_diff),
            'eff_diff_surv_mean': safe_mean(surv_eff_diff),
            'phys_diff_all_mean': np.mean(all_phys_achieved),
            'phys_diff_all_max': np.max(all_phys_achieved),
            'phys_diff_surv_mean': safe_mean(surv_phys_achieved),
            'phys_attempted_mean': np.mean(all_phys_attempted),
            'phys_attempted_max': np.max(all_phys_attempted),
            
            # Attempts
            'attempts_all_mean': np.mean(all_attempts),
            'attempts_surv_mean': safe_mean(surv_attempts),
            'success_rate_mean': np.mean(success_rates),
            
            # Routes
            'routes_all_mean': np.mean(routes_tried),
            'routes_surv_mean': safe_mean(surv_routes),
            
            # Fear and consequence
            'fear_all_mean': safe_mean(avg_fears),
            'fear_surv_mean': safe_mean(surv_avg_fears),
            'cons_all_mean': safe_mean(avg_cons),
            'cons_surv_mean': safe_mean(surv_avg_cons),
            
            # Death timing
            'death_attempt_mean': safe_mean(death_attempts),
            'death_attempt_std': safe_std(death_attempts),
        }
    
    return metrics


# =============================================================================
# OUTPUT FORMATTING (matches run_exploration.py)
# =============================================================================

def print_results(metrics: Dict, thetas: List[float]):
    """Print comprehensive results tables."""
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    print()
    
    # --- OUTCOMES TABLE ---
    print("OUTCOMES (Completed = full career, Stagnated = ran out of routes, Dead = fatal accident)")
    print(f"{'θ':>5} | {'Runs':>6} | {'Completed':>10} | {'Stagnated':>10} | {'Dead':>8} | {'Completed%':>10}")
    print("-" * 70)
    for theta in thetas:
        m = metrics[theta]
        print(f"{theta:>5.1f} | {m['n_runs']:>6} | {m['n_survivors']:>10} | {m['n_stagnated']:>10} | "
              f"{m['n_dead']:>8} | {m['survival_rate']:>9.1%}")
    print()
    
    # --- NOVELTY TABLE ---
    print("NOVELTY (Meaning Acquired)")
    print(f"{'θ':>5} | {'Mean':>10} | {'Std':>10} | {'Survivors':>10} | {'Deaths':>10}")
    print("-" * 60)
    for theta in thetas:
        m = metrics[theta]
        all_mean = m['novelty_all_mean']
        all_std = m['novelty_all_std']
        surv_mean = fmt(m['novelty_surv_mean'], ".0f")
        dead_mean = fmt(m['novelty_dead_mean'], ".0f")
        print(f"{theta:>5.1f} | {all_mean:>10.0f} | {all_std:>10.0f} | {surv_mean:>10} | {dead_mean:>10}")
    print()
    
    # --- SKILL TABLE (TWO-SKILL MODEL) ---
    print("SKILL DEVELOPMENT (all agents)")
    print(f"{'θ':>5} | {'Physical':>10} | {'Mental':>10} | {'Total':>10} | {'Phys/Mental':>12}")
    print("-" * 60)
    for theta in thetas:
        m = metrics[theta]
        phys = m['phys_skill_all_mean']
        mental = m['mental_skill_all_mean']
        total = m['skill_all_mean']
        
        if np.isnan(phys):
            print(f"{theta:>5.1f} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>12}")
        else:
            ratio = phys / mental if mental > 0 else float('inf')
            print(f"{theta:>5.1f} | {phys:>10.2f} | {mental:>10.2f} | {total:>10.2f} | {ratio:>11.1f}x")
    print()
    
    # --- DIFFICULTY TABLE ---
    print("DIFFICULTY ACHIEVED (successful climbs only)")
    print(f"{'θ':>5} | {'Eff Mean':>10} | {'Eff Max':>10} | {'Phys Mean':>10} | "
          f"{'Phys Max':>10} | {'Mental+':>10}")
    print("-" * 70)
    for theta in thetas:
        m = metrics[theta]
        # Mental load = (effective - physical) / physical × 100
        if m['phys_diff_all_mean'] > 0:
            mental_pct = (m['eff_diff_all_mean'] / m['phys_diff_all_mean'] - 1) * 100
        else:
            mental_pct = 0
        print(f"{theta:>5.1f} | {m['eff_diff_all_mean']:>10.2f} | {m['eff_diff_all_max']:>10.2f} | "
              f"{m['phys_diff_all_mean']:>10.2f} | {m['phys_diff_all_max']:>10.2f} | {mental_pct:>9.0f}%")
    print()
    
    # --- BEHAVIOR TABLE ---
    print("BEHAVIOR & RISK PROFILE")
    print(f"{'θ':>5} | {'Attempts':>10} | {'Success%':>10} | {'Routes':>8} | "
          f"{'Avg Fear':>10} | {'Avg Cons':>12} | {'Death @ Att':>12}")
    print("-" * 90)
    for theta in thetas:
        m = metrics[theta]
        death_mean = m['death_attempt_mean']
        death_std = m['death_attempt_std']
        if death_mean is None or (isinstance(death_mean, float) and np.isnan(death_mean)):
            death_str = "N/A"
        elif death_std is None or (isinstance(death_std, float) and np.isnan(death_std)):
            death_str = f"{death_mean:.0f}"
        else:
            death_str = f"{death_mean:.0f}±{death_std:.0f}"
        print(f"{theta:>5.1f} | {m['attempts_all_mean']:>10.0f} | {m['success_rate_mean']:>9.1%} | "
              f"{m['routes_all_mean']:>8.1f} | {m['fear_all_mean']:>10.2f} | "
              f"{m['cons_all_mean']:>12.5f} | {death_str:>12}")
    print()


def print_insights(metrics: Dict, thetas: List[float]):
    """Print key insights from the simulation."""
    print("=" * 90)
    print("KEY INSIGHTS")
    print("=" * 90)
    
    theta_bold = min(thetas)
    theta_cautious = max(thetas)
    
    # Calculate key comparisons using ALL agents
    all_novelties = [metrics[t]['novelty_all_mean'] for t in thetas 
                     if not np.isnan(metrics[t]['novelty_all_mean'])]
    novelty_cv = np.std(all_novelties) / np.mean(all_novelties) * 100 if len(all_novelties) > 1 else 0
    
    all_skills = [metrics[t]['skill_all_mean'] for t in thetas
                  if not np.isnan(metrics[t]['skill_all_mean'])]
    skill_cv = np.std(all_skills) / np.mean(all_skills) * 100 if len(all_skills) > 1 else 0
    
    # Consequence ratio
    cons_bold = metrics[theta_bold]['cons_all_mean']
    cons_cautious = metrics[theta_cautious]['cons_all_mean']
    cons_ratio = cons_bold / cons_cautious if cons_cautious > 0 else 0
    
    # Survival ratio
    surv_bold = metrics[theta_bold]['survival_rate']
    surv_cautious = metrics[theta_cautious]['survival_rate']
    surv_ratio = surv_cautious / surv_bold if surv_bold > 0 else float('inf')
    
    # Get skill info
    def fmt_skill(theta, skill_type):
        val = metrics[theta][f'{skill_type}_skill_all_mean']
        if np.isnan(val):
            return "N/A"
        return f"{val:.1f}"
    
    phys_bold = fmt_skill(theta_bold, 'phys')
    mental_bold = fmt_skill(theta_bold, 'mental')
    phys_cautious = fmt_skill(theta_cautious, 'phys')
    mental_cautious = fmt_skill(theta_cautious, 'mental')
    
    print(f"""
1. OUTCOMES BY FEAR SENSITIVITY
   - Novelty: {min(all_novelties):.0f} - {max(all_novelties):.0f} (CV = {novelty_cv:.1f}%)
   - Skill: {min(all_skills):.1f} - {max(all_skills):.1f} (CV = {skill_cv:.1f}%)
   - {'Similar outcomes' if novelty_cv < 20 else 'Varied outcomes'} across θ values

2. RISK-REWARD TRADEOFF
   - θ={theta_bold} (bold): {surv_bold:.0%} survival, avg consequence {cons_bold:.4f}
   - θ={theta_cautious} (cautious): {surv_cautious:.0%} survival, avg consequence {cons_cautious:.5f}
   - Bold face {cons_ratio:.0f}x higher consequence → {surv_ratio:.1f}x lower survival

3. TWO-SKILL MODEL: DIFFERENT CAPABILITIES
   - θ={theta_bold}: Physical {phys_bold} + Mental {mental_bold}
   - θ={theta_cautious}: Physical {phys_cautious} + Mental {mental_cautious}
   - Bold climbers develop mental fortitude through consequence exposure
   - Cautious climbers focus on technical mastery with lower mortality

4. THE TRADEOFF
   - θ determines WHERE you climb and WHAT risks you accept
   - Lower θ → higher consequence tolerance → more deaths but different skill profile
   - Higher θ → lower consequence tolerance → higher survival but limited territory
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run strategic agent exploration')
    parser.add_argument('--quick', action='store_true', help='Quick mode: 50 runs per θ')
    parser.add_argument('--full', action='store_true', help='Full mode: 200 runs per θ')
    parser.add_argument('--max-attempts', type=int, default=100, 
                        help='Max attempts per agent (default: 100)')
    parser.add_argument('--plot', action='store_true', help='Generate visualization')
    parser.add_argument('--heatmap', action='store_true', help='Generate exploration heatmaps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='output/strategic_learning_curves.png',
                        help='Output path for visualization')
    parser.add_argument('--landscape', type=str, default='compressed', 
                        choices=['fair', 'stratified', 'compressed'],
                        help='Landscape type (default: compressed)')
    parser.add_argument('--thetas', type=str, default='2.0,4.0,6.0,8.0',
                        help='Comma-separated θ values')
    parser.add_argument('--load-weights', type=str, default=None,
                        help='Load evolved weights from JSON file (from run_evolution.py --save)')
    args = parser.parse_args()
    
    print("=" * 90)
    print("ALEX CLIMBER v2 - STRATEGIC AGENT EXPLORATION")
    print("Using Strategic Agent (unlock/freshness/diversity weighting)")
    print("=" * 90)
    print()
    
    # Configuration
    if args.quick:
        n_runs = 50
        mode = "Quick (50 runs per θ)"
    elif args.full:
        n_runs = 200
        mode = "Full (200 runs per θ)"
    else:
        n_runs = 100
        mode = "Standard (100 runs per θ)"
    
    thetas = [float(t.strip()) for t in args.thetas.split(',')]
    
    print(f"Mode: {mode}")
    print(f"Max attempts per agent: {args.max_attempts}")
    print(f"Random seed: {args.seed}")
    print(f"Landscape: {args.landscape}")
    print(f"θ values: {thetas}")
    print()
    
    # Create landscape
    random.seed(args.seed)
    if args.landscape == 'fair':
        landscape = Landscape.create_fair(thetas=thetas)
    elif args.landscape == 'stratified':
        landscape = Landscape.create_stratified_coverage(seed=args.seed)
    else:  # compressed
        landscape = Landscape.create_compressed()
    
    print("LANDSCAPE")
    print("-" * 90)
    print(f"Name: {landscape.name}")
    print(f"Total routes: {len(landscape.routes)}")
    cons_min = min(r.consequence for r in landscape.routes)
    cons_max = max(r.consequence for r in landscape.routes)
    diff_min = min(r.difficulty for r in landscape.routes)
    diff_max = max(r.difficulty for r in landscape.routes)
    print(f"Difficulty range: {diff_min:.2f} - {diff_max:.2f}")
    print(f"Consequence range: {cons_min:.5f} - {cons_max:.4f}")
    print()
    
    # Load evolved weights if provided
    strategy_configs = None
    if args.load_weights:
        print("EVOLVED WEIGHTS")
        print("-" * 90)
        strategy_configs = load_evolved_weights(args.load_weights)
        print(f"Loaded from: {args.load_weights}")
        for theta in sorted(strategy_configs.keys()):
            if theta in thetas:
                s = strategy_configs[theta]
                print(f"  θ={theta:.1f}: unlock={s.unlock_weight:.3f}, fresh={s.freshness_weight:.3f}, diversity={s.diversity_weight:.3f}")
        # Check for missing thetas
        missing = [t for t in thetas if t not in strategy_configs]
        if missing:
            print(f"  Warning: No evolved weights for θ={missing}, using defaults")
        print()
    
    # Run simulations
    print(f"Running {n_runs} strategic explorations per θ...")
    print("-" * 90)
    
    start_time = time.time()
    results = run_strategic_batch(
        thetas=thetas,
        landscape=landscape,
        n_runs=n_runs,
        max_attempts=args.max_attempts,
        seed_base=args.seed,
        strategy_configs=strategy_configs,
    )
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print()
    
    # Compute metrics and print results
    metrics = compute_metrics(results)
    print_results(metrics, thetas)
    print_insights(metrics, thetas)
    
    # Generate visualization
    if args.plot:
        print("Generating visualization...")
        os.makedirs(os.path.dirname(args.output) or 'output', exist_ok=True)
        create_learning_curves(results, metrics, args.output)
    
    # Generate heatmaps
    if args.heatmap:
        output_dir = os.path.dirname(args.output) or 'output'
        
        # Exploration heatmap (frequency)
        heatmap_path = os.path.join(output_dir, 'strategic_exploration_heatmap.png')
        print("Generating exploration heatmap...")
        create_exploration_heatmap(results, landscape, heatmap_path)
        
        # Progression heatmap (timing)
        progression_path = os.path.join(output_dir, 'strategic_progression_heatmap.png')
        print("Generating progression heatmap...")
        create_progression_heatmap(results, landscape, progression_path)


if __name__ == '__main__':
    main()