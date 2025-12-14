#!/usr/bin/env python3
"""
Alex Climber v2 - Exploration Mode

First Principles Model:
- θ = Death Tolerance (not difficulty sensitivity)
- Effective Difficulty = Physical + Mental Load from Consequence
- Consequence IS Death Probability (no artificial caps)
- Two-Skill Model: Physical (from difficulty) + Mental (from consequence)

Usage:
    python run_exploration.py              # Standard: 100 runs, fair landscape
    python run_exploration.py --quick      # Quick: 50 runs
    python run_exploration.py --full       # Full: 200 runs
    python run_exploration.py --plot       # Generate visualization
    
    # Landscape options:
    python run_exploration.py --landscape stratified   # θ-agnostic landscape (robustness test)
    python run_exploration.py --landscape fair         # θ-aware landscape (default)
    
    # Custom θ values:
    python run_exploration.py --thetas 1.0,2.0,4.0,6.0,8.0
    
    # Combined:
    python run_exploration.py --quick --landscape stratified --thetas 1.0,2.0,4.0,8.0 --plot
"""

import sys
import os
import argparse
import random
import time

sys.path.insert(0, 'src')

import numpy as np
from landscape import Landscape
from agent import Alex, AgentConfig
from exploration import run_exploration_batch, ExplorationTrace
from visualizations import create_learning_curves, create_exploration_heatmap, create_progression_heatmap
from typing import Dict, List


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_header():
    """Print simulation header."""
    print("=" * 90)
    print("ALEX CLIMBER v2 - EXPLORATION MODE")
    print("First Principles: θ = Death Tolerance, Consequence = Death Probability")
    print("=" * 90)
    print()


def print_landscape_info(landscape: Landscape, thetas: List[float]):
    """Print landscape statistics."""
    print("LANDSCAPE")
    print("-" * 90)
    print(f"Name: {landscape.name}")
    print(f"Total routes: {len(landscape.routes)}")
    print(f"Difficulty range: {min(r.difficulty for r in landscape.routes):.2f} - "
          f"{max(r.difficulty for r in landscape.routes):.2f}")
    print(f"Consequence range: {min(r.consequence for r in landscape.routes):.5f} - "
          f"{max(r.consequence for r in landscape.routes):.4f}")
    print()
    
    # Route style breakdown
    boulders = [r for r in landscape.routes if r.consequence < 0.005]
    mixed = [r for r in landscape.routes if 0.005 <= r.consequence < 0.03]
    bigwalls = [r for r in landscape.routes if r.consequence >= 0.03]
    print(f"Route styles: {len(boulders)} boulders (<0.5%), "
          f"{len(mixed)} mixed (0.5-3%), {len(bigwalls)} big walls (>3%)")
    print()
    
    # Route accessibility by θ
    print("Route Accessibility by θ (sweet spot = fear 0.5-0.9):")
    print(f"{'θ':>5} | {'Sweet':>8} | {'Challenge':>10} | {'Blocked':>8} | "
          f"{'Avg Diff':>9} | {'Avg Cons':>10}")
    print("-" * 70)
    
    for theta in thetas:
        config = AgentConfig(theta=theta)
        alex = Alex(config)
        
        sweet = [r for r in landscape.routes if 0.5 <= alex.compute_fear(r) <= 0.9]
        challenge = [r for r in landscape.routes if 0.9 < alex.compute_fear(r) <= 1.0]
        blocked = [r for r in landscape.routes if alex.compute_fear(r) > 1.0]
        
        avg_diff = np.mean([r.difficulty for r in sweet]) if sweet else 0
        avg_cons = np.mean([r.consequence for r in sweet]) if sweet else 0
        
        print(f"{theta:>5.1f} | {len(sweet):>8} | {len(challenge):>10} | "
              f"{len(blocked):>8} | {avg_diff:>9.2f} | {avg_cons:>10.4f}")
    print()


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def safe_mean(data: List[float]) -> float:
    """Return mean or NaN if empty."""
    return np.mean(data) if data else float('nan')

def safe_std(data: List[float], ddof: int = 1) -> float:
    """Return sample std (ddof=1) or NaN if insufficient data."""
    if len(data) < 2:
        return float('nan')
    return np.std(data, ddof=ddof)

def safe_min(data: List[float]) -> float:
    """Return min or NaN if empty."""
    return np.min(data) if data else float('nan')

def safe_max(data: List[float]) -> float:
    """Return max or NaN if empty."""
    return np.max(data) if data else float('nan')


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
# RESULTS DISPLAY
# =============================================================================

def fmt(value: float, fmt_spec: str = ".0f", na_str: str = "N/A") -> str:
    """Format a value, returning na_str if NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return na_str
    return f"{value:{fmt_spec}}"

def fmt_with_n(value: float, n: int, fmt_spec: str = ".0f", na_str: str = "N/A", warn_threshold: int = 5) -> str:
    """Format a value with asterisk if sample size < warn_threshold."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return na_str
    suffix = "*" if n < warn_threshold else ""
    return f"{value:{fmt_spec}}{suffix}"


def print_results(metrics: Dict, thetas: List[float]):
    """Print comprehensive results tables."""
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    print()
    
    # --- SURVIVAL TABLE ---
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


def print_expected_value(metrics: Dict, thetas: List[float]):
    """Print risk-adjusted expected value analysis."""
    print("=" * 90)
    print("EXPECTED VALUE ANALYSIS")
    print("=" * 90)
    print()
    print(f"{'θ':>5} | {'P(survive)':>11} | {'E[Nov|surv]':>14} | {'Risk-Adj':>14} | {'Rank':>6}")
    print("-" * 65)
    
    values = []
    for theta in thetas:
        m = metrics[theta]
        p_surv = m['survival_rate']
        e_nov = m['novelty_surv_mean']
        risk_adj = p_surv * e_nov
        values.append((theta, p_surv, e_nov, risk_adj))
    
    values.sort(key=lambda x: x[3], reverse=True)
    
    for rank, (theta, p_surv, e_nov, risk_adj) in enumerate(values, 1):
        print(f"{theta:>5.1f} | {p_surv:>11.1%} | {e_nov:>14.0f} | {risk_adj:>14.0f} | {rank:>6}")
    
    best = values[0]
    print()
    print(f"Optimal θ = {best[0]} (maximizes survival × novelty = {best[3]:.0f})")
    print()


def print_insights(metrics: Dict, thetas: List[float]):
    """Print key insights from the simulation."""
    print("=" * 90)
    print("KEY INSIGHTS")
    print("=" * 90)
    
    theta_bold = min(thetas)  # Boldest θ
    theta_cautious = max(thetas)  # Most cautious θ
    
    # Calculate key comparisons using ALL agents (not just survivors)
    all_novelties = [metrics[t]['novelty_all_mean'] for t in thetas 
                     if not np.isnan(metrics[t]['novelty_all_mean'])]
    novelty_cv = np.std(all_novelties) / np.mean(all_novelties) * 100 if len(all_novelties) > 1 else 0
    
    all_skills = [metrics[t]['skill_all_mean'] for t in thetas
                  if not np.isnan(metrics[t]['skill_all_mean'])]
    skill_cv = np.std(all_skills) / np.mean(all_skills) * 100 if len(all_skills) > 1 else 0
    
    # Consequence ratio (bold vs cautious)
    cons_bold = metrics[theta_bold]['cons_all_mean']
    cons_cautious = metrics[theta_cautious]['cons_all_mean']
    cons_ratio = cons_bold / cons_cautious if cons_cautious > 0 else 0
    
    # Survival ratio
    surv_bold = metrics[theta_bold]['survival_rate']
    surv_cautious = metrics[theta_cautious]['survival_rate']
    surv_ratio = surv_cautious / surv_bold if surv_bold > 0 else float('inf')
    
    # Get skill info using ALL agents
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
# VISUALIZATION - See src/visualizations.py for all plotting functions
# =============================================================================


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Alex Climber exploration simulations')
    parser.add_argument('--quick', action='store_true', help='Quick mode: 50 runs per θ')
    parser.add_argument('--full', action='store_true', help='Full mode: 200 runs per θ')
    parser.add_argument('--max-attempts', type=int, default=100, 
                        help='Max attempts per agent (default: 100)')
    parser.add_argument('--plot', action='store_true', help='Generate visualization')
    parser.add_argument('--heatmap', action='store_true', help='Generate exploration heatmap')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='output/learning_curves.png',
                        help='Output path for visualization')
    parser.add_argument('--landscape', type=str, default='fair', 
                        choices=['fair', 'stratified', 'compressed'],
                        help='Landscape type: fair (θ-aware), stratified (θ-agnostic), or compressed (full grid explorable)')
    parser.add_argument('--thetas', type=str, default='0.5,1.0,2.0,3.0,5.0',
                        help='Comma-separated θ values (default: 0.5,1.0,2.0,3.0,5.0)')
    args = parser.parse_args()

    print_header()

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
    
    # Parse theta values
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
    elif args.landscape == 'compressed':
        landscape = Landscape.create_compressed(seed=args.seed)
    else:
        landscape = Landscape.create_fair(thetas=thetas)
    
    print_landscape_info(landscape, thetas)

    # Run simulations
    print(f"Running {n_runs} explorations per θ...")
    print("-" * 90)
    
    start = time.time()
    results = run_exploration_batch(
        thetas=thetas,
        landscape=landscape,
        n_runs=n_runs,
        max_attempts=args.max_attempts,
        seed_base=args.seed,
    )
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print()

    # Compute and display metrics
    metrics = compute_metrics(results)
    print_results(metrics, thetas)
    print_expected_value(metrics, thetas)
    print_insights(metrics, thetas)

    # Generate visualization
    if args.plot:
        print("Generating visualization...")
        create_learning_curves(results, metrics, args.output)

    # Generate exploration heatmap
    if args.heatmap:
        import os
        output_dir = os.path.dirname(args.output) or 'output'
        
        # Frequency heatmap (where agents spend time)
        heatmap_path = os.path.join(output_dir, 'exploration_heatmap.png')
        print("Generating exploration heatmap...")
        create_exploration_heatmap(results, landscape, heatmap_path)
        
        # Progression heatmap (when agents visit each area)
        progression_path = os.path.join(output_dir, 'progression_heatmap.png')
        print("Generating progression heatmap...")
        create_progression_heatmap(results, landscape, progression_path)


if __name__ == '__main__':
    main()