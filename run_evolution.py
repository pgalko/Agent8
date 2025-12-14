#!/usr/bin/env python3
"""
Alex Climber v2 - Evolutionary Strategy Optimization

Finds optimal strategy weights through evolutionary optimization.
Evolves [unlock_weight, freshness_weight, diversity_weight] for each θ.

Usage:
    python run_evolution.py                         # Standard optimization
    python run_evolution.py --quick                 # Quick test (fewer generations)
    python run_evolution.py --full                  # Full optimization
    python run_evolution.py --thetas 2,4,6,8       # Optimize for specific θ values
    python run_evolution.py --plot                  # Generate visualization

Example:
    python run_evolution.py --quick --thetas 4,6,8 --plot
"""

import sys
import os
import argparse
import random
import json
import numpy as np
from typing import Dict, List

sys.path.insert(0, 'src')

from landscape import Landscape
from evolution import (
    EvolutionaryOptimizer, 
    EvolutionConfig, 
    optimize_all_thetas,
    create_evolution_visualization,
    Individual,
)
from strategic_agent import StrategicAlex, StrategyConfig
from agent import Alex, AgentConfig
from exploration import run_exploration, ExplorationTrace
from visualizations import create_exploration_heatmap, create_progression_heatmap


def save_evolved_weights(results: Dict, output_path: str):
    """
    Save evolved weights to a JSON file for later use.
    
    The file can be loaded with load_evolved_weights() in run_strategic_exploration.py
    """
    weights_data = {}
    
    for theta in sorted(results.keys()):
        ind = results[theta][2]  # Individual object
        weights_data[str(theta)] = {
            'theta': theta,
            'raw_weights': ind.genes,
            'magnitude': ind.magnitude,
            'direction': ind.direction,
            'strategy_type': ind.strategy_type,
            'performance': {
                'avg_novelty': ind.avg_novelty,
                'avg_survival': ind.avg_survival,
                'avg_skill': ind.avg_skill,
                'avg_diversity': ind.avg_diversity,
            }
        }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    print(f"\nEvolved weights saved to: {output_path}")
    print(f"Load with: python run_strategic_exploration.py --load-weights {output_path}")


def evaluate_baseline_strategies(
    landscape: Landscape,
    thetas: List[float],
    n_runs: int,
    seed: int,
) -> Dict:
    """Evaluate greedy and hand-tuned baselines for comparison."""
    baseline_data = {}
    
    for theta in thetas:
        greedy_result = evaluate_strategy(
            landscape, theta,
            StrategyConfig(unlock_weight=0.0, freshness_weight=0.0, diversity_weight=0.0),
            n_runs, seed
        )
        handtuned_result = evaluate_strategy(
            landscape, theta,
            StrategyConfig(unlock_weight=0.2, freshness_weight=0.1, diversity_weight=0.1),
            n_runs, seed
        )
        baseline_data[theta] = {
            'greedy': greedy_result,
            'handtuned': handtuned_result,
        }
    
    return baseline_data


def print_header():
    print("=" * 80)
    print("ALEX CLIMBER v2 - EVOLUTIONARY STRATEGY OPTIMIZATION")
    print("Finding Optimal Weights Through Evolution")
    print("=" * 80)
    print()


def print_comparison_table(results: Dict, baseline_data: Dict):
    """
    Print table comparing evolved vs hand-tuned vs greedy performance.
    """
    print("\n" + "=" * 115)
    print("PERFORMANCE COMPARISON: EVOLVED vs HAND-TUNED vs GREEDY")
    print("=" * 115)
    print()
    
    # Headers - U=Unlock, F=Freshness, D=Diversity (strategy behaviors)
    print(f"{'θ':>5} | {'Method':>12} | {'Mag':>5} | {'U%':>4} | {'F%':>4} | {'D%':>4} | {'Novelty':>10} | {'Survival':>10} | {'Skill':>8} | {'Cells':>10}")
    print("-" * 115)
    
    thetas = sorted(results.keys())
    
    for theta in thetas:
        evolved_ind = results[theta][2]  # Individual object
        
        greedy = baseline_data[theta]['greedy']
        handtuned = baseline_data[theta]['handtuned']
        
        # Evolved row with normalized weights
        div_str = f"{evolved_ind.avg_diversity:.1f}/25" if evolved_ind.avg_diversity > 0 else "N/A"
        print(f"{theta:>5.1f} | {'Evolved':>12} | {evolved_ind.magnitude:>5.2f} | {evolved_ind.unlock_pct:>3.0f}% | {evolved_ind.fresh_pct:>3.0f}% | {evolved_ind.diversity_pct:>3.0f}% | {evolved_ind.avg_novelty:>10.0f} | {evolved_ind.avg_survival:>9.1%} | {evolved_ind.avg_skill:>8.2f} | {div_str:>10}")
        
        # Hand-tuned row (magnitude = 0.4, direction = 50/25/25)
        hand_mag = 0.4
        print(f"{'':>5} | {'Hand-tuned':>12} | {hand_mag:>5.2f} | {'50':>3}% | {'25':>3}% | {'25':>3}% | {handtuned['novelty']:>10.0f} | {handtuned['survival']:>9.1%} | {handtuned['skill']:>8.2f} | {'-':>10}")
        
        # Greedy row (magnitude = 0)
        print(f"{'':>5} | {'Greedy':>12} | {'0.00':>5} | {'-':>4} | {'-':>4} | {'-':>4} | {greedy['novelty']:>10.0f} | {greedy['survival']:>9.1%} | {greedy['skill']:>8.2f} | {'-':>10}")
        
        # Improvement rows
        nov_imp_hand = (evolved_ind.avg_novelty - handtuned['novelty']) / handtuned['novelty'] * 100 if handtuned['novelty'] > 0 else 0
        nov_imp_greedy = (evolved_ind.avg_novelty - greedy['novelty']) / greedy['novelty'] * 100 if greedy['novelty'] > 0 else 0
        
        print(f"{'':>5} | {'Δ vs Hand':>12} | {'':>5} | {'':>4} | {'':>4} | {'':>4} | {nov_imp_hand:>+9.1f}% |")
        print(f"{'':>5} | {'Δ vs Greedy':>12} | {'':>5} | {'':>4} | {'':>4} | {'':>4} | {nov_imp_greedy:>+9.1f}% |")
        print("-" * 115)


def evaluate_strategy(
    landscape: Landscape, 
    theta: float, 
    strategy: StrategyConfig, 
    n_runs: int,
    seed: int,
) -> Dict:
    """Evaluate a strategy configuration."""
    novelties = []
    survivals = []
    skills = []
    
    for run in range(n_runs):
        random.seed(seed + run)
        agent = StrategicAlex(AgentConfig(theta=theta), strategy=strategy)
        trace = run_exploration(agent, landscape, max_attempts=100)
        novelties.append(trace.total_novelty)
        # True survival = completed full career
        survivals.append(1.0 if (trace.alive and trace.completed) else 0.0)
        skills.append(trace.total_skill)
    
    return {
        'novelty': np.mean(novelties),
        'survival': np.mean(survivals),
        'skill': np.mean(skills),
    }


def print_optimal_weights_summary(results: Dict):
    """Print summary of optimal normalized weights per θ."""
    print("\n" + "=" * 100)
    print("NORMALIZED STRATEGY WEIGHTS BY θ")
    print("=" * 100)
    print()
    print("Magnitude = how much strategy matters (>2: critical, 1.5-2: important, 1-1.5: moderate, <1: optional)")
    print("Direction = which behaviors matter most (percentages sum to 100%)\n")
    
    print(f"{'θ':>5} | {'Magnitude':>10} | {'Unlock':>8} | {'Fresh':>8} | {'Diversity':>10} | {'Strategy Type':>20}")
    print("-" * 85)
    
    thetas = sorted(results.keys())
    
    for theta in thetas:
        ind = results[theta][2]  # Individual object
        
        # Magnitude interpretation
        if ind.magnitude > 2.0:
            mag_label = "CRITICAL"
        elif ind.magnitude > 1.5:
            mag_label = "IMPORTANT"
        elif ind.magnitude > 1.0:
            mag_label = "MODERATE"
        else:
            mag_label = "OPTIONAL"
        
        print(f"{theta:>5.1f} | {ind.magnitude:>6.2f} ({mag_label:>8}) | {ind.unlock_pct:>7.0f}% | {ind.fresh_pct:>7.0f}% | {ind.diversity_pct:>9.0f}% | {ind.strategy_type:>20}")
    
    print()
    print("Key Insights:")
    
    # Analyze patterns
    bold_thetas = [t for t in thetas if t <= 2]
    fearful_thetas = [t for t in thetas if t >= 6]
    
    if bold_thetas:
        bold_mags = [results[t][2].magnitude for t in bold_thetas]
        avg_bold_mag = np.mean(bold_mags)
        print(f"  • Bold climbers (θ≤2): Avg magnitude {avg_bold_mag:.2f} - strategy is {'CRITICAL' if avg_bold_mag > 2 else 'IMPORTANT'}")
    
    if fearful_thetas:
        fearful_mags = [results[t][2].magnitude for t in fearful_thetas]
        avg_fearful_mag = np.mean(fearful_mags)
        print(f"  • Fearful climbers (θ≥6): Avg magnitude {avg_fearful_mag:.2f} - strategy is {'MODERATE' if avg_fearful_mag > 1 else 'OPTIONAL'}")
    
    # Strategy type distribution
    strategy_counts = {}
    for theta in thetas:
        stype = results[theta][2].strategy_type
        strategy_counts[stype] = strategy_counts.get(stype, 0) + 1
    
    print(f"  • Strategy distribution: {strategy_counts}")


def print_code_snippet(results: Dict):
    """Print Python code snippet for using optimized weights."""
    print("\n" + "=" * 100)
    print("CODE: Using Optimized Weights")
    print("=" * 100)
    print()
    print("# Copy this into your code to use the evolved optimal weights:\n")
    print("OPTIMAL_STRATEGIES = {")
    
    for theta in sorted(results.keys()):
        ind = results[theta][2]
        print(f"    {theta}: {{")
        print(f"        'magnitude': {ind.magnitude:.4f},")
        print(f"        'direction': [{ind.direction[0]:.4f}, {ind.direction[1]:.4f}, {ind.direction[2]:.4f}],  # [unlock, fresh, diversity]")
        print(f"        'raw_weights': [{ind.genes[0]:.4f}, {ind.genes[1]:.4f}, {ind.genes[2]:.4f}],")
        print(f"        'strategy_type': '{ind.strategy_type}',")
        print(f"    }},")
    
    print("}")
    print()
    print("# Usage with raw weights:")
    print("# theta = 6.0")
    print("# weights = OPTIMAL_STRATEGIES[theta]['raw_weights']")
    print("# strategy = StrategyConfig(unlock_weight=weights[0], freshness_weight=weights[1], diversity_weight=weights[2])")
    print("# agent = StrategicAlex(AgentConfig(theta=theta), strategy=strategy)")


def run_evolved_simulations_for_heatmap(
    landscape: Landscape,
    results: Dict,
    n_runs: int = 50,
    max_attempts: int = 100,
    seed: int = 42,
) -> Dict[float, List[ExplorationTrace]]:
    """
    Run simulations using evolved strategies to collect route_attempts for heatmap.
    
    Args:
        landscape: The landscape to use
        results: Dict from optimize_all_thetas containing evolved strategies
        n_runs: Number of runs per θ
        max_attempts: Max attempts per run
        seed: Random seed
        
    Returns:
        Dict mapping θ -> list of ExplorationTrace (with route_attempts populated)
    """
    print("\n" + "=" * 70)
    print("Running simulations with evolved strategies for heatmap...")
    print("=" * 70)
    
    heatmap_results = {}
    
    for theta in sorted(results.keys()):
        # Get evolved strategy for this θ
        evolved_ind = results[theta][2]  # Individual object
        strategy = evolved_ind.to_strategy_config()
        
        traces = []
        for run in range(n_runs):
            random.seed(seed + run + int(theta * 1000))
            agent = StrategicAlex(AgentConfig(theta=theta), strategy=strategy)
            trace = run_exploration(agent, landscape, max_attempts=max_attempts)
            traces.append(trace)
        
        heatmap_results[theta] = traces
        
        # Progress indicator
        completed_count = sum(1 for t in traces if t.alive and t.completed)
        stagnated_count = sum(1 for t in traces if t.alive and not t.completed)
        avg_novelty = np.mean([t.total_novelty for t in traces])
        print(f"  θ={theta}: {n_runs} runs, {completed_count} completed, {stagnated_count} stagnated, avg novelty={avg_novelty:.0f}")
    
    return heatmap_results


def main():
    parser = argparse.ArgumentParser(description='Evolutionary strategy optimization')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode: 10 generations, smaller population')
    parser.add_argument('--full', action='store_true',
                        help='Full mode: 50 generations, larger population')
    parser.add_argument('--thetas', type=str, default='2.0,4.0,6.0,8.0',
                        help='Comma-separated θ values (default: 2.0,4.0,6.0,8.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--heatmap', action='store_true',
                        help='Generate exploration heatmap using evolved strategies')
    parser.add_argument('--output', type=str, default='output/evolution_results.png',
                        help='Output path for visualization')
    parser.add_argument('--landscape', type=str, default='compressed',
                        choices=['fair', 'stratified', 'compressed'],
                        help='Landscape type (default: compressed)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save evolved weights to JSON file (e.g., output/optimal_weights.json)')
    args = parser.parse_args()
    
    print_header()
    
    # Parse thetas
    thetas = [float(t.strip()) for t in args.thetas.split(',')]
    
    # Configure evolution
    if args.quick:
        config = EvolutionConfig(
            population_size=12,
            n_generations=10,
            n_eval_runs=20,
            seed=args.seed,
        )
        mode = "Quick (10 generations, pop=12)"
    elif args.full:
        config = EvolutionConfig(
            population_size=30,
            n_generations=50,
            n_eval_runs=50,
            seed=args.seed,
        )
        mode = "Full (50 generations, pop=30)"
    else:
        config = EvolutionConfig(
            population_size=20,
            n_generations=30,
            n_eval_runs=30,
            seed=args.seed,
        )
        mode = "Standard (30 generations, pop=20)"
    
    print(f"Mode: {mode}")
    print(f"θ values: {thetas}")
    print(f"Landscape: {args.landscape}")
    print(f"Seed: {args.seed}")
    print()
    
    # Create landscape
    random.seed(args.seed)
    if args.landscape == 'fair':
        landscape = Landscape.create_fair(thetas=thetas)
    elif args.landscape == 'compressed':
        landscape = Landscape.create_compressed(seed=args.seed)
    else:
        landscape = Landscape.create_stratified_coverage(seed=args.seed)
    
    print(f"Landscape: {landscape.name}")
    print(f"Total routes: {len(landscape.routes)}")
    
    # Run optimization for each θ
    results = optimize_all_thetas(
        landscape=landscape,
        thetas=thetas,
        config=config,
        verbose=True,
    )
    
    # Evaluate baseline strategies for comparison
    print("\n" + "=" * 70)
    print("Evaluating baseline strategies (greedy, hand-tuned)...")
    print("=" * 70)
    baseline_data = evaluate_baseline_strategies(
        landscape=landscape,
        thetas=thetas,
        n_runs=config.n_eval_runs,
        seed=args.seed,
    )
    
    # Print results
    print_comparison_table(results, baseline_data)
    print_optimal_weights_summary(results)
    print_code_snippet(results)
    
    # Generate visualization
    if args.plot:
        print("\nGenerating evolution visualization...")
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        create_evolution_visualization(results, args.output, baseline_data)
    
    # Generate exploration heatmap using evolved strategies
    if args.heatmap:
        heatmap_results = run_evolved_simulations_for_heatmap(
            landscape=landscape,
            results=results,
            n_runs=config.n_eval_runs,
            max_attempts=config.max_attempts,
            seed=args.seed,
        )
        
        output_dir = os.path.dirname(args.output) or 'output'
        
        # Frequency heatmap
        heatmap_path = os.path.join(output_dir, 'evolution_heatmap.png')
        print("\nGenerating exploration heatmap...")
        create_exploration_heatmap(heatmap_results, landscape, heatmap_path)
        
        # Progression heatmap
        progression_path = os.path.join(output_dir, 'evolution_progression.png')
        print("Generating progression heatmap...")
        create_progression_heatmap(heatmap_results, landscape, progression_path)
    
    # Save evolved weights to JSON
    if args.save:
        save_evolved_weights(results, args.save)


if __name__ == '__main__':
    main()