"""
Evolutionary Strategy Optimizer

Finds optimal strategy weights through evolutionary optimization.

Evolves [unlock_weight, freshness_weight, diversity_weight] (behaviors)
to maximize fitness (novelty + skill).

Key Concepts:
    - Individual: A set of strategy weights [u, f, d]
    - Population: Collection of individuals
    - Fitness: Weighted combination of novelty and skill
    - Selection: Keep top performers
    - Crossover: Blend genes from two parents
    - Mutation: Add random noise to explore

Usage:
    from evolution import EvolutionaryOptimizer
    
    optimizer = EvolutionaryOptimizer(
        landscape=landscape,
        theta=6.0,
        population_size=20,
        n_generations=30,
    )
    best_weights, history = optimizer.run()
"""

import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from agent import Alex, AgentConfig
from landscape import Landscape
from exploration import run_exploration
from strategic_agent import StrategicAlex, StrategyConfig


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
class Individual:
    """An individual in the population - represents a strategy configuration.
    
    Genes are [unlock_weight, freshness_weight, diversity_weight] - the behaviors.
    """
    genes: List[float]  # [unlock_weight, freshness_weight, diversity_weight]
    fitness: float = 0.0
    
    # Detailed metrics from evaluation
    avg_novelty: float = 0.0
    avg_survival: float = 0.0
    avg_skill: float = 0.0
    avg_diversity: float = 0.0  # Unique grid cells visited
    
    def __post_init__(self):
        # Ensure genes are in valid range [0, 1]
        self.genes = [max(0.0, min(1.0, g)) for g in self.genes]
    
    @property
    def magnitude(self) -> float:
        """Total magnitude of strategy weights (how much strategy matters)."""
        return sum(self.genes)
    
    @property
    def direction(self) -> List[float]:
        """Normalized weights (which factors matter most, sums to 1)."""
        mag = self.magnitude
        if mag == 0:
            return [0.33, 0.33, 0.34]
        return [g / mag for g in self.genes]
    
    @property
    def unlock_pct(self) -> float:
        return self.direction[0] * 100
    
    @property
    def fresh_pct(self) -> float:
        return self.direction[1] * 100
    
    @property
    def diversity_pct(self) -> float:
        return self.direction[2] * 100
    
    @property
    def strategy_type(self) -> str:
        """Classify the dominant strategy."""
        d = self.direction
        if d[0] > 0.45:
            return "Unlock-focused"
        elif d[1] > 0.45:
            return "Freshness-focused"
        elif d[2] > 0.45:
            return "Diversity-focused"
        else:
            return "Balanced"
    
    def to_strategy_config(self) -> StrategyConfig:
        """Convert genes to StrategyConfig."""
        return StrategyConfig(
            unlock_weight=self.genes[0],
            freshness_weight=self.genes[1],
            diversity_weight=self.genes[2],
        )
    
    def copy(self) -> 'Individual':
        """Create a copy of this individual."""
        ind = Individual(genes=self.genes.copy())
        ind.fitness = self.fitness
        ind.avg_novelty = self.avg_novelty
        ind.avg_survival = self.avg_survival
        ind.avg_skill = self.avg_skill
        ind.avg_diversity = self.avg_diversity
        return ind


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization.
    
    Fitness weights control WHAT outcomes we value:
    - novelty: total meaning accumulated
    - survival: complete the full career
    - skill: total capability developed
    """
    # Population
    population_size: int = 20
    n_generations: int = 30
    
    # Selection
    elite_ratio: float = 0.1      # Top 10% survive unchanged
    selection_ratio: float = 0.5  # Top 50% can reproduce
    
    # Crossover
    crossover_rate: float = 0.8   # Probability of crossover vs cloning
    
    # Mutation
    mutation_rate: float = 0.3    # Probability of mutating each gene
    mutation_strength: float = 0.15  # Std dev of mutation noise
    
    # Fitness evaluation
    n_eval_runs: int = 30         # Simulations per fitness evaluation
    max_attempts: int = 100       # Max attempts per simulation
    
    # Fitness function weights (what outcomes we optimize for)
    novelty_weight: float = 1.0   # Weight for novelty in fitness
    survival_weight: float = 0.0  # Weight for survival
    skill_weight: float = 0.5     # Weight for skill development
    
    # Misc
    random_injection: float = 0.1  # Fraction of population replaced with random each gen
    seed: int = 42


class EvolutionaryOptimizer:
    """
    Evolutionary optimizer for strategy weights.
    
    Evolves [unlock_weight, freshness_weight, diversity_weight] to maximize
    fitness (novelty + skill) for a given θ.
    """
    
    def __init__(
        self,
        landscape: Landscape,
        theta: float,
        config: EvolutionConfig = None,
        verbose: bool = True,
    ):
        self.landscape = landscape
        self.theta = theta
        self.config = config or EvolutionConfig()
        self.verbose = verbose
        
        # Track evolution history
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_genes': [],
            'diversity': [],
        }
        
        # Best individual found
        self.best_individual: Optional[Individual] = None
        
    def run(self) -> Tuple[List[float], Dict]:
        """
        Run evolutionary optimization.
        
        Returns:
            best_weights: Optimal [unlock_w, fresh_w, diversity_w]
            history: Dict tracking evolution metrics
        """
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Initialize population
        population = self._initialize_population()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"EVOLUTIONARY OPTIMIZATION")
            print(f"{'='*70}")
            print(f"θ = {self.theta}")
            print(f"Population size: {self.config.population_size}")
            print(f"Generations: {self.config.n_generations}")
            print(f"Eval runs per individual: {self.config.n_eval_runs}")
            print(f"{'='*70}\n")
        
        # Evolution loop
        for gen in range(self.config.n_generations):
            # Evaluate fitness
            self._evaluate_population(population, gen)
            
            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best
            if self.best_individual is None or population[0].fitness > self.best_individual.fitness:
                self.best_individual = population[0].copy()
            
            # Record history
            self._record_history(gen, population)
            
            # Print progress
            if self.verbose:
                self._print_generation(gen, population)
            
            # Create next generation (except on last iteration)
            if gen < self.config.n_generations - 1:
                population = self._create_next_generation(population)
        
        if self.verbose:
            self._print_final_results()
        
        return self.best_individual.genes, self.history
    
    def _initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        population = []
        
        for i in range(self.config.population_size):
            # Random weights in [0, 1]
            genes = [
                random.random(),  # unlock_weight
                random.random(),  # freshness_weight
                random.random(),  # diversity_weight
            ]
            population.append(Individual(genes=genes))
        
        # Include our hand-tuned baseline for comparison
        population[0] = Individual(genes=[0.2, 0.1, 0.1])
        
        return population
    
    def _evaluate_population(self, population: List[Individual], generation: int):
        """Evaluate fitness for all individuals."""
        for i, ind in enumerate(population):
            # Skip if already evaluated (elites from previous gen)
            if ind.fitness > 0 and generation > 0:
                continue
            
            fitness, metrics = self._evaluate_individual(ind, generation, i)
            ind.fitness = fitness
            ind.avg_novelty = metrics['avg_novelty']
            ind.avg_survival = metrics['avg_survival']
            ind.avg_skill = metrics['avg_skill']
            ind.avg_diversity = metrics['avg_diversity']
    
    def _evaluate_individual(
        self, 
        individual: Individual, 
        generation: int,
        ind_idx: int,
    ) -> Tuple[float, Dict]:
        """
        Evaluate a single individual by running simulations.
        
        Fitness = weighted combination of novelty, survival, skill, diversity
        """
        strategy = individual.to_strategy_config()
        
        novelties = []
        survivals = []
        skills = []
        diversities = []  # Track unique grid cells visited
        
        for run in range(self.config.n_eval_runs):
            # Unique seed for reproducibility
            run_seed = self.config.seed + generation * 10000 + ind_idx * 100 + run
            random.seed(run_seed)
            
            # Create and run agent
            agent = StrategicAlex(
                AgentConfig(theta=self.theta),
                strategy=strategy,
            )
            trace = run_exploration(
                agent, 
                self.landscape, 
                max_attempts=self.config.max_attempts,
            )
            
            novelties.append(trace.total_novelty)
            # True survival = completed full career (not just alive but stagnated)
            survivals.append(1.0 if (trace.alive and trace.completed) else 0.0)
            skills.append(trace.total_skill)
            
            # Track diversity from agent's visited cells (for reporting)
            diversities.append(len(agent.visited_cells))
        
        # Compute metrics
        avg_novelty = np.mean(novelties)
        avg_survival = np.mean(survivals)
        avg_skill = np.mean(skills)
        avg_diversity = np.mean(diversities)
        
        # Compute fitness (goals we optimize for)
        fitness = (
            self.config.novelty_weight * avg_novelty +
            self.config.survival_weight * avg_survival * 1000 +  # Scale survival
            self.config.skill_weight * avg_skill * 100  # Scale skill
        )
        
        metrics = {
            'avg_novelty': avg_novelty,
            'avg_survival': avg_survival,
            'avg_skill': avg_skill,
            'avg_diversity': avg_diversity,
        }
        
        return fitness, metrics
    
    def _create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """Create next generation through selection, crossover, and mutation."""
        new_population = []
        n_pop = self.config.population_size
        
        # 1. ELITISM: Keep top performers unchanged
        n_elite = max(1, int(n_pop * self.config.elite_ratio))
        for i in range(n_elite):
            new_population.append(population[i].copy())
        
        # 2. SELECTION: Identify individuals that can reproduce
        n_select = max(2, int(n_pop * self.config.selection_ratio))
        parents = population[:n_select]
        
        # 3. Fill rest with offspring
        while len(new_population) < n_pop:
            # Random injection: sometimes add completely random individual
            if random.random() < self.config.random_injection:
                child = Individual(genes=[random.random() for _ in range(3)])
            else:
                # Select two parents (tournament selection)
                parent1 = self._tournament_select(parents)
                parent2 = self._tournament_select(parents)
                
                # Crossover or clone
                if random.random() < self.config.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population[:n_pop]
    
    def _tournament_select(self, parents: List[Individual], k: int = 3) -> Individual:
        """Select parent via tournament selection."""
        tournament = random.sample(parents, min(k, len(parents)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create child by blending parent genes."""
        child_genes = []
        for g1, g2 in zip(parent1.genes, parent2.genes):
            # Blend crossover with random alpha
            alpha = random.random()
            child_genes.append(alpha * g1 + (1 - alpha) * g2)
        return Individual(genes=child_genes)
    
    def _mutate(self, individual: Individual) -> Individual:
        """Apply mutation to individual."""
        mutated_genes = []
        for gene in individual.genes:
            if random.random() < self.config.mutation_rate:
                # Gaussian mutation
                gene += random.gauss(0, self.config.mutation_strength)
                gene = max(0.0, min(1.0, gene))  # Clamp
            mutated_genes.append(gene)
        return Individual(genes=mutated_genes)
    
    def _record_history(self, generation: int, population: List[Individual]):
        """Record metrics for this generation."""
        fitnesses = [ind.fitness for ind in population]
        
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(max(fitnesses))
        self.history['avg_fitness'].append(np.mean(fitnesses))
        self.history['best_genes'].append(population[0].genes.copy())
        
        # Diversity: average pairwise distance
        if len(population) > 1:
            distances = []
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    d = sum((g1 - g2) ** 2 for g1, g2 in 
                            zip(population[i].genes, population[j].genes)) ** 0.5
                    distances.append(d)
            self.history['diversity'].append(np.mean(distances))
        else:
            self.history['diversity'].append(0)
    
    def _print_generation(self, gen: int, population: List[Individual]):
        """Print progress for this generation."""
        best = population[0]
        avg_fit = np.mean([ind.fitness for ind in population])
        
        print(f"Gen {gen:3d} | "
              f"Best: {best.fitness:7.1f} (nov={best.avg_novelty:.0f}, "
              f"surv={best.avg_survival:.0%}) | "
              f"Avg: {avg_fit:7.1f} | "
              f"Genes: [{best.genes[0]:.3f}, {best.genes[1]:.3f}, {best.genes[2]:.3f}]")
    
    def _print_final_results(self):
        """Print final optimization results."""
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        
        best = self.best_individual
        
        print(f"\nBest weights found for θ={self.theta}:")
        print(f"  Raw weights:  [{best.genes[0]:.4f}, {best.genes[1]:.4f}, {best.genes[2]:.4f}]")
        print(f"\n  NORMALIZED:")
        print(f"    Magnitude:    {best.magnitude:.2f} ({'CRITICAL' if best.magnitude > 2 else 'IMPORTANT' if best.magnitude > 1.5 else 'MODERATE' if best.magnitude > 1 else 'OPTIONAL'})")
        print(f"    Unlock:       {best.unlock_pct:.0f}%")
        print(f"    Freshness:    {best.fresh_pct:.0f}%")
        print(f"    Diversity:    {best.diversity_pct:.0f}%")
        print(f"    Strategy:     {best.strategy_type}")
        
        print(f"\nPerformance:")
        print(f"  Avg novelty:  {best.avg_novelty:.1f}")
        print(f"  Survival:     {best.avg_survival:.1%}")
        print(f"  Avg skill:    {best.avg_skill:.2f}")
        print(f"  Avg cells:    {best.avg_diversity:.1f}/25")
        print(f"\nCompare to hand-tuned [0.2, 0.1, 0.1]:")
        
        # Evaluate baseline for comparison
        baseline = Individual(genes=[0.2, 0.1, 0.1])
        baseline_fitness, baseline_metrics = self._evaluate_individual(baseline, -1, -1)
        
        improvement = (best.avg_novelty - baseline_metrics['avg_novelty']) / baseline_metrics['avg_novelty'] * 100
        
        print(f"  Baseline novelty: {baseline_metrics['avg_novelty']:.1f}")
        print(f"  Optimized novelty: {best.avg_novelty:.1f}")
        print(f"  Improvement: {improvement:+.1f}%")


def optimize_all_thetas(
    landscape: Landscape,
    thetas: List[float],
    config: EvolutionConfig = None,
    verbose: bool = True,
) -> Dict[float, Tuple[List[float], Dict]]:
    """
    Run evolutionary optimization for multiple θ values.
    
    Returns dict mapping θ → (best_weights, history)
    """
    config = config or EvolutionConfig()
    results = {}
    
    for theta in thetas:
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# OPTIMIZING θ = {theta}")
            print(f"{'#'*70}")
        
        optimizer = EvolutionaryOptimizer(
            landscape=landscape,
            theta=theta,
            config=config,
            verbose=verbose,
        )
        
        best_weights, history = optimizer.run()
        results[theta] = (best_weights, history, optimizer.best_individual)
    
    return results


def create_evolution_visualization(results: Dict, output_path: str, baseline_data: Dict = None):
    """Create visualization of evolution results with normalized weights."""
    import matplotlib.pyplot as plt
    
    thetas = sorted(results.keys())
    n_thetas = len(thetas)
    
    # Extract data
    magnitudes = []
    directions_unlock = []
    directions_fresh = []
    directions_diversity = []
    evolved_novelty = []
    evolved_survival = []
    
    for theta in thetas:
        best_ind = results[theta][2]  # Individual object
        magnitudes.append(best_ind.magnitude)
        directions_unlock.append(best_ind.direction[0])
        directions_fresh.append(best_ind.direction[1])
        directions_diversity.append(best_ind.direction[2])
        evolved_novelty.append(best_ind.avg_novelty)
        evolved_survival.append(best_ind.avg_survival * 100)
    
    # Get baseline data if provided
    if baseline_data:
        greedy_novelty = [baseline_data[t]['greedy']['novelty'] for t in thetas]
        greedy_survival = [baseline_data[t]['greedy']['survival'] * 100 for t in thetas]
        handtuned_novelty = [baseline_data[t]['handtuned']['novelty'] for t in thetas]
        handtuned_survival = [baseline_data[t]['handtuned']['survival'] * 100 for t in thetas]
    else:
        # Placeholder if no baseline
        greedy_novelty = [0] * n_thetas
        greedy_survival = [0] * n_thetas
        handtuned_novelty = [0] * n_thetas
        handtuned_survival = [0] * n_thetas
    
    # Create figure
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Evolutionary Strategy Optimization Results", fontsize=16, fontweight='bold', y=0.98)
    
    # Color scheme
    color_unlock = '#2ecc71'
    color_fresh = '#e74c3c'
    color_diversity = '#3498db'
    color_evolved = '#9b59b6'
    color_handtuned = '#95a5a6'
    color_greedy = '#bdc3c7'
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_thetas))
    
    x = np.arange(n_thetas)
    
    # =========================================================================
    # Plot 1: Fitness Evolution (top left)
    # =========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    
    for idx, theta in enumerate(thetas):
        history = results[theta][1]
        ax1.plot(history['generation'], history['best_fitness'], 
                color=colors[idx], label=f'θ={theta:.0f}', linewidth=2)
    
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Best Fitness', fontsize=11)
    ax1.set_title('Fitness Evolution', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # =========================================================================
    # Plot 2: Normalized Weights - Stacked Bar (Direction) + Magnitude Line
    # =========================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    
    width = 0.7
    
    # Stacked bar for direction (sums to 1)
    bars1 = ax2.bar(x, directions_unlock, width, label='Unlock', color=color_unlock, alpha=0.9)
    bars2 = ax2.bar(x, directions_fresh, width, bottom=directions_unlock, label='Freshness', color=color_fresh, alpha=0.9)
    bars3 = ax2.bar(x, directions_diversity, width, 
                    bottom=np.array(directions_unlock)+np.array(directions_fresh), 
                    label='Diversity', color=color_diversity, alpha=0.9)
    
    # Magnitude as line on secondary axis
    ax2b = ax2.twinx()
    ax2b.plot(x, magnitudes, 'ko-', linewidth=2.5, markersize=10, label='Magnitude')
    ax2b.set_ylabel('Strategy Magnitude\n(how much strategy matters)', fontsize=10)
    ax2b.set_ylim(0, 3)
    
    ax2.set_xlabel('θ (Fear Sensitivity)', fontsize=11)
    ax2.set_ylabel('Strategy Direction\n(which factors matter)', fontsize=10)
    ax2.set_title('Normalized Strategy Weights', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{t:.0f}' for t in thetas])
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left', fontsize=9)
    ax2b.legend(loc='upper right', fontsize=9)
    
    # =========================================================================
    # Plot 3: Novelty Comparison - Evolved vs Baselines
    # =========================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    
    width = 0.25
    
    bars_greedy = ax3.bar(x - width, greedy_novelty, width, label='Greedy', 
                          color=color_greedy, edgecolor='black', linewidth=0.5)
    bars_hand = ax3.bar(x, handtuned_novelty, width, label='Hand-tuned', 
                        color=color_handtuned, edgecolor='black', linewidth=0.5)
    bars_evolved = ax3.bar(x + width, evolved_novelty, width, label='Evolved', 
                           color=color_evolved, edgecolor='black', linewidth=0.5)
    
    # Add improvement percentages above evolved bars
    if baseline_data:
        for i, (e, g) in enumerate(zip(evolved_novelty, greedy_novelty)):
            if g > 0:
                imp_vs_greedy = (e - g) / g * 100
                ax3.annotate(f'+{imp_vs_greedy:.0f}%', 
                            xy=(x[i] + width, e), 
                            ha='center', va='bottom', fontsize=8, fontweight='bold', color=color_evolved)
    
    ax3.set_xlabel('θ (Fear Sensitivity)', fontsize=11)
    ax3.set_ylabel('Average Novelty', fontsize=11)
    ax3.set_title('Novelty: Evolved vs Baselines', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{t:.0f}' for t in thetas])
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Plot 4: Genetic Diversity Over Time
    # =========================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    
    for idx, theta in enumerate(thetas):
        history = results[theta][1]
        ax4.plot(history['generation'], history['diversity'], 
                color=colors[idx], label=f'θ={theta:.0f}', linewidth=2)
    
    ax4.set_xlabel('Generation', fontsize=11)
    ax4.set_ylabel('Population Diversity', fontsize=11)
    ax4.set_title('Genetic Diversity Over Time', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(alpha=0.3)
    
    # =========================================================================
    # Plot 5: Improvement Breakdown (vs Greedy baseline)
    # =========================================================================
    ax5 = fig.add_subplot(2, 3, 5)
    
    if baseline_data:
        # Calculate improvements
        imp_evolved_vs_greedy = [(e - g) / g * 100 if g > 0 else 0 
                                  for e, g in zip(evolved_novelty, greedy_novelty)]
        imp_hand_vs_greedy = [(h - g) / g * 100 if g > 0 else 0 
                              for h, g in zip(handtuned_novelty, greedy_novelty)]
        
        # Stacked improvement bar
        ax5.bar(x, imp_hand_vs_greedy, width=0.6, label='Hand-tuned improvement', 
                color=color_handtuned, edgecolor='black', linewidth=0.5)
        evolution_boost = np.array(imp_evolved_vs_greedy) - np.array(imp_hand_vs_greedy)
        ax5.bar(x, evolution_boost, width=0.6, bottom=imp_hand_vs_greedy, 
                label='+ Evolution boost', color=color_evolved, edgecolor='black', linewidth=0.5)
        
        # Add total improvement labels
        for i, imp in enumerate(imp_evolved_vs_greedy):
            ax5.annotate(f'{imp:.0f}%', xy=(x[i], imp + 5), ha='center', fontsize=9, fontweight='bold')
    
    ax5.set_xlabel('θ (Fear Sensitivity)', fontsize=11)
    ax5.set_ylabel('Improvement vs Greedy (%)', fontsize=11)
    ax5.set_title('Improvement Breakdown', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{t:.0f}' for t in thetas])
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Plot 6: Survival Improvement
    # =========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    
    width = 0.25
    
    bars_greedy = ax6.bar(x - width, greedy_survival, width, label='Greedy', 
                          color=color_greedy, edgecolor='black', linewidth=0.5)
    bars_hand = ax6.bar(x, handtuned_survival, width, label='Hand-tuned', 
                        color=color_handtuned, edgecolor='black', linewidth=0.5)
    bars_evolved = ax6.bar(x + width, evolved_survival, width, label='Evolved', 
                           color=color_evolved, edgecolor='black', linewidth=0.5)
    
    # Add survival delta annotations for evolved
    if baseline_data:
        for i, (e, g) in enumerate(zip(evolved_survival, greedy_survival)):
            delta = e - g
            if delta > 0:
                ax6.annotate(f'+{delta:.0f}pp', 
                            xy=(x[i] + width, e), 
                            ha='center', va='bottom', fontsize=8, fontweight='bold', color=color_evolved)
    
    ax6.set_xlabel('θ (Fear Sensitivity)', fontsize=11)
    ax6.set_ylabel('Survival Rate (%)', fontsize=11)
    ax6.set_title('Survival: Evolved vs Baselines', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'{t:.0f}' for t in thetas])
    ax6.legend(loc='lower right', fontsize=9)
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_ylim(0, 100)
    
    # =========================================================================
    # Finalize
    # =========================================================================
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    # Quick test
    print("Testing Evolutionary Optimizer...")
    
    landscape = Landscape.create_compressed(seed=42)
    
    config = EvolutionConfig(
        population_size=10,
        n_generations=5,
        n_eval_runs=10,
    )
    
    optimizer = EvolutionaryOptimizer(
        landscape=landscape,
        theta=6.0,
        config=config,
        verbose=True,
    )
    
    best_weights, history = optimizer.run()
    print(f"\nBest weights: {best_weights}")