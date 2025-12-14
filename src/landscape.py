"""
Landscape Module - Environment for agent exploration

Instead of a prescribed curriculum, agents explore a landscape of routes
and CHOOSE which ones to attempt based on their fear-value function.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Route:
    """A climbing route with difficulty and consequence."""
    name: str
    difficulty: float  # Physical difficulty (0-1+)
    consequence: float  # Death probability per fall (0-1)


@dataclass
class Landscape:
    """
    A landscape of climbing routes with varying difficulty and consequence.
    
    Agents explore this landscape freely, choosing routes based on their
    fear-based value function.
    """
    routes: List[Route]
    name: str = "Climbing Landscape"
    
    @classmethod
    def create_grid(
        cls,
        difficulty_range: Tuple[float, float] = (0.02, 0.50),
        consequence_range: Tuple[float, float] = (0.01, 0.35),
        n_difficulty_levels: int = 15,
        n_consequence_levels: int = 10,
    ) -> 'Landscape':
        """
        Create a grid of routes spanning difficulty × consequence space.
        
        This gives agents a full range of options from easy/safe to hard/dangerous.
        """
        routes = []
        
        diff_min, diff_max = difficulty_range
        cons_min, cons_max = consequence_range
        
        for i in range(n_difficulty_levels):
            for j in range(n_consequence_levels):
                # Linear interpolation
                diff = diff_min + (diff_max - diff_min) * i / (n_difficulty_levels - 1)
                cons = cons_min + (cons_max - cons_min) * j / (n_consequence_levels - 1)
                
                route = Route(
                    name=f"R_d{i}_c{j}",
                    difficulty=diff,
                    consequence=cons,
                )
                routes.append(route)
        
        return cls(routes=routes, name=f"Grid {n_difficulty_levels}x{n_consequence_levels}")
    
    @classmethod
    def create_realistic(
        cls,
        n_routes: int = 100,
        seed: int = None,
    ) -> 'Landscape':
        """
        Create a realistic landscape where difficulty and consequence are correlated.
        
        In real climbing, harder routes tend to be more dangerous.
        """
        if seed is not None:
            random.seed(seed)
        
        routes = []
        
        for i in range(n_routes):
            # Base difficulty uniform across range
            base_diff = random.uniform(0.02, 0.50)
            
            # Consequence correlated with difficulty + some noise
            # Harder routes are generally more dangerous
            base_cons = base_diff * 0.6 + random.gauss(0, 0.05)
            cons = max(0.01, min(0.35, base_cons))
            
            route = Route(
                name=f"Route_{i}",
                difficulty=base_diff,
                consequence=cons,
            )
            routes.append(route)
        
        return cls(routes=routes, name=f"Realistic ({n_routes} routes)")
    
    def get_routes_in_range(
        self,
        diff_range: Tuple[float, float] = None,
        cons_range: Tuple[float, float] = None,
    ) -> List[Route]:
        """Get routes within specified difficulty/consequence ranges."""
        result = self.routes
        
        if diff_range:
            result = [r for r in result if diff_range[0] <= r.difficulty <= diff_range[1]]
        if cons_range:
            result = [r for r in result if cons_range[0] <= r.consequence <= cons_range[1]]
        
        return result
    
    def __repr__(self):
        return f"Landscape({self.name}, {len(self.routes)} routes)"
    
    @classmethod
    def create_fair(
        cls,
        thetas: list = [0.5, 1.0, 2.0, 3.0, 5.0],
        base_routes_per_theta: int = 400,
        max_consequence: float = 0.30,
    ) -> 'Landscape':
        """
        Create a LARGE FAIR landscape where each θ has approximately EQUAL 
        access to sweet spot routes.
        
        Compensates for the fact that routes designed for middle thetas
        tend to also be accessible to neighboring thetas, by generating
        more routes for extreme thetas (0.5 and 5.0).
        
        Total routes: ~2000
        """
        from agent import AgentConfig
        
        routes = []
        
        # Compensation multipliers - fine-tuned for equal sweet spot distribution
        compensation = {
            0.5: 1.1,   
            1.0: 0.85,  
            2.0: 0.85,  
            3.0: 1.05,  
            5.0: 1.20,  # More routes for good coverage
        }
        
        # Route style distribution for variety
        # BOULDERS: 30% - very safe, hard moves
        # MIXED: 40% - varying splits
        # BIG WALLS: 30% - scary, moderate moves
        
        for theta in thetas:
            routes_for_theta = int(base_routes_per_theta * compensation.get(theta, 1.0))
            
            # Route distribution: include expert and elite routes for skilled climbers
            n_boulder = int(routes_for_theta * 0.20)
            n_mixed = int(routes_for_theta * 0.25)
            n_bigwall = int(routes_for_theta * 0.20)
            n_expert = int(routes_for_theta * 0.20)  # High difficulty, accessible at skill 2-3
            n_elite = routes_for_theta - n_boulder - n_mixed - n_bigwall - n_expert  # ~15% elite
            
            config = AgentConfig(theta=theta)
            
            # Calculate Lauda line for this theta
            lauda = config.lauda_base * math.exp(-config.lauda_k * theta) + config.lauda_floor
            k2 = config.difficulty_multiplier
            
            # BOULDER routes: cons_fraction 5-15% (very safe, hard moves)
            for i in range(n_boulder):
                target_fear = 0.50 + 0.40 * i / max(1, n_boulder - 1)
                cons_frac = 0.05 + 0.10 * random.random()
                routes.append(_create_route_with_cons_fraction(
                    theta, target_fear, lauda, k2, max_consequence, cons_frac, f"R_t{theta}_boulder{i}"))
            
            # MIXED routes: cons_fraction 30-60%
            for i in range(n_mixed):
                target_fear = 0.50 + 0.40 * i / max(1, n_mixed - 1)
                cons_frac = 0.30 + 0.30 * random.random()
                routes.append(_create_route_with_cons_fraction(
                    theta, target_fear, lauda, k2, max_consequence, cons_frac, f"R_t{theta}_mixed{i}"))
            
            # BIG WALL routes: cons_fraction 70-90% (scary, moderate moves)
            for i in range(n_bigwall):
                target_fear = 0.50 + 0.40 * i / max(1, n_bigwall - 1)
                cons_frac = 0.70 + 0.20 * random.random()
                routes.append(_create_route_with_cons_fraction(
                    theta, target_fear, lauda, k2, max_consequence, cons_frac, f"R_t{theta}_bigwall{i}"))
            
            # EXPERT routes: high fear at skill=0, becomes sweet spot at skill=2-3
            # These have fear 1.0-1.5 at skill=0, requiring skill progression to access
            for i in range(n_expert):
                target_fear = 1.0 + 0.5 * i / max(1, n_expert - 1)  # 1.0 to 1.5
                cons_frac = 0.05 + 0.15 * random.random()  # Low consequence (expert boulders)
                routes.append(_create_route_with_cons_fraction(
                    theta, target_fear, lauda, k2, max_consequence, cons_frac, f"R_t{theta}_expert{i}"))
            
            # ELITE routes: very high fear at skill=0, requires skill=3-4 to access
            # These have fear 1.5-2.5 at skill=0, producing difficulty 3.0-5.0
            for i in range(n_elite):
                target_fear = 1.5 + 1.0 * i / max(1, n_elite - 1)  # 1.5 to 2.5
                cons_frac = 0.03 + 0.07 * random.random()  # Very low consequence (pure boulders)
                routes.append(_create_route_with_cons_fraction(
                    theta, target_fear, lauda, k2, max_consequence, cons_frac, f"R_t{theta}_elite{i}"))
        
        # Shuffle to mix routes from different thetas
        random.shuffle(routes)
        
        return cls(routes=routes, name=f"Fair Landscape ({len(routes)} routes)")


def _create_route_with_cons_fraction(theta: float, target_fear: float, lauda: float, k2: float,
                                      max_consequence: float, cons_fraction: float, name: str) -> Route:
    """Create a route with a SPECIFIC consequence/difficulty split."""
    cons_fear = target_fear * cons_fraction
    diff_fear = target_fear * (1 - cons_fraction)
    
    cons = cons_fear * lauda
    diff = diff_fear / k2
    
    # BOULDER SAFETY BONUS: Routes with low cons_fraction are MUCH safer
    # Real bouldering: falling 2m onto crash pads ≈ 0.01% death
    # Scale consequence down for low cons_fraction routes
    if cons_fraction < 0.20:
        # Boulder: scale down to realistic safety levels
        cons *= 0.1  # 10x safer than the math suggests
    
    # Add small noise
    cons *= (0.90 + 0.20 * random.random())
    diff *= (0.90 + 0.20 * random.random())
    
    # Clamp
    cons = max(0.0001, min(max_consequence, cons))
    diff = max(0.01, min(5.0, diff))  # Allow up to 5.0 for elite climbers
    
    return Route(name=name, difficulty=diff, consequence=cons)


def _create_route_for_fear(theta: float, target_fear: float, lauda: float, k2: float, 
                           max_consequence: float, name: str) -> Route:
    """
    Helper to create a route targeting a specific fear level for a given theta.
    
    NEW MODEL: θ only affects consequence tolerance
    fear = cons/lauda + k2 × diff
    
    For a given θ and target_fear:
    - consequence_fear = cons/lauda 
    - difficulty_fear = k2 × diff (SAME for everyone)
    
    Routes vary from "safe but hard" (boulders) to "scary but moderate" (big walls)
    """
    # Vary the split between consequence and difficulty WIDELY
    # 5% cons = pure bouldering (very hard moves, very safe)
    # 85% cons = big wall (moderate moves, very scary)
    cons_fraction = 0.05 + 0.80 * random.random()  # 5-85% from consequence
    
    cons_fear = target_fear * cons_fraction
    diff_fear = target_fear * (1 - cons_fraction)
    
    # Calculate actual consequence and difficulty
    cons = cons_fear * lauda
    diff = diff_fear / k2
    
    # Add noise for variety
    cons *= (0.85 + 0.3 * random.random())  # 0.85x to 1.15x
    diff *= (0.85 + 0.3 * random.random())
    
    # Clamp to reasonable ranges
    # TRUE bouldering has ~0.01% death per fall
    cons = max(0.0001, min(max_consequence, cons))
    diff = max(0.01, min(5.0, diff))  # Allow up to 5.0 for elite climbers
    
    return Route(name=name, difficulty=diff, consequence=cons)


def create_stratified_coverage(
    n_difficulty_bands: int = 5,
    n_consequence_bands: int = 5,
    routes_per_cell: int = 80,
    difficulty_range: tuple = (0.1, 5.0),
    consequence_range: tuple = (0.001, 0.25),
    diagonal_boost: float = 1.0,
    seed: int = None,
) -> Landscape:
    """
    Create a θ-AGNOSTIC landscape with stratified coverage across
    the difficulty × consequence space.
    
    This landscape is NOT designed for any specific θ. Instead, it ensures
    uniform coverage so that any θ in a reasonable range (1-8) will find
    routes in their sweet spot - IF the thesis holds.
    
    Use this to test robustness of "equal meaning at your edge" without
    baking fairness into the landscape design.
    
    Args:
        n_difficulty_bands: Number of difficulty levels (rows)
        n_consequence_bands: Number of consequence levels (columns)
        routes_per_cell: Base routes per grid cell
        difficulty_range: (min, max) difficulty values
        consequence_range: (min, max) consequence values  
        diagonal_boost: Multiplier for routes where diff ∝ cons (1.0 = no boost)
        seed: Random seed for reproducibility
        
    Returns:
        Landscape with n_difficulty_bands × n_consequence_bands × routes_per_cell routes
        
    Grid structure:
                            CONSEQUENCE
                     Very Low → → → → High
                  ┌────────┬────────┬────────┐
        Easy      │  cell  │  cell  │  cell  │
                  ├────────┼────────┼────────┤
        DIFF      │  cell  │  cell  │  cell  │
                  ├────────┼────────┼────────┤
        Hard      │  cell  │  cell  │  cell  │
                  └────────┴────────┴────────┘
                  
    Each cell contains routes_per_cell routes with uniform random
    sampling within the cell's bounds.
    """
    if seed is not None:
        random.seed(seed)
    
    routes = []
    
    diff_min, diff_max = difficulty_range
    cons_min, cons_max = consequence_range
    
    # Use log scale for consequence (spans 0.001 to 0.25 = 2.5 orders of magnitude)
    log_cons_min = math.log10(cons_min)
    log_cons_max = math.log10(cons_max)
    
    # Define band edges
    diff_edges = [diff_min + (diff_max - diff_min) * i / n_difficulty_bands 
                  for i in range(n_difficulty_bands + 1)]
    cons_edges = [10 ** (log_cons_min + (log_cons_max - log_cons_min) * i / n_consequence_bands)
                  for i in range(n_consequence_bands + 1)]
    
    route_id = 0
    
    for i in range(n_difficulty_bands):
        diff_lo, diff_hi = diff_edges[i], diff_edges[i + 1]
        
        for j in range(n_consequence_bands):
            cons_lo, cons_hi = cons_edges[j], cons_edges[j + 1]
            
            # Optional diagonal boost: more routes where difficulty ~ consequence
            # (i.e., harder routes tend to be more dangerous)
            is_diagonal = abs(i - j) <= 1  # On or adjacent to diagonal
            cell_routes = routes_per_cell
            if diagonal_boost != 1.0 and is_diagonal:
                cell_routes = int(routes_per_cell * diagonal_boost)
            
            # Generate routes uniformly within this cell
            for k in range(cell_routes):
                # Uniform in difficulty
                diff = random.uniform(diff_lo, diff_hi)
                
                # Uniform in LOG consequence (so we get even coverage across orders of magnitude)
                log_cons = random.uniform(math.log10(cons_lo), math.log10(cons_hi))
                cons = 10 ** log_cons
                
                # Small noise to avoid exact grid alignment
                diff *= (0.95 + 0.10 * random.random())
                cons *= (0.95 + 0.10 * random.random())
                
                # Clamp to valid ranges
                diff = max(0.01, min(5.0, diff))
                cons = max(0.0001, min(0.30, cons))
                
                # Name encodes position for debugging
                style = _get_style_name(i, j, n_difficulty_bands, n_consequence_bands)
                route = Route(
                    name=f"S_{style}_{route_id}",
                    difficulty=diff,
                    consequence=cons,
                )
                routes.append(route)
                route_id += 1
    
    random.shuffle(routes)
    
    return Landscape(
        routes=routes, 
        name=f"Stratified Coverage ({len(routes)} routes, {n_difficulty_bands}x{n_consequence_bands} grid)"
    )


def _get_style_name(diff_band: int, cons_band: int, n_diff: int, n_cons: int) -> str:
    """Generate a descriptive style name based on grid position."""
    # Difficulty names
    if n_diff == 5:
        diff_names = ['easy', 'moderate', 'hard', 'expert', 'elite']
    else:
        diff_names = [f'd{i}' for i in range(n_diff)]
    
    # Consequence names  
    if n_cons == 5:
        cons_names = ['safe', 'low', 'mid', 'high', 'extreme']
    else:
        cons_names = [f'c{i}' for i in range(n_cons)]
    
    diff_name = diff_names[diff_band] if diff_band < len(diff_names) else f'd{diff_band}'
    cons_name = cons_names[cons_band] if cons_band < len(cons_names) else f'c{cons_band}'
    
    return f"{diff_name}_{cons_name}"


# Add to Landscape class as a classmethod wrapper
Landscape.create_stratified_coverage = classmethod(lambda cls, **kwargs: create_stratified_coverage(**kwargs))


def create_compressed_landscape(
    n_difficulty_bands: int = 5,
    n_consequence_bands: int = 5,
    routes_per_cell: int = 80,
    difficulty_range: tuple = (0.1, 5.0),
    seed: int = None,
) -> Landscape:
    """
    Create a COMPRESSED landscape where the entire grid is explorable.
    
    Uses ULTRA-TIGHT consequence bands (0.1%-3.5%) designed so that:
    
    - C1 (0.10%-0.40%): Accessible to ALL θ values
    - C2 (0.40%-0.70%): Accessible to θ≤7
    - C3 (0.70%-1.20%): Accessible to θ≤6
    - C4 (1.20%-2.00%): Accessible to θ≤5
    - C5 (2.00%-3.50%): Accessible to θ≤4
    
    This ensures agents of different θ values explore different regions
    of the grid, creating visible "territory" patterns in the heatmap.
    
    The bands are calculated based on fear < 1.0 at the band MIDPOINT,
    not just the band minimum, ensuring agents actually explore there.
    
    Args:
        n_difficulty_bands: Number of difficulty levels (default 5)
        n_consequence_bands: Number of consequence levels (default 5)
        routes_per_cell: Routes per grid cell (default 80)
        difficulty_range: (min, max) difficulty values
        seed: Random seed for reproducibility
        
    Returns:
        Landscape with 2000 routes (5×5×80) spanning explorable space
    """
    if seed is not None:
        random.seed(seed)
    
    routes = []
    
    # Difficulty range (same as before)
    diff_min, diff_max = difficulty_range
    
    # ULTRA-TIGHT consequence bands
    # Designed so band MIDPOINTS have fear < 1.0 for target θ
    consequence_edges = [
        0.001,   # C1 min: 0.10%
        0.004,   # C1/C2 boundary: 0.40%
        0.007,   # C2/C3 boundary: 0.70%
        0.012,   # C3/C4 boundary: 1.20%
        0.020,   # C4/C5 boundary: 2.00%
        0.035,   # C5 max: 3.50%
    ]
    
    # Difficulty bands (linear spacing)
    diff_edges = [diff_min + (diff_max - diff_min) * i / n_difficulty_bands 
                  for i in range(n_difficulty_bands + 1)]
    
    route_id = 0
    
    for i in range(n_difficulty_bands):
        diff_lo, diff_hi = diff_edges[i], diff_edges[i + 1]
        
        for j in range(n_consequence_bands):
            cons_lo, cons_hi = consequence_edges[j], consequence_edges[j + 1]
            
            # Generate routes uniformly within this cell
            for k in range(routes_per_cell):
                # Uniform in difficulty
                diff = random.uniform(diff_lo, diff_hi)
                
                # Uniform in consequence (linear, not log - range is small)
                cons = random.uniform(cons_lo, cons_hi)
                
                # Small noise to avoid exact grid alignment
                diff *= (0.95 + 0.10 * random.random())
                cons *= (0.95 + 0.10 * random.random())
                
                # Clamp to valid ranges
                diff = max(0.01, min(5.0, diff))
                cons = max(0.0001, min(0.05, cons))
                
                # Name encodes position for debugging
                style = _get_style_name(i, j, n_difficulty_bands, n_consequence_bands)
                route = Route(
                    name=f"C_{style}_{route_id}",  # C_ prefix for Compressed
                    difficulty=diff,
                    consequence=cons,
                )
                routes.append(route)
                route_id += 1
    
    random.shuffle(routes)
    
    return Landscape(
        routes=routes, 
        name=f"Compressed Coverage ({len(routes)} routes, {n_difficulty_bands}x{n_consequence_bands} grid)"
    )


# Add to Landscape class
Landscape.create_compressed = classmethod(lambda cls, **kwargs: create_compressed_landscape(**kwargs))


if __name__ == '__main__':
    # Test landscape creation
    grid = Landscape.create_grid()
    print(f"Grid landscape: {grid}")
    print(f"  Difficulty range: {min(r.difficulty for r in grid.routes):.3f} - {max(r.difficulty for r in grid.routes):.3f}")
    print(f"  Consequence range: {min(r.consequence for r in grid.routes):.3f} - {max(r.consequence for r in grid.routes):.3f}")
    
    realistic = Landscape.create_realistic(seed=42)
    print(f"\nRealistic landscape: {realistic}")
    print(f"  Difficulty range: {min(r.difficulty for r in realistic.routes):.3f} - {max(r.difficulty for r in realistic.routes):.3f}")
    print(f"  Consequence range: {min(r.consequence for r in realistic.routes):.3f} - {max(r.consequence for r in realistic.routes):.3f}")
    
    # Test stratified coverage
    print("\n" + "="*70)
    print("STRATIFIED COVERAGE LANDSCAPE")
    print("="*70)
    
    stratified = Landscape.create_stratified_coverage(seed=42)
    print(f"\n{stratified.name}")
    print(f"  Total routes: {len(stratified.routes)}")
    print(f"  Difficulty range: {min(r.difficulty for r in stratified.routes):.3f} - {max(r.difficulty for r in stratified.routes):.3f}")
    print(f"  Consequence range: {min(r.consequence for r in stratified.routes):.4f} - {max(r.consequence for r in stratified.routes):.4f}")
    
    # Show distribution by style
    from collections import Counter
    styles = [r.name.split('_')[1] + '_' + r.name.split('_')[2] for r in stratified.routes]
    style_counts = Counter(styles)
    
    print(f"\n  Routes by cell (difficulty_consequence):")
    
    # Group by difficulty band
    diff_bands = ['easy', 'moderate', 'hard', 'expert', 'elite']
    cons_bands = ['safe', 'low', 'mid', 'high', 'extreme']
    
    # Print header
    print(f"\n  {'':12}", end='')
    for c in cons_bands:
        print(f"{c:>8}", end='')
    print()
    print(f"  {'-'*12}" + "-"*40)
    
    for d in diff_bands:
        print(f"  {d:12}", end='')
        for c in cons_bands:
            key = f"{d}_{c}"
            count = style_counts.get(key, 0)
            print(f"{count:>8}", end='')
        print()
    
    # Show what different θ values would see
    print("\n" + "-"*70)
    print("ACCESSIBILITY BY θ (fear 0.5-1.0 = sweet spot)")
    print("-"*70)
    
    import sys
    sys.path.insert(0, '.')
    from agent import AgentConfig, Alex
    
    for theta in [1.0, 2.0, 4.0, 6.0, 8.0]:
        alex = Alex(AgentConfig(theta=theta))
        
        sweet = 0
        blocked = 0
        boring = 0
        
        for r in stratified.routes:
            fear = alex.compute_fear(r)
            if fear > 1.0:
                blocked += 1
            elif fear < 0.5:
                boring += 1
            else:
                sweet += 1
        
        print(f"  θ={theta}: {sweet:4} sweet spot | {boring:4} boring | {blocked:4} blocked")