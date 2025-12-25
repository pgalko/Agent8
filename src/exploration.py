"""
Exploration Mode - Agents choose their own path

Instead of following a prescribed curriculum, agents explore a landscape
of routes, choosing based on their fear-value function.

This allows novelty to EMERGE from agent choices rather than being
determined by curriculum design.
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math

from landscape import Landscape, Route
from agent import Alex, AgentConfig, RouteAttempt


@dataclass
class ExplorationTrace:
    """Record of an exploration run."""
    theta: float
    alive: bool
    completed: bool  # True if reached max_attempts (vs stopped early due to no routes)
    total_novelty: float
    total_skill: float  # Combined (physical + mental) for backward compat
    physical_skill: float  # Trained by difficulty
    mental_skill: float    # Trained by consequence exposure
    attempts: int
    successes: int
    failures: int
    routes_tried: int  # Unique routes
    max_difficulty_reached: float  # Max effective difficulty ACHIEVED (success)
    max_physical_achieved: float  # Max physical difficulty ACHIEVED (success)
    max_physical_attempted: float  # Max physical difficulty ATTEMPTED (any)
    death_route: Optional[str] = None
    
    # Trajectory for analysis
    novelty_over_time: List[float] = None
    skill_over_time: List[float] = None
    difficulty_over_time: List[float] = None  # Effective difficulty
    physical_diff_over_time: List[float] = None
    fear_over_time: List[float] = None
    consequence_over_time: List[float] = None
    
    # Per-attempt outcome tracking for journey visualization
    # Each entry: "success", "failure", or "fatal"
    outcome_over_time: List[str] = None
    
    # Route-level tracking for heatmaps
    route_attempts: Dict[str, int] = None  # route_name -> attempt count
    route_attempt_times: Dict[str, List[int]] = None  # route_name -> list of attempt numbers


def run_exploration(
    alex: Alex,
    landscape: Landscape,
    max_attempts: int = 500,
    verbose: bool = False,
) -> ExplorationTrace:
    """
    Run an exploration where agent CHOOSES their own routes.
    
    The agent uses their fear-based value function to select routes
    from the landscape. Different θ values will make different choices
    and end up with different outcomes.
    """
    # Trajectory tracking
    novelty_over_time = [0.0]
    skill_over_time = [0.0]
    difficulty_over_time = []  # Effective difficulty
    physical_diff_over_time = []
    fear_over_time = []
    consequence_over_time = []
    outcome_over_time = []  # "success", "failure", or "fatal"
    
    routes_tried = set()
    route_attempt_counts = {}  # Track attempt counts per route for heatmaps
    route_attempt_times = {}   # Track when each route was attempted (attempt numbers)
    successes = 0
    failures = 0
    max_physical_attempted = 0.0
    max_physical_achieved = 0.0
    
    attempt = 0
    
    while alex.alive and attempt < max_attempts:
        # Agent CHOOSES which route to attempt
        route = alex.choose_route(landscape.routes)
        
        if route is None:
            if verbose:
                print(f"  Attempt {attempt}: No acceptable routes found. Stopping.")
            break
        
        routes_tried.add(route.name)
        route_attempt_counts[route.name] = route_attempt_counts.get(route.name, 0) + 1
        
        # Track timing of this attempt
        if route.name not in route_attempt_times:
            route_attempt_times[route.name] = []
        route_attempt_times[route.name].append(attempt)
        
        # Record pre-attempt state
        fear = alex.compute_fear(route)
        fear_over_time.append(fear)
        physical_diff_over_time.append(route.difficulty)
        consequence_over_time.append(route.consequence)
        
        # Effective difficulty = physical × (1 + mental_load × consequence)
        eff_diff = route.difficulty * (1 + alex.config.mental_load_factor * route.consequence)
        difficulty_over_time.append(eff_diff)
        
        # Track max physical attempted (any route tried)
        if route.difficulty > max_physical_attempted:
            max_physical_attempted = route.difficulty
        
        if verbose:
            value = alex.compute_route_value(route)
            print(f"  Attempt {attempt}: {route.name} (diff={route.difficulty:.3f}, "
                  f"eff_diff={eff_diff:.3f}, fear={fear:.2f}, value={value:.3f})")
        
        # Attempt the route
        result = alex.attempt_route(route)
        
        if result.success:
            successes += 1
            outcome_over_time.append("success")
            # Track max physical difficulty achieved (successful climbs only)
            if route.difficulty > max_physical_achieved:
                max_physical_achieved = route.difficulty
        else:
            failures += 1
            # Check if this was fatal
            if not alex.alive:
                outcome_over_time.append("fatal")
            else:
                outcome_over_time.append("failure")
        
        # Record post-attempt state
        novelty_over_time.append(alex.total_novelty)
        skill_over_time.append(alex.skill)
        
        if not alex.alive:
            if verbose:
                print(f"  DEATH on {route.name}")
            return ExplorationTrace(
                theta=alex.theta,
                alive=False,
                completed=False,
                total_novelty=alex.total_novelty,
                total_skill=alex.skill,
                physical_skill=alex.physical_skill,
                mental_skill=alex.mental_skill,
                attempts=attempt + 1,
                successes=successes,
                failures=failures,
                routes_tried=len(routes_tried),
                max_difficulty_reached=alex.max_difficulty_achieved,
                max_physical_achieved=max_physical_achieved,
                max_physical_attempted=max_physical_attempted,
                death_route=route.name,
                novelty_over_time=novelty_over_time,
                skill_over_time=skill_over_time,
                difficulty_over_time=difficulty_over_time,
                physical_diff_over_time=physical_diff_over_time,
                fear_over_time=fear_over_time,
                consequence_over_time=consequence_over_time,
                outcome_over_time=outcome_over_time,
                route_attempts=route_attempt_counts,
                route_attempt_times=route_attempt_times,
            )
        
        attempt += 1
    
    # completed = True only if we reached max_attempts (not stopped due to no routes)
    completed = (attempt >= max_attempts)
    
    return ExplorationTrace(
        theta=alex.theta,
        alive=True,
        completed=completed,
        total_novelty=alex.total_novelty,
        total_skill=alex.skill,
        physical_skill=alex.physical_skill,
        mental_skill=alex.mental_skill,
        attempts=attempt,
        successes=successes,
        failures=failures,
        routes_tried=len(routes_tried),
        max_difficulty_reached=alex.max_difficulty_achieved,
        max_physical_achieved=max_physical_achieved,
        max_physical_attempted=max_physical_attempted,
        novelty_over_time=novelty_over_time,
        skill_over_time=skill_over_time,
        difficulty_over_time=difficulty_over_time,
        physical_diff_over_time=physical_diff_over_time,
        fear_over_time=fear_over_time,
        consequence_over_time=consequence_over_time,
        outcome_over_time=outcome_over_time,
        route_attempts=route_attempt_counts,
        route_attempt_times=route_attempt_times,
    )


def run_exploration_batch(
    thetas: List[float],
    landscape: Landscape,
    n_runs: int = 100,
    max_attempts: int = 500,
    seed_base: int = 0,
) -> Dict[float, List[ExplorationTrace]]:
    """
    Run multiple exploration simulations for each theta.
    
    Returns dict mapping theta -> list of traces.
    """
    results = {theta: [] for theta in thetas}
    
    for theta in thetas:
        for run in range(n_runs):
            random.seed(seed_base + run + int(theta * 1000))
            
            config = AgentConfig(theta=theta)
            alex = Alex(config)
            
            trace = run_exploration(alex, landscape, max_attempts=max_attempts)
            results[theta].append(trace)
    
    return results