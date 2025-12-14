"""
Agent module for climbing simulation.

Alex explores a landscape of routes, choosing based on fear-value function.
"""

import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from landscape import Route


@dataclass
class AgentConfig:
    """Configuration for Alex."""
    
    # Constitutional parameter - fear sensitivity
    theta: float = 2.0  # 2.0 is now "normal"
    
    # Fear model parameters - LAUDA LINE MODEL
    # fear = consequence/lauda_line(θ) + k2 * θ * difficulty * habituation * boost
    #
    # lauda_line(θ) = lauda_base * exp(-lauda_k * θ) + lauda_floor
    # This is the max consequence a person with sensitivity θ will accept
    # When consequence = lauda_line, fear from consequence alone = 1.0 (100%)
    #
    # Calibration (based on Niki Lauda's "20% but not 21%" insight):
    # θ=0.5 (Honnold/Lauda): accepts ~27% death risk
    # θ=2.0 (bold): accepts ~12% death risk
    # θ=3.0 (normal): accepts ~7% death risk  
    # θ=6.0 (fearful): accepts ~2% death risk
    
    lauda_base: float = 0.35              # Base coefficient for Lauda line
    lauda_k: float = 0.6                  # Decay rate  
    lauda_floor: float = 0.01             # Minimum (even θ=∞ tolerates ~1%)
    
    difficulty_multiplier: float = 0.5    # k2 - scales difficulty component
    fear_max: float = 1.0                 # constant threshold for everyone
    
    # Learning rates
    alpha: float = 0.3          # Fear habituation rate
    beta: float = 0.5           # Novelty consumption rate  
    gamma: float = 0.05         # Skill growth rate
    
    # Other
    novelty_max: float = 1.0    # Starting novelty for routes
    novelty_noise: float = 0.5  # Per-attempt noise coefficient (scales with difficulty)
    talent_variance: float = 0.15  # Individual talent variance (σ for talent ~ Normal(1, σ))
    
    # Fall effects
    fall_fear_multiplier: float = 1.3   # Fear boost after fall
    failure_novelty_fraction: float = 0.95  # Novelty gained from non-fatal failure (95%)
    
    # Effective difficulty calculation
    mental_load_factor: float = 2.0  # How much consequence adds to effective difficulty
    
    # Value function for route choice (exploration mode)
    optimal_fear: float = 0.75          # Fear level with highest value ("sweet spot")
    fear_tolerance_width: float = 0.25  # Width of the bell curve (how picky)
    exploration_noise: float = 0.1      # Randomness in route selection


@dataclass
class RouteAttempt:
    """Record of a single route attempt."""
    route_name: str
    success: bool
    fall: bool
    death: bool
    fear_before: float
    fear_after: float
    novelty_gained: float
    skill_gained: float


@dataclass
class TrainingHistory:
    """Training history for visualization."""
    timesteps: List[int] = field(default_factory=list)
    fear_over_time: List[float] = field(default_factory=list)
    novelty_over_time: List[float] = field(default_factory=list)
    skill_over_time: List[float] = field(default_factory=list)  # Total skill for backward compat
    physical_skill_over_time: List[float] = field(default_factory=list)
    mental_skill_over_time: List[float] = field(default_factory=list)


class Alex:
    """
    The climbing agent, now curriculum-aware.
    
    Alex progresses through a curriculum, building skill and
    reducing fear through repetition.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.theta = config.theta
        
        # INDIVIDUAL TALENT - drawn once per person
        # Some people naturally extract more meaning from experiences
        # talent ~ Normal(1.0, σ) where σ = talent_variance
        self.talent = max(0.5, random.gauss(1.0, config.talent_variance))
        
        # State - TWO SKILLS MODEL
        self.physical_skill = 0.0  # Trained by difficulty
        self.mental_skill = 0.0    # Trained by consequence exposure
        self.alive = True
        self.total_novelty = 0.0
        
        # Experience per route: route_name -> attempt count
        self.experience: Dict[str, int] = defaultdict(int)
        
        # Route difficulties: route_name -> difficulty (for variety calculation)
        self.route_difficulties: Dict[str, float] = {}
        
        # Fear boost per route (from falls)
        self.fear_boost: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Progression tracking - for novelty bonus
        self.max_difficulty_achieved: float = 0.0  # Hardest route successfully climbed
        
        # Training tracking
        self.history = TrainingHistory()
        self.timestep = 0
        
        # Attempts log
        self.attempts: List[RouteAttempt] = []
    
    @property
    def skill(self) -> float:
        """Total skill (backward compatibility) = physical + mental."""
        return self.physical_skill + self.mental_skill
    
    def compute_fear(self, route: Route) -> float:
        """
        Compute current fear for a route.
        
        FIRST PRINCIPLES MODEL:
        fear = consequence/lauda_line(θ) + k2 × difficulty × (1-confidence) × habituation × boost
        
        θ ONLY affects consequence tolerance (Lauda line).
        
        Difficulty fear is SKILL-RELATIVE via confidence:
        - confidence = skill / (k + skill)  [diminishing returns]
        - Beginner (skill=0): confidence=0, full difficulty fear
        - Intermediate (skill=2): confidence=0.5, half difficulty fear
        - Expert (skill=4): confidence=0.67, 1/3 difficulty fear
        
        This creates natural progression - as skill grows, you seek harder routes
        to maintain the same fear level (your edge keeps moving up).
        """
        exp = self.experience[route.name]
        habituation = math.exp(-self.config.alpha * exp)
        boost = self.fear_boost[route.name]
        
        # Calculate this person's Lauda line (max acceptable consequence)
        lauda_line = (self.config.lauda_base * math.exp(-self.config.lauda_k * self.theta) 
                      + self.config.lauda_floor)
        
        # Fear from consequence - reaches 1.0 at their Lauda line
        consequence_fear = route.consequence / lauda_line
        
        # Fear from difficulty - SKILL RELATIVE with confidence
        # As you gain PHYSICAL skill, routes feel less intimidating
        # confidence = physical_skill / (k + physical_skill) gives diminishing returns
        # At physical_skill=3, confidence ≈ 0.6, so difficulty fear reduced by 60%
        confidence_k = 2.0  # Controls how fast confidence builds
        confidence = self.physical_skill / (confidence_k + self.physical_skill)
        
        difficulty_fear = (self.config.difficulty_multiplier * 
                          route.difficulty * (1 - confidence) * habituation * boost)
        
        fear = consequence_fear + difficulty_fear
        
        return fear
    
    def get_lauda_line(self) -> float:
        """Return this agent's Lauda line (max acceptable consequence)."""
        return (self.config.lauda_base * math.exp(-self.config.lauda_k * self.theta) 
                + self.config.lauda_floor)
    
    def compute_novelty(self, route: Route) -> float:
        """
        Compute remaining novelty for a route.
        
        KEY PRINCIPLE: AT YOUR EDGE = EQUAL MEANING
        
        Fear IS the value function. When you're at 80% fear, you're at your edge
        and extracting maximum meaning from the experience. The difficulty that
        gets you there doesn't matter - what matters is being at YOUR edge.
        
        Formula: novelty = base × ENGAGEMENT × freshness × talent × noise
        
        ENGAGEMENT: Peaks when fear ≈ 80% of your maximum
        - At your sweet spot (fear 70-90%): fully engaged, max novelty
        - Too easy (low fear): bored, less novelty
        - Too scary (near fear_max): stressed, less novelty
        
        This ensures:
        - θ=0.5 at fear=0.80 (on hard routes) → same novelty as
        - θ=5.0 at fear=0.80 (on easy routes)
        
        Both are equally "at their edge" → equal meaning.
        """
        exp = self.experience[route.name]
        
        # Freshness decays with repetition
        freshness = math.exp(-self.config.beta * exp)
        
        # ENGAGEMENT: peaks when fear is ~80% of max
        fear = self.compute_fear(route)
        fear_max = self.fear_max()
        relative_fear = min(fear / fear_max, 1.0)  # Cap at 1.0
        
        # Bell curve centered at optimal fear (0.80)
        optimal_relative = 0.80
        width = 0.30
        engagement = math.exp(-((relative_fear - optimal_relative) / width) ** 2)
        
        # BASE VALUE: constant for everyone (group-relative)
        base_value = 10.0
        
        # NOISE (per-attempt) - scales with fear (higher stakes = more variance)
        noise_std = self.config.novelty_noise * relative_fear
        noise = random.gauss(0, noise_std)
        noise_multiplier = max(0.1, 1.0 + noise)
        
        # TALENT (per-person) - scales with engagement (talent shows when pushed)
        talent_scaling = math.sqrt(engagement)
        effective_talent = 1.0 + (self.talent - 1.0) * talent_scaling
        effective_talent = max(0.5, effective_talent)
        
        novelty = (self.config.novelty_max * base_value * engagement * 
                   freshness * effective_talent * noise_multiplier)
        
        return novelty
    
    def fear_max(self) -> float:
        """Maximum fear Alex will tolerate (constant for everyone)."""
        return self.config.fear_max
    
    def will_attempt(self, route: Route) -> bool:
        """Check if Alex is willing to attempt this route."""
        fear = self.compute_fear(route)
        return fear <= self.fear_max()
    
    def compute_route_value(self, route: Route) -> float:
        """
        Compute the VALUE of attempting a route.
        
        This is the heart of the choice model. Value depends on:
        1. FEAR SWEETNESS - peaks at optimal_fear, drops for "too boring" or "too scary"
        2. NOVELTY POTENTIAL - how much novelty is available
        
        Different θ values have DIFFERENT value functions over the SAME routes
        because they experience different fear levels.
        
        θ=0.5 on easy route: fear=0.2 → "boring" → low value
        θ=5.0 on easy route: fear=0.8 → "exciting!" → high value
        
        θ=0.5 on hard route: fear=0.8 → "exciting!" → high value  
        θ=5.0 on hard route: fear=1.5 → "terrifying" → won't attempt
        
        CONSEQUENCE PREFERENCE (NEW):
        Low θ climbers NEED stakes to feel engaged. A hard boulder might hit
        the right fear level, but feels hollow without real consequence.
        - θ=0.5: needs ~5% consequence to feel fully engaged
        - θ=5.0: satisfied with any consequence level
        """
        fear = self.compute_fear(route)
        
        # Can't attempt if fear exceeds max
        if fear > self.fear_max():
            return 0.0
        
        # FEAR SWEETNESS: Bell curve centered at optimal_fear
        # Value peaks when fear ≈ optimal_fear (e.g., 0.75)
        # Value drops for "too boring" (low fear) or "too scary" (high fear)
        optimal = self.config.optimal_fear
        width = self.config.fear_tolerance_width
        fear_sweetness = math.exp(-((fear - optimal) / width) ** 2)
        
        # STAKES ENGAGEMENT: Low θ needs real consequence to feel satisfied
        # This is separate from fear - a hard boulder has fear but no stakes
        # required_consequence = what this person needs for full engagement
        # θ=0.5 → needs ~5% death rate; θ=5.0 → needs ~0.5% (easily satisfied)
        required_consequence = 0.025 / self.theta  # 0.05 for θ=0.5, 0.005 for θ=5.0
        stakes_ratio = route.consequence / required_consequence
        stakes_engagement = min(1.0, stakes_ratio)  # Caps at 1.0 (fully engaged)
        
        # NOVELTY POTENTIAL: How much can we gain from this route?
        novelty_potential = self.compute_novelty(route)
        
        # Combined value: need BOTH fear sweetness AND stakes engagement
        value = fear_sweetness * stakes_engagement * novelty_potential
        
        return value
    
    def choose_route(self, available_routes: list) -> 'Route':
        """
        Choose which route to attempt from available options.
        
        Uses the fear-based value function to evaluate routes,
        then selects probabilistically (with some exploration noise).
        
        KEY: Only consider routes in the PRODUCTIVE ZONE (fear 0.5-1.0)
        - Too boring (fear < 0.5): not engaging, skip
        - Too scary (fear > 1.0): won't attempt
        """
        if not available_routes:
            return None
        
        # Filter to PRODUCTIVE routes only:
        # - Must be willing to attempt (fear <= max)
        # - Must be engaging enough (fear >= 0.5)
        MIN_FEAR = 0.5  # Below this, route is "boring" - not worth doing
        productive_routes = [
            r for r in available_routes 
            if self.will_attempt(r) and self.compute_fear(r) >= MIN_FEAR
        ]
        
        if not productive_routes:
            return None
        
        # Compute value for each productive route
        values = [self.compute_route_value(r) for r in productive_routes]
        
        # If all values are zero, return None
        if sum(values) == 0:
            return None
        
        # Add exploration noise and normalize to probabilities
        noise = self.config.exploration_noise
        max_val = max(values) if values else 1.0
        noisy_values = [v + random.uniform(0, noise * max_val) for v in values]
        
        # Select proportionally to noisy values
        total = sum(noisy_values)
        if total == 0:
            return random.choice(productive_routes)
        
        r = random.uniform(0, total)
        cumsum = 0
        for route, value in zip(productive_routes, noisy_values):
            cumsum += value
            if r <= cumsum:
                return route
        
        return productive_routes[-1]

    def attempt_route(self, route: Route) -> RouteAttempt:
        """
        Attempt to climb a route.
        
        NOVELTY FROM ATTEMPTS (not just success):
        - Being at your edge IS meaningful, even if you fail
        - First attempt and fall: huge learning, transformative experience
        - Repeated attempts: freshness decays, less novelty each time
        - Success: full novelty (you completed the experience)
        - Failure (alive): partial novelty (you still learned)
        - Death: no novelty (journey ends)
        
        Returns detailed attempt record.
        """
        # Record route difficulty for variety calculation
        self.route_difficulties[route.name] = route.difficulty
        
        fear_before = self.compute_fear(route)
        novelty_available = self.compute_novelty(route)  # Uses current experience
        
        # EFFECTIVE DIFFICULTY: physical + mental load from consequence
        # The same physical moves are HARDER to execute when stakes are high
        # Your body knows the void is there even if your mind is calm
        #
        # effective_diff = physical_diff × (1 + mental_load_factor × consequence)
        # - Boulder (cons=0.01): effective ≈ physical (pure technique)
        # - El Cap (cons=0.10): effective = physical × 1.5 (technique + mental fortitude)
        effective_difficulty = route.difficulty * (1 + self.config.mental_load_factor * route.consequence)
        
        # Compute success probability using EFFECTIVE difficulty
        # TWO-SKILL MODEL:
        # 1. Physical skill determines base success probability
        # 2. Mental skill prevents "choking" under pressure
        
        # Base success from PHYSICAL skill vs route difficulty
        phys_factor = self.physical_skill / (1 + self.physical_skill)  # Diminishing returns
        base_success = 1 - route.difficulty  # Physical difficulty only
        p_physical_success = min(0.99, max(0.05, base_success * (1 + phys_factor)))
        
        # CHOKING: High consequence + low mental skill = likely to choke
        # mental_capacity = how much pressure you can handle (0-1 scale)
        # choke_pressure = consequence scaled by your mental weakness
        mental_capacity = self.mental_skill / (0.5 + self.mental_skill)
        choke_pressure = route.consequence * (1 - mental_capacity) * 5
        p_choke = min(0.5, choke_pressure)  # Cap at 50%
        
        # Combined: must physically succeed AND not choke
        p_success = p_physical_success * (1 - p_choke)
        
        # Attempt the route
        if random.random() < p_success:
            # Success!
            success = True
            fall = False
            death = False
            
            # Update progression tracking with EFFECTIVE difficulty
            # This reflects total mastery (physical + mental)
            if effective_difficulty > self.max_difficulty_achieved:
                self.max_difficulty_achieved = effective_difficulty
            
            # Compute rewards - FULL novelty on success
            fear_after = self.compute_fear(route)
            novelty_gained = novelty_available * (1 + fear_before)
            
            # TWO-SKILL GAINS:
            # Physical skill from difficulty (the moves)
            # Mental skill from consequence (the pressure)
            physical_gained = self.config.gamma * route.difficulty
            mental_gained = self.config.gamma * route.consequence * 10  # Scale up
            
            self.total_novelty += novelty_gained
            self.physical_skill += physical_gained
            self.mental_skill += mental_gained
            skill_gained = physical_gained + mental_gained
            
        else:
            # Failed - did we fall?
            fall = True
            success = False
            
            # Check for death - FIRST PRINCIPLES: consequence IS death probability
            # Boulder (cons=0.001): 0.1% death per fall → safe
            # El Cap (cons=0.10): 10% death per fall → dangerous
            # The Lauda line already filters what routes each θ will attempt
            if random.random() < route.consequence:
                # Fatal fall - no novelty, journey ends
                death = True
                self.alive = False
                fear_after = fear_before
                novelty_gained = 0
                skill_gained = 0
            else:
                # Non-fatal fall - PARTIAL novelty (you still learned!)
                death = False
                
                # Fear increases for this route
                self.fear_boost[route.name] *= self.config.fall_fear_multiplier
                
                fear_after = self.compute_fear(route)
                
                # Partial novelty from failure - the journey matters
                novelty_gained = novelty_available * (1 + fear_before) * self.config.failure_novelty_fraction
                
                # TWO-SKILL GAINS from failure (reduced):
                physical_gained = self.config.gamma * route.difficulty * 0.5
                mental_gained = self.config.gamma * route.consequence * 10 * 0.5
                
                self.total_novelty += novelty_gained
                self.physical_skill += physical_gained
                self.mental_skill += mental_gained
                skill_gained = physical_gained + mental_gained
        
        # Update experience AFTER attempt - freshness decays for ALL outcomes
        # (even death, though it won't matter for that agent)
        self.experience[route.name] += 1
        
        # Record attempt
        attempt = RouteAttempt(
            route_name=route.name,
            success=success,
            fall=fall,
            death=death,
            fear_before=fear_before,
            fear_after=fear_after,
            novelty_gained=novelty_gained,
            skill_gained=skill_gained,
        )
        self.attempts.append(attempt)
        self.timestep += 1
        
        return attempt
    
    def record_state(self, phase_idx: int, route_name: str):
        """Record current state for visualization."""
        current_fear = 0
        if route_name and route_name in [a.route_name for a in self.attempts]:
            # Get most recent fear for this route
            for a in reversed(self.attempts):
                if a.route_name == route_name:
                    current_fear = a.fear_after
                    break
        
        self.history.timesteps.append(self.timestep)
        self.history.fear_over_time.append(current_fear)
        self.history.novelty_over_time.append(self.total_novelty)
        self.history.skill_over_time.append(self.skill)  # Total for backward compat
        self.history.physical_skill_over_time.append(self.physical_skill)
        self.history.mental_skill_over_time.append(self.mental_skill)
    
    def get_summary(self) -> dict:
        """Get summary of agent's exploration."""
        return {
            'theta': self.theta,
            'talent': self.talent,
            'alive': self.alive,
            'total_novelty': self.total_novelty,
            'final_skill': self.skill,  # Total
            'final_physical_skill': self.physical_skill,
            'final_mental_skill': self.mental_skill,
            'total_attempts': len(self.attempts),
            'total_successes': sum(1 for a in self.attempts if a.success),
            'total_falls': sum(1 for a in self.attempts if a.fall),
            'unique_routes': len(set(a.route_name for a in self.attempts)),
            'max_difficulty': self.max_difficulty_achieved,
        }
    
    def __repr__(self):
        status = "alive" if self.alive else "dead"
        return (f"Alex(θ={self.theta:.2f}, {status}, "
                f"novelty={self.total_novelty:.1f}, skill={self.skill:.2f})")


