"""
Visualization module for Alex Climber v2.

This module provides functions for visualizing exploration results,
skill development, and comparing different fear sensitivity (θ) values.

Can be imported and used independently for custom analysis.
All functions are agnostic to simulation parameters - they adapt to whatever
θ values are present in the results.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import exploration trace for type hints
try:
    from .exploration import ExplorationTrace
except ImportError:
    from exploration import ExplorationTrace


def generate_colors(thetas: List[float], colormap: str = 'viridis') -> Dict[float, str]:
    """
    Generate colors dynamically for any set of θ values.
    
    Args:
        thetas: List of θ values to generate colors for
        colormap: Matplotlib colormap name (default: 'viridis')
                  Other good options: 'plasma', 'coolwarm', 'RdYlGn', 'Spectral'
    
    Returns:
        Dict mapping θ -> hex color string
    """
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
    except ImportError:
        # Fallback to simple interpolation if matplotlib not available
        def simple_color(i, n):
            # Simple purple -> yellow gradient
            t = i / max(1, n - 1)
            r = int(123 + (191 - 123) * t)
            g = int(44 + (191 - 44) * t)
            b = int(191 + (44 - 191) * t)
            return f'#{r:02x}{g:02x}{b:02x}'
        return {theta: simple_color(i, len(thetas)) for i, theta in enumerate(sorted(thetas))}
    
    sorted_thetas = sorted(thetas)
    n = len(sorted_thetas)
    
    colors = {}
    for i, theta in enumerate(sorted_thetas):
        # Map index to colormap position (0 to 1)
        position = i / max(1, n - 1)
        rgba = cmap(position)
        # Convert to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255)
        )
        colors[theta] = hex_color
    
    return colors


def aggregate_trajectories(traces: List[ExplorationTrace], 
                          attr: str, 
                          length: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Aggregate trajectory data across multiple traces (all agents, not just survivors).
    
    Dead agents' trajectories are padded with their final value, representing
    that their skill/novelty stops growing after death.
    
    Args:
        traces: List of ExplorationTrace objects
        attr: Attribute name to aggregate (e.g., 'novelty_over_time')
        length: Target length for padding
        
    Returns:
        Tuple of (mean, std) arrays, or (None, None) if no valid data
    """
    valid = [t for t in traces if getattr(t, attr, None)]
    if not valid:
        return None, None
    
    padded = []
    for t in valid:
        traj = list(getattr(t, attr))
        if len(traj) < length:
            # Pad with final value (skill/novelty frozen after death or end)
            traj = traj + [traj[-1]] * (length - len(traj))
        padded.append(traj[:length])
    
    arr = np.array(padded)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def create_learning_curves(results: Dict[float, List[ExplorationTrace]], 
                           metrics: Dict,
                           output_path: str = "output/learning_curves.png",
                           colors: Optional[Dict[float, str]] = None,
                           figsize: Tuple[int, int] = (14, 10),
                           dpi: int = 150):
    """
    Create 4-panel visualization of learning curves.
    
    Panels:
    1. Novelty Accumulation (top-left)
    2. Fear Throughout Training (top-right)
    3. Skill Development (bottom-left)
    4. Difficulty Frontier (bottom-right)
    
    Args:
        results: Dict mapping θ -> list of ExplorationTrace
        metrics: Dict of computed metrics (from compute_metrics)
        output_path: Path to save figure
        colors: Optional dict mapping θ -> color string
        figsize: Figure size tuple
        dpi: Resolution for saved figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    thetas = sorted(results.keys())
    
    if colors is None:
        colors = generate_colors(thetas)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Learning Curves by Fear Sensitivity (θ)", fontsize=14, fontweight='bold')
    
    # Find max trajectory length across ALL agents (for reference, but we'll plot per-theta)
    all_traces = [t for traces in results.values() for t in traces]
    if not all_traces:
        print("No traces to plot!")
        return
    
    # Plot each theta with its own trajectory length (curves end when last agent dies)
    for theta in thetas:
        traces = results[theta]
        color = colors.get(theta, '#333333')
        
        if not traces:
            continue
        
        # Use this theta's max trajectory length
        traj_len = max(len(t.novelty_over_time) for t in traces)
        x = np.arange(traj_len)
        
        # 1. Novelty Accumulation (top-left)
        ax1 = axes[0, 0]
        mean_nov, std_nov = aggregate_trajectories(traces, 'novelty_over_time', traj_len)
        if mean_nov is not None:
            ax1.plot(x, mean_nov, color=color, label=f'θ={theta}', linewidth=2)
            ax1.fill_between(x, mean_nov - std_nov, mean_nov + std_nov, color=color, alpha=0.15)
        
        # 2. Fear Throughout Training (top-right)
        ax2 = axes[0, 1]
        mean_fear, _ = aggregate_trajectories(traces, 'fear_over_time', traj_len - 1)
        if mean_fear is not None:
            ax2.plot(np.arange(len(mean_fear)), mean_fear, color=color, label=f'θ={theta}', linewidth=2)
        
        # 3. Skill Development (bottom-left)
        ax3 = axes[1, 0]
        mean_skill, std_skill = aggregate_trajectories(traces, 'skill_over_time', traj_len)
        if mean_skill is not None:
            ax3.plot(x, mean_skill, color=color, label=f'θ={theta}', linewidth=2)
        
        # 4. Difficulty Frontier (bottom-right) - cumulative max physical difficulty
        ax4 = axes[1, 1]
        
        valid = [t for t in traces if t.physical_diff_over_time]
        if valid:
            cum_maxes = []
            for t in valid:
                diffs = t.physical_diff_over_time
                cum_max = np.maximum.accumulate(diffs)
                cum_maxes.append(cum_max)
            
            # Pad to this theta's trajectory length
            max_len = max(len(cm) for cm in cum_maxes)
            padded = []
            for cm in cum_maxes:
                cm_list = list(cm)
                if len(cm_list) < max_len:
                    cm_list = cm_list + [cm_list[-1]] * (max_len - len(cm_list))
                padded.append(cm_list[:max_len])
            
            mean_frontier = np.mean(padded, axis=0)
            ax4.plot(np.arange(len(mean_frontier)), mean_frontier, color=color, 
                    label=f'θ={theta}', linewidth=2)
    
    # Configure subplots
    ax1.set_title("Novelty Accumulation", fontweight='bold')
    ax1.set_xlabel("Training Attempts")
    ax1.set_ylabel("Cumulative Novelty")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Fear Throughout Training", fontweight='bold')
    ax2.set_xlabel("Training Attempts")
    ax2.set_ylabel("Current Fear")
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Fear Max')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.15)
    
    ax3.set_title("Skill Development", fontweight='bold')
    ax3.set_xlabel("Training Attempts")
    ax3.set_ylabel("Skill Level")
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title("Difficulty Frontier (max achieved)", fontweight='bold')
    ax4.set_xlabel("Training Attempts")
    ax4.set_ylabel("Max Physical Difficulty")
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()
    
    return fig


def create_two_skill_comparison(results: Dict[float, List[ExplorationTrace]],
                                output_path: str = "output/two_skill_comparison.png",
                                colors: Optional[Dict[float, str]] = None,
                                figsize: Tuple[int, int] = (12, 5),
                                dpi: int = 150):
    """
    Create visualization comparing physical vs mental skill development.
    
    Args:
        results: Dict mapping θ -> list of ExplorationTrace
        output_path: Path to save figure
        colors: Optional dict mapping θ -> color string
        figsize: Figure size tuple
        dpi: Resolution for saved figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    thetas = sorted(results.keys())
    
    if colors is None:
        colors = generate_colors(thetas)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Two-Skill Model: Physical vs Mental Development", fontsize=14, fontweight='bold')
    
    # Collect all agent data
    phys_skills = []
    mental_skills = []
    theta_labels = []
    bar_colors = []
    
    for theta in thetas:
        traces = results[theta]
        if traces:
            phys_skills.append(np.mean([t.physical_skill for t in traces]))
            mental_skills.append(np.mean([t.mental_skill for t in traces]))
            theta_labels.append(f'θ={theta}')
            bar_colors.append(colors.get(theta, '#333333'))
    
    if not phys_skills:
        print("No traces to plot!")
        return
    
    x = np.arange(len(theta_labels))
    width = 0.35
    
    # Bar chart comparison
    ax1 = axes[0]
    ax1.bar(x - width/2, phys_skills, width, label='Physical', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, mental_skills, width, label='Mental', color='coral', alpha=0.8)
    ax1.set_xlabel('Fear Sensitivity')
    ax1.set_ylabel('Skill Level')
    ax1.set_title('Skill Levels by Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(theta_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Scatter plot: Physical vs Mental
    ax2 = axes[1]
    for i, theta in enumerate(thetas):
        traces = results[theta]
        if traces:
            phys = [t.physical_skill for t in traces]
            mental = [t.mental_skill for t in traces]
            ax2.scatter(phys, mental, c=colors.get(theta, '#333333'), 
                       label=f'θ={theta}', alpha=0.6, s=50)
    
    ax2.set_xlabel('Physical Skill')
    ax2.set_ylabel('Mental Skill')
    ax2.set_title('Physical vs Mental (each dot = 1 agent)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    max_val = max(max(phys_skills), max(mental_skills)) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal skills')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\nTwo-skill comparison saved to: {output_path}")
    plt.close()
    
    return fig


def create_survival_analysis(results: Dict[float, List[ExplorationTrace]],
                             output_path: str = "output/survival_analysis.png",
                             colors: Optional[Dict[float, str]] = None,
                             figsize: Tuple[int, int] = (12, 5),
                             dpi: int = 150):
    """
    Create survival analysis visualization.
    
    Args:
        results: Dict mapping θ -> list of ExplorationTrace
        output_path: Path to save figure
        colors: Optional dict mapping θ -> color string
        figsize: Figure size tuple
        dpi: Resolution for saved figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    thetas = sorted(results.keys())
    
    if colors is None:
        colors = generate_colors(thetas)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Survival Analysis", fontsize=14, fontweight='bold')
    
    # Survival rates
    survival_rates = []
    for theta in thetas:
        traces = results[theta]
        rate = sum(1 for t in traces if t.alive) / len(traces) * 100
        survival_rates.append(rate)
    
    ax1 = axes[0]
    bars = ax1.bar([f'θ={t}' for t in thetas], survival_rates, 
                   color=[colors.get(t, '#333333') for t in thetas], alpha=0.8)
    ax1.set_ylabel('Survival Rate (%)')
    ax1.set_title('Survival by Fear Sensitivity')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, survival_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=10)
    
    # Death timing histogram
    ax2 = axes[1]
    for theta in thetas:
        dead = [t for t in results[theta] if not t.alive]
        if dead:
            death_attempts = [t.attempts for t in dead]
            ax2.hist(death_attempts, bins=20, alpha=0.5, 
                    color=colors.get(theta, '#333333'), label=f'θ={theta}')
    
    ax2.set_xlabel('Attempt Number at Death')
    ax2.set_ylabel('Count')
    ax2.set_title('When Deaths Occur')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\nSurvival analysis saved to: {output_path}")
    plt.close()
    
    return fig


def create_consequence_analysis(results: Dict[float, List[ExplorationTrace]],
                                output_path: str = "output/consequence_analysis.png",
                                colors: Optional[Dict[float, str]] = None,
                                figsize: Tuple[int, int] = (12, 5),
                                dpi: int = 150):
    """
    Create consequence exposure analysis visualization.
    
    Args:
        results: Dict mapping θ -> list of ExplorationTrace
        output_path: Path to save figure
        colors: Optional dict mapping θ -> color string
        figsize: Figure size tuple
        dpi: Resolution for saved figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    thetas = sorted(results.keys())
    
    if colors is None:
        colors = generate_colors(thetas)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Consequence Exposure Analysis", fontsize=14, fontweight='bold')
    
    # Average consequence over time (all agents, padded after death)
    ax1 = axes[0]
    for theta in thetas:
        traces = [t for t in results[theta] if t.consequence_over_time]
        if traces:
            max_len = max(len(t.consequence_over_time) for t in traces)
            padded = []
            for t in traces:
                cons = list(t.consequence_over_time)
                if len(cons) < max_len:
                    # Pad with final value (consequence exposure stops after death)
                    cons = cons + [cons[-1]] * (max_len - len(cons))
                padded.append(cons[:max_len])
            
            mean_cons = np.mean(padded, axis=0)
            ax1.plot(mean_cons, color=colors.get(theta, '#333333'), 
                    label=f'θ={theta}', linewidth=2)
    
    ax1.set_xlabel('Training Attempts')
    ax1.set_ylabel('Consequence (Death Probability)')
    ax1.set_title('Consequence Exposure Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Consequence distribution (all agents)
    ax2 = axes[1]
    all_cons = []
    positions = []
    colors_list = []
    
    for i, theta in enumerate(thetas):
        traces = [t for t in results[theta] if t.consequence_over_time]
        if traces:
            flat_cons = [c for t in traces for c in t.consequence_over_time]
            all_cons.append(flat_cons)
            positions.append(i)
            colors_list.append(colors.get(theta, '#333333'))
    
    if all_cons:
        bp = ax2.boxplot(all_cons, positions=positions, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.set_xticks(positions)
        ax2.set_xticklabels([f'θ={t}' for t in thetas])
        ax2.set_ylabel('Consequence')
        ax2.set_title('Consequence Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\nConsequence analysis saved to: {output_path}")
    plt.close()
    
    return fig


def create_all_visualizations(results: Dict[float, List[ExplorationTrace]],
                              metrics: Dict,
                              output_dir: str = "output"):
    """
    Create all available visualizations.
    
    Args:
        results: Dict mapping θ -> list of ExplorationTrace
        metrics: Dict of computed metrics
        output_dir: Directory to save all figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating all visualizations...")
    
    create_learning_curves(results, metrics, f"{output_dir}/learning_curves.png")
    create_two_skill_comparison(results, f"{output_dir}/two_skill_comparison.png")
    create_survival_analysis(results, f"{output_dir}/survival_analysis.png")
    create_consequence_analysis(results, f"{output_dir}/consequence_analysis.png")
    
    print(f"\nAll visualizations saved to: {output_dir}/")


# =============================================================================
# EXPLORATION HEATMAP VISUALIZATIONS
# =============================================================================

def create_route_grid_mapping(
    landscape,
    n_diff_bands: int = 5,
    n_cons_bands: int = 5,
) -> Dict[str, Tuple[int, int]]:
    """
    Map each route to its (difficulty_band, consequence_band) position.
    
    Returns:
        Dict mapping route_name -> (diff_band, cons_band)
    """
    # Get difficulty and consequence ranges from routes
    all_diffs = [r.difficulty for r in landscape.routes]
    all_cons = [r.consequence for r in landscape.routes]
    
    diff_min, diff_max = min(all_diffs), max(all_diffs)
    cons_min, cons_max = min(all_cons), max(all_cons)
    
    # Add small epsilon to avoid edge cases
    diff_max += 0.001
    cons_max += 0.001
    
    diff_step = (diff_max - diff_min) / n_diff_bands
    cons_step = (cons_max - cons_min) / n_cons_bands
    
    mapping = {}
    for route in landscape.routes:
        diff_band = min(n_diff_bands - 1, int((route.difficulty - diff_min) / diff_step))
        cons_band = min(n_cons_bands - 1, int((route.consequence - cons_min) / cons_step))
        mapping[route.name] = (diff_band, cons_band)
    
    return mapping


def aggregate_route_attempts(
    results: Dict[float, List],
) -> Dict[float, Dict[str, int]]:
    """
    Aggregate route attempts across all traces for each θ.
    
    Args:
        results: Dict mapping θ -> list of ExplorationTrace
        
    Returns:
        Dict mapping θ -> {route_name: total_attempts}
    """
    aggregated = {}
    
    for theta, traces in results.items():
        combined = {}
        for trace in traces:
            if trace.route_attempts:
                for route_name, count in trace.route_attempts.items():
                    combined[route_name] = combined.get(route_name, 0) + count
        aggregated[theta] = combined
    
    return aggregated


def create_heatmap_matrix(
    route_attempts: Dict[str, int],
    route_grid: Dict[str, Tuple[int, int]],
    landscape,
    n_diff_bands: int = 5,
    n_cons_bands: int = 5,
) -> np.ndarray:
    """
    Create a fine-grained heatmap matrix from route attempts.
    """
    from collections import defaultdict
    
    sub_rows = 8  # 8 rows per cell
    sub_cols = 10  # 10 cols per cell
    
    matrix = np.zeros((n_cons_bands * sub_rows, n_diff_bands * sub_cols))
    
    # Count routes per cell and build list
    cell_route_list = defaultdict(list)
    for route in landscape.routes:
        if route.name in route_grid:
            cell = route_grid[route.name]
            cell_route_list[cell].append(route.name)
    
    # Fill matrix
    for route in landscape.routes:
        if route.name not in route_grid:
            continue
        
        diff_band, cons_band = route_grid[route.name]
        attempts = route_attempts.get(route.name, 0)
        
        if attempts == 0:
            continue
        
        # Find position within cell
        cell_routes = cell_route_list[(diff_band, cons_band)]
        try:
            route_idx = cell_routes.index(route.name)
        except ValueError:
            continue
        
        sub_row = route_idx // sub_cols
        sub_col = route_idx % sub_cols
        
        sub_row = min(sub_row, sub_rows - 1)
        sub_col = min(sub_col, sub_cols - 1)
        
        row = cons_band * sub_rows + sub_row
        col = diff_band * sub_cols + sub_col
        
        matrix[row, col] = attempts
    
    return matrix


def create_exploration_heatmap(
    results: Dict[float, List],
    landscape,
    output_path: str = 'output/exploration_heatmap.png',
    n_diff_bands: int = 5,
    n_cons_bands: int = 5,
):
    """
    Create faceted heatmap visualization showing where each θ explores.
    
    Args:
        results: Dict mapping θ -> list of ExplorationTrace
        landscape: The Landscape object used in simulation
        output_path: Where to save the figure
        n_diff_bands: Number of difficulty bands (default 5)
        n_cons_bands: Number of consequence bands (default 5)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    route_grid = create_route_grid_mapping(landscape, n_diff_bands, n_cons_bands)
    aggregated = aggregate_route_attempts(results)
    
    thetas = sorted(results.keys())
    n_thetas = len(thetas)
    
    if n_thetas == 0:
        print("No data to plot")
        return
    
    # Determine grid layout
    if n_thetas <= 2:
        nrows, ncols = 1, n_thetas
    elif n_thetas <= 4:
        nrows, ncols = 2, 2
    elif n_thetas <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = (n_thetas + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_thetas == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    fig.suptitle('Agent Exploration Patterns by θ', fontsize=14, fontweight='bold')
    
    # Color maps for each θ
    theta_cmaps = {
        1: 'Reds', 2: 'Reds', 3: 'Oranges', 4: 'Oranges',
        5: 'YlGn', 6: 'Greens', 7: 'Blues', 8: 'Blues',
    }
    
    theta_labels = {
        1: 'θ=1 (Very Bold)', 2: 'θ=2 (Bold)', 3: 'θ=3', 4: 'θ=4 (Moderate)',
        5: 'θ=5', 6: 'θ=6 (Cautious)', 7: 'θ=7', 8: 'θ=8 (Fearful)',
    }
    
    # Build matrices and find global max
    all_matrices = []
    for theta in thetas:
        matrix = create_heatmap_matrix(
            aggregated.get(theta, {}), route_grid, landscape, n_diff_bands, n_cons_bands
        )
        all_matrices.append(matrix)
    
    global_max = max(m.max() for m in all_matrices) if all_matrices else 1
    if global_max == 0:
        global_max = 1
    
    sub_rows, sub_cols = 8, 10
    
    for idx, theta in enumerate(thetas):
        ax = axes[idx]
        matrix = all_matrices[idx]
        matrix_norm = matrix / global_max
        
        cmap = theta_cmaps.get(int(theta), 'Purples')
        im = ax.imshow(matrix_norm, origin='lower', cmap=cmap, aspect='equal', vmin=0, vmax=1)
        
        # Draw grid lines
        for i in range(n_diff_bands + 1):
            ax.axvline(x=i * sub_cols - 0.5, color='black', linewidth=1.5, alpha=0.7)
        for j in range(n_cons_bands + 1):
            ax.axhline(y=j * sub_rows - 0.5, color='black', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Difficulty →', fontsize=10)
        ax.set_ylabel('Consequence →', fontsize=10)
        ax.set_title(theta_labels.get(int(theta), f'θ={theta}'), fontsize=11, fontweight='bold')
        
        ax.set_xticks([i * sub_cols + sub_cols/2 for i in range(n_diff_bands)])
        ax.set_xticklabels([f'D{i+1}' for i in range(n_diff_bands)])
        ax.set_yticks([i * sub_rows + sub_rows/2 for i in range(n_cons_bands)])
        ax.set_yticklabels([f'C{i+1}' for i in range(n_cons_bands)])
        
        plt.colorbar(im, ax=ax, label='Attempt Frequency', shrink=0.8)
    
    # Hide unused axes
    for idx in range(len(thetas), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Exploration heatmap saved to: {output_path}")


def aggregate_route_timing(
    results: Dict[float, List],
) -> Dict[float, Dict[str, float]]:
    """
    Aggregate route attempt timing across all traces for each θ.
    
    For each route, computes the average attempt number (normalized by max attempts).
    This shows when in their "career" agents typically visit each route.
    
    Args:
        results: Dict mapping θ -> list of ExplorationTrace
        
    Returns:
        Dict mapping θ -> {route_name: avg_normalized_time}
        where avg_normalized_time is in [0, 1], 0=early, 1=late
    """
    aggregated = {}
    
    for theta, traces in results.items():
        route_times = {}  # route_name -> list of all attempt times
        max_attempts_seen = 1
        
        for trace in traces:
            if trace.route_attempt_times:
                max_attempts_seen = max(max_attempts_seen, trace.attempts)
                for route_name, times in trace.route_attempt_times.items():
                    if route_name not in route_times:
                        route_times[route_name] = []
                    # Normalize times by the agent's total attempts
                    if trace.attempts > 0:
                        normalized_times = [t / trace.attempts for t in times]
                        route_times[route_name].extend(normalized_times)
        
        # Compute average normalized time for each route
        avg_times = {}
        for route_name, times in route_times.items():
            if times:
                avg_times[route_name] = sum(times) / len(times)
        
        aggregated[theta] = avg_times
    
    return aggregated


def create_progression_matrix(
    route_timing: Dict[str, float],
    route_grid: Dict[str, Tuple[int, int]],
    landscape,
    n_diff_bands: int = 5,
    n_cons_bands: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create matrices for progression visualization.
    
    Returns:
        timing_matrix: Average normalized time for each cell (0=early, 1=late)
        count_matrix: Number of attempts in each cell (for alpha/visibility)
    """
    from collections import defaultdict
    
    sub_rows = 8
    sub_cols = 10
    
    timing_matrix = np.full((n_cons_bands * sub_rows, n_diff_bands * sub_cols), np.nan)
    count_matrix = np.zeros((n_cons_bands * sub_rows, n_diff_bands * sub_cols))
    
    # Count routes per cell
    cell_route_list = defaultdict(list)
    for route in landscape.routes:
        if route.name in route_grid:
            cell = route_grid[route.name]
            cell_route_list[cell].append(route.name)
    
    # Fill matrices
    for route in landscape.routes:
        if route.name not in route_grid:
            continue
        if route.name not in route_timing:
            continue
        
        diff_band, cons_band = route_grid[route.name]
        avg_time = route_timing[route.name]
        
        # Find position within cell
        cell_routes = cell_route_list[(diff_band, cons_band)]
        try:
            route_idx = cell_routes.index(route.name)
        except ValueError:
            continue
        
        sub_row = route_idx // sub_cols
        sub_col = route_idx % sub_cols
        
        sub_row = min(sub_row, sub_rows - 1)
        sub_col = min(sub_col, sub_cols - 1)
        
        row = cons_band * sub_rows + sub_row
        col = diff_band * sub_cols + sub_col
        
        timing_matrix[row, col] = avg_time
        count_matrix[row, col] = 1  # Mark as visited
    
    return timing_matrix, count_matrix


def create_progression_heatmap(
    results: Dict[float, List],
    landscape,
    output_path: str = 'output/progression_heatmap.png',
    n_diff_bands: int = 5,
    n_cons_bands: int = 5,
):
    """
    Create faceted heatmap showing WHEN agents explore different regions.
    
    Color gradient shows progression through simulation:
    - Dark purple/blue = early in career (attempt 0-25%)
    - Light blue/cyan = mid-early (25-50%)
    - Yellow/green = mid-late (50-75%)
    - Orange/red = late in career (75-100%)
    
    This reveals how agents expand their territory over time.
    
    Args:
        results: Dict mapping θ -> list of ExplorationTrace
        landscape: The Landscape object used in simulation
        output_path: Where to save the figure
        n_diff_bands: Number of difficulty bands (default 5)
        n_cons_bands: Number of consequence bands (default 5)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    route_grid = create_route_grid_mapping(landscape, n_diff_bands, n_cons_bands)
    timing_data = aggregate_route_timing(results)
    
    thetas = sorted(results.keys())
    n_thetas = len(thetas)
    
    if n_thetas == 0:
        print("No data to plot")
        return
    
    # Determine grid layout
    if n_thetas <= 2:
        nrows, ncols = 1, n_thetas
    elif n_thetas <= 4:
        nrows, ncols = 2, 2
    elif n_thetas <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = (n_thetas + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_thetas == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    fig.suptitle('Agent Progression Through Landscape by θ\n(Dark=Early, Bright=Late)', 
                 fontsize=14, fontweight='bold')
    
    theta_labels = {
        1: 'θ=1 (Very Bold)', 2: 'θ=2 (Bold)', 3: 'θ=3', 4: 'θ=4 (Moderate)',
        5: 'θ=5', 6: 'θ=6 (Cautious)', 7: 'θ=7', 8: 'θ=8 (Fearful)',
    }
    
    # Use a perceptually uniform colormap for progression
    # plasma: dark purple (early) -> yellow (late)
    cmap = plt.cm.plasma
    
    sub_rows, sub_cols = 8, 10
    
    for idx, theta in enumerate(thetas):
        ax = axes[idx]
        
        timing_matrix, count_matrix = create_progression_matrix(
            timing_data.get(theta, {}), route_grid, landscape, n_diff_bands, n_cons_bands
        )
        
        # Create a masked array for proper handling of unvisited cells
        masked_timing = np.ma.masked_where(np.isnan(timing_matrix), timing_matrix)
        
        # Plot with white background for unvisited
        ax.set_facecolor('white')
        im = ax.imshow(masked_timing, origin='lower', cmap=cmap, aspect='equal', 
                       vmin=0, vmax=1)
        
        # Draw grid lines
        for i in range(n_diff_bands + 1):
            ax.axvline(x=i * sub_cols - 0.5, color='black', linewidth=1.5, alpha=0.7)
        for j in range(n_cons_bands + 1):
            ax.axhline(y=j * sub_rows - 0.5, color='black', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Difficulty →', fontsize=10)
        ax.set_ylabel('Consequence →', fontsize=10)
        ax.set_title(theta_labels.get(int(theta), f'θ={theta}'), fontsize=11, fontweight='bold')
        
        ax.set_xticks([i * sub_cols + sub_cols/2 for i in range(n_diff_bands)])
        ax.set_xticklabels([f'D{i+1}' for i in range(n_diff_bands)])
        ax.set_yticks([i * sub_rows + sub_rows/2 for i in range(n_cons_bands)])
        ax.set_yticklabels([f'C{i+1}' for i in range(n_cons_bands)])
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Career Stage', fontsize=9)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['Early', '25%', '50%', '75%', 'Late'])
    
    # Hide unused axes
    for idx in range(len(thetas), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Progression heatmap saved to: {output_path}")


def create_journey_scatter(
    results: Dict[float, List],
    output_path: str,
    y_axis: str = 'difficulty',  # 'difficulty' or 'consequence'
):
    """
    Create scatter plot showing all agent journeys.
    
    Each dot represents one route attempt:
    - X axis: attempt number (career progression)
    - Y axis: difficulty or consequence of route attempted
    - Color: green = success, orange = failure (survived), red X = fatal
    
    Args:
        results: Dict mapping theta -> list of ExplorationTrace
        output_path: Where to save the visualization
        y_axis: 'difficulty' or 'consequence'
    """
    import matplotlib.pyplot as plt
    
    thetas = sorted(results.keys())
    n_thetas = len(thetas)
    
    # Determine grid layout
    if n_thetas <= 2:
        n_rows, n_cols = 1, n_thetas
    elif n_thetas <= 4:
        n_rows, n_cols = 2, 2
    elif n_thetas <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_thetas == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'Individual Journeys: {y_axis.capitalize()} Over Career', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    for idx, theta in enumerate(thetas):
        ax = axes[idx]
        traces = results[theta]
        
        # Collect all attempts from all agents
        success_x, success_y = [], []
        failure_x, failure_y = [], []
        fatal_x, fatal_y = [], []
        
        for trace in traces:
            if trace.outcome_over_time is None:
                continue
                
            # Get y-values based on axis choice
            if y_axis == 'difficulty':
                y_values = trace.physical_diff_over_time
            else:
                y_values = trace.consequence_over_time
            
            if y_values is None:
                continue
            
            for attempt_num, (outcome, y_val) in enumerate(zip(trace.outcome_over_time, y_values)):
                if outcome == 'success':
                    success_x.append(attempt_num)
                    success_y.append(y_val)
                elif outcome == 'failure':
                    failure_x.append(attempt_num)
                    failure_y.append(y_val)
                elif outcome == 'fatal':
                    fatal_x.append(attempt_num)
                    fatal_y.append(y_val)
        
        # Plot with transparency to show density
        if success_x:
            ax.scatter(success_x, success_y, c='#2ecc71', alpha=0.6, s=8, 
                      label=f'Success ({len(success_x)})', edgecolors='none')
        if failure_x:
            ax.scatter(failure_x, failure_y, c='#e67e22', alpha=0.3, s=8, 
                      label=f'Failure ({len(failure_x)})', edgecolors='none')
        if fatal_x:
            ax.scatter(fatal_x, fatal_y, c='#e74c3c', alpha=0.9, s=8, 
                      label=f'Fatal ({len(fatal_x)})', edgecolors='none')
        
        # Add success trendline (mean of successful attempts per bin)
        if success_x and len(success_x) > 10:
            # Bin successes by attempt number
            max_attempt = max(success_x) if success_x else 0
            n_bins = min(20, max(5, max_attempt // 5))  # 5-20 bins depending on career length
            bin_edges = np.linspace(0, max_attempt + 1, n_bins + 1)
            
            bin_centers = []
            bin_means = []
            
            for i in range(n_bins):
                bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
                # Get successes in this bin
                bin_values = [y for x, y in zip(success_x, success_y) if bin_start <= x < bin_end]
                if len(bin_values) >= 3:  # Only plot if enough data points
                    bin_centers.append((bin_start + bin_end) / 2)
                    bin_means.append(np.mean(bin_values))
            
            if len(bin_centers) >= 2:
                ax.plot(bin_centers, bin_means, 'k-', linewidth=1.5, alpha=0.6, 
                       label='Success trend', zorder=10)
                ax.plot(bin_centers, bin_means, 'k--', linewidth=1, alpha=0.4, zorder=9)
        
        ax.set_xlabel('Attempt Number', fontsize=10)
        ax.set_ylabel(y_axis.capitalize(), fontsize=10)
        ax.set_title(f'θ = {theta}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)
        
        # Count outcomes for subtitle
        total = len(success_x) + len(failure_x) + len(fatal_x)
        n_agents = len(traces)
        if total > 0:
            success_rate = len(success_x) / total * 100
            ax.text(0.98, 0.02, f'{n_agents} agents, {success_rate:.0f}% success rate', 
                   transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='gray')
    
    # Hide unused axes
    for idx in range(len(thetas), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Journey scatter saved to: {output_path}")