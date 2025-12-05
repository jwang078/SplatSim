#!/usr/bin/env python3
"""
Script to create publication-quality plots for user study data.

Generates 7 plots:
1. Perceived safety by trajectory type (bar chart)
2. Goal alignment by trajectory type (bar chart)
3. Safety vs alignment scatter (density-based)
4. Safety vs user completion time (scatter)
5. Alignment vs user completion time (scatter)
6. Safety vs trajectory execution time (scatter)
7. Alignment vs trajectory execution time (scatter)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import zarr
import re
from load_question_answers import (
    load_question_answers,
    get_answers_by_trajectory_type,
    classify_trajectory_type
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Timestep durations
GELLO_DT = 0.5  # seconds per timestep for gello trajectories
TRAJ_DT = 0.1   # seconds per timestep for base/modified trajectories

# Colorblind-friendly palette (Paul Tol's colors)
COLOR_MAP = {
    'basetraj': '#0173B2',
    'modifiednoshowobstacles': '#DE8F05',
    'modified': '#CC78BC'
}

DISPLAY_NAMES = {
    'basetraj': 'Base Trajectory',
    'modified': 'Modified Trajectory',
    'modifiednoshowobstacles': 'Modified Trajectory\n(No Obstacles Shown)'
}

# Set publication-quality plot style
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")


# ============================================================================
# DATA EXTRACTION UTILITIES
# ============================================================================

def identify_question_type(question):
    """
    Identify question type based on keywords.

    Args:
        question: Question string

    Returns:
        str: 'safety', 'alignment', or 'other'
    """
    question_lower = question.lower()
    if 'safe' in question_lower:
        return 'safety'
    elif 'align' in question_lower or 'intended' in question_lower:
        return 'alignment'
    elif 'deviat' in question_lower and 'reason' in question_lower:
        return 'exclude'
    else:
        return 'other'


def extract_data_by_question_type(question_answers_dict):
    """
    Extract data grouped by question type (safety, alignment).

    Args:
        question_answers_dict: Dictionary returned by load_question_answers()

    Returns:
        dict: {question_type: {traj_type: [values]}}
    """
    data_by_question_type = defaultdict(lambda: defaultdict(list))

    for scenario_name, trajectories in question_answers_dict.items():
        for traj_name, users in trajectories.items():
            traj_type = classify_trajectory_type(traj_name)

            # Skip gello trajectories
            if traj_type == 'gello_traj':
                continue

            for user_name, answers in users.items():
                for question, answer in answers.items():
                    question_type = identify_question_type(question)

                    # Skip excluded questions
                    if question_type == 'exclude':
                        continue

                    if isinstance(answer, (int, float)):
                        data_by_question_type[question_type][traj_type].append(answer)

    return dict(data_by_question_type)


def extract_paired_data(question_answers_dict):
    """
    Extract paired (safety, alignment) data for scatter plot.

    Args:
        question_answers_dict: Dictionary returned by load_question_answers()

    Returns:
        tuple: (alignment_values, safety_values, traj_types)
    """
    # Store responses by (scenario, traj, user) to ensure proper pairing
    responses = defaultdict(lambda: {})

    for scenario_name, trajectories in question_answers_dict.items():
        for traj_name, users in trajectories.items():
            traj_type = classify_trajectory_type(traj_name)

            # Skip gello trajectories
            if traj_type == 'gello_traj':
                continue

            for user_name, answers in users.items():
                key = (scenario_name, traj_name, user_name)

                for question, answer in answers.items():
                    question_type = identify_question_type(question)

                    if question_type == 'exclude':
                        continue

                    if isinstance(answer, (int, float)):
                        responses[key][question_type] = answer
                        responses[key]['traj_type'] = traj_type

    # Extract paired values
    alignment_values = []
    safety_values = []
    traj_types = []

    for key, data in responses.items():
        if 'alignment' in data and 'safety' in data:
            alignment_values.append(data['alignment'])
            safety_values.append(data['safety'])
            traj_types.append(data['traj_type'])

    return alignment_values, safety_values, traj_types


def get_trajectory_type_display_name(traj_type):
    """Get display name for trajectory type."""
    return DISPLAY_NAMES.get(traj_type, traj_type)


# ============================================================================
# PLOTTING FUNCTIONS - BAR CHARTS
# ============================================================================

def plot_perceived_safety_by_trajectory_type(data_by_question_type, output_path):
    """
    Create bar chart of perceived safety across trajectory types.

    Args:
        data_by_question_type: Dictionary from extract_data_by_question_type()
        output_path: Path to save the figure
    """
    safety_data = data_by_question_type.get('safety', {})

    # Define order of trajectory types
    traj_order = ['basetraj', 'modifiednoshowobstacles', 'modified']

    # Prepare data
    means = []
    stds = []
    labels = []

    for traj_type in traj_order:
        if traj_type in safety_data and len(safety_data[traj_type]) > 0:
            values = safety_data[traj_type]
            means.append(np.mean(values))
            stds.append(np.std(values) / np.sqrt(len(values)))  # Standard error
            labels.append(get_trajectory_type_display_name(traj_type))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(labels))
    # Colorblind-friendly palette (blue, orange, purple)
    colors = ['#0173B2', '#DE8F05', '#CC78BC']

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=colors[:len(labels)], alpha=0.8,
                  edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_ylabel('Perceived Safety (1-5 scale)', fontweight='bold')
    ax.set_xlabel('Trajectory Type', fontweight='bold')
    ax.set_title('Perceived Safety Across Trajectory Types', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 5.5])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_alignment_by_trajectory_type(data_by_question_type, output_path):
    """
    Create bar chart of alignment across trajectory types.

    Args:
        data_by_question_type: Dictionary from extract_data_by_question_type()
        output_path: Path to save the figure
    """
    alignment_data = data_by_question_type.get('alignment', {})

    # Define order of trajectory types
    traj_order = ['basetraj', 'modifiednoshowobstacles', 'modified']

    # Prepare data
    means = []
    stds = []
    labels = []

    for traj_type in traj_order:
        if traj_type in alignment_data and len(alignment_data[traj_type]) > 0:
            values = alignment_data[traj_type]
            means.append(np.mean(values))
            stds.append(np.std(values) / np.sqrt(len(values)))  # Standard error
            labels.append(get_trajectory_type_display_name(traj_type))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(labels))
    # Colorblind-friendly palette (blue, orange, purple)
    colors = ['#0173B2', '#DE8F05', '#CC78BC']

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                  color=colors[:len(labels)], alpha=0.8,
                  edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_ylabel('Goal Alignment (1-5 scale)', fontweight='bold')
    ax.set_xlabel('Trajectory Type', fontweight='bold')
    ax.set_title('Goal Alignment Across Trajectory Types', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 5.5])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# PLOTTING FUNCTIONS - SCATTER PLOTS
# ============================================================================

def plot_safety_vs_alignment_scatter(alignment_values, safety_values, traj_types, output_path):
    """
    Create scatter plot of perceived safety vs goal alignment.
    Points are colored and sized based on density (number of overlapping responses).

    Args:
        alignment_values: List of alignment scores
        safety_values: List of safety scores
        traj_types: List of trajectory types for each data point
        output_path: Path to save the figure
    """
    from collections import Counter

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Count occurrences of each (alignment, safety) coordinate
    point_counts = Counter(zip(alignment_values, safety_values))

    # Extract unique points with their counts
    unique_points = list(point_counts.keys())
    counts = np.array([point_counts[p] for p in unique_points])

    # Separate x and y coordinates
    unique_alignment = np.array([p[0] for p in unique_points])
    unique_safety = np.array([p[1] for p in unique_points])

    # Create colormap based on counts
    # Use a perceptually uniform colormap
    norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
    cmap = plt.cm.viridis

    # Size points based on count (larger = more responses)
    sizes = 100 + 200 * (counts - counts.min()) / (counts.max() - counts.min() + 1e-6)

    # Color points based on count
    colors = cmap(norm(counts))

    # Plot points
    scatter = ax.scatter(unique_alignment, unique_safety,
                        c=counts, cmap='viridis',
                        s=sizes, alpha=0.7,
                        vmin=counts.min(), vmax=counts.max())

    # Add colorbar to show count mapping
    cbar = plt.colorbar(scatter, ax=ax, label='Number of Responses')
    cbar.set_label('Number of Responses', fontweight='bold')

    # Add regression line for all data
    if len(alignment_values) > 0:
        z = np.polyfit(alignment_values, safety_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(alignment_values), max(alignment_values), 100)
        correlation = np.corrcoef(alignment_values, safety_values)[0, 1]
        ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2.5,
               label=f'Linear fit (r={correlation:.2f})', zorder=5)

    # Customize plot
    ax.set_xlabel('Goal Alignment (1-5 scale)', fontweight='bold')
    ax.set_ylabel('Perceived Safety (1-5 scale)', fontweight='bold')
    ax.set_title('Perceived Safety vs. Goal Alignment\n(Size and color indicate response density)',
                fontweight='bold')
    ax.set_xlim([0.5, 5.5])
    ax.set_ylim([0.5, 5.5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)

    # Add diagonal reference line (perfect correlation)
    ax.plot([0.5, 5.5], [0.5, 5.5], 'gray', linestyle=':', alpha=0.3,
           linewidth=1.5, label='y=x', zorder=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_statistics(data_by_question_type, alignment_values, safety_values):
    """Print summary statistics for the data."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Statistics by trajectory type
    for question_type in ['safety', 'alignment']:
        if question_type in data_by_question_type:
            print(f"\n{question_type.upper()}:")
            data = data_by_question_type[question_type]

            for traj_type in ['basetraj', 'modifiednoshowobstacles', 'modified']:
                if traj_type in data and len(data[traj_type]) > 0:
                    values = data[traj_type]
                    print(f"  {get_trajectory_type_display_name(traj_type)}:")
                    print(f"    Mean: {np.mean(values):.2f} ± {np.std(values):.2f}")
                    print(f"    Range: [{np.min(values)}, {np.max(values)}]")
                    print(f"    N: {len(values)}")

    # Correlation between safety and alignment
    if len(alignment_values) > 0 and len(safety_values) > 0:
        correlation = np.corrcoef(alignment_values, safety_values)[0, 1]
        print(f"\nCorrelation between Goal Alignment and Perceived Safety: {correlation:.3f}")
        print(f"Total paired responses: {len(alignment_values)}")


# ============================================================================
# TRAJECTORY LENGTH ANALYSIS
# ============================================================================

def extract_trajectory_data_with_lengths(zarr_path, question_answers_dict):
    """
    Extract trajectory lengths paired with question responses.

    Returns:
        dict with keys:
            'safety_vs_gello_time': list of (traj_type, safety_score, gello_time_sec)
            'alignment_vs_gello_time': list of (traj_type, alignment_score, gello_time_sec)
            'safety_vs_traj_length': list of (traj_type, safety_score, traj_time_sec)
            'alignment_vs_traj_length': list of (traj_type, alignment_score, traj_time_sec)
    """
    root = zarr.open(zarr_path, mode='r')
    scenarios_group = root['trajectories']

    data = {
        'safety_vs_gello_time': [],
        'alignment_vs_gello_time': [],
        'safety_vs_traj_length': [],
        'alignment_vs_traj_length': []
    }

    scenario_re = re.compile(r"^scenario_(\d+)$")

    for scenario_name in sorted(scenarios_group.keys()):
        if not scenario_re.match(scenario_name):
            continue

        scenario_group = scenarios_group[scenario_name]

        # Get gello trajectory lengths for this scenario (from obstacle_config_00/traj_00)
        gello_times_by_user = {}
        if 'obstacle_config_00' in scenario_group:
            obstacle_00 = scenario_group['obstacle_config_00']
            if 'traj_00' in obstacle_00:
                traj_00 = obstacle_00['traj_00']
                # Find all qs_{username} datasets (gello trajectories)
                for key in traj_00.keys():
                    if key.startswith('qs_'):
                        user_name = key[3:]  # Remove 'qs_' prefix
                        gello_qs = np.array(traj_00[key])
                        gello_time_sec = len(gello_qs) * GELLO_DT
                        gello_times_by_user[user_name] = gello_time_sec

        # Process trajectory responses from question_answers_dict
        if scenario_name in question_answers_dict:
            for traj_name, users in question_answers_dict[scenario_name].items():
                traj_type = classify_trajectory_type(traj_name)

                # Skip gello trajectories
                if traj_type == 'gello_traj':
                    continue

                # Parse the trajectory name to find the actual trajectory in zarr
                # Format: scenario_XXXX_obstacle_config_YY_traj_ZZ_suffix
                obstacle_match = re.search(r'obstacle_config_(\d+)', traj_name)
                traj_match = re.search(r'traj_(\d+)', traj_name)

                if obstacle_match and traj_match:
                    obstacle_name = f"obstacle_config_{obstacle_match.group(1)}"
                    traj_id = f"traj_{traj_match.group(1)}"

                    # Get trajectory length from zarr
                    traj_time_sec = None
                    if obstacle_name in scenario_group:
                        obstacle_group = scenario_group[obstacle_name]
                        if traj_id in obstacle_group:
                            traj_group = obstacle_group[traj_id]
                            if 'qs' in traj_group:
                                qs = np.array(traj_group['qs'])
                                traj_time_sec = len(qs) * TRAJ_DT

                    # Get responses for each user
                    for user_name, answers in users.items():
                        safety_score = None
                        alignment_score = None

                        for question, answer in answers.items():
                            q_type = identify_question_type(question)
                            if q_type == 'safety' and isinstance(answer, (int, float)):
                                safety_score = answer
                            elif q_type == 'alignment' and isinstance(answer, (int, float)):
                                alignment_score = answer

                        # Get gello time for this user
                        gello_time = gello_times_by_user.get(user_name, None)

                        # Add to datasets
                        if safety_score is not None and gello_time is not None:
                            data['safety_vs_gello_time'].append((traj_type, safety_score, gello_time))
                        if alignment_score is not None and gello_time is not None:
                            data['alignment_vs_gello_time'].append((traj_type, alignment_score, gello_time))
                        if safety_score is not None and traj_time_sec is not None:
                            data['safety_vs_traj_length'].append((traj_type, safety_score, traj_time_sec))
                        if alignment_score is not None and traj_time_sec is not None:
                            data['alignment_vs_traj_length'].append((traj_type, alignment_score, traj_time_sec))

    return data


def plot_response_vs_gello_time(data_list, response_type, output_path):
    """
    Plot average response (safety or alignment) vs gello completion time.
    Each point represents a user's average score across all trajectory types
    they evaluated, plotted against their gello completion time.

    Args:
        data_list: List of (traj_type, response_score, gello_time_sec) tuples
        response_type: 'safety' or 'alignment'
        output_path: Path to save the figure
    """
    from collections import defaultdict

    fig, ax = plt.subplots(figsize=(10, 8))

    # Group responses by (user's gello_time) and compute average score
    # We assume same gello_time = same user
    responses_by_time = defaultdict(list)
    for _, score, time in data_list:
        responses_by_time[time].append(score)

    # Calculate average score for each user (gello completion time)
    gello_times = []
    avg_scores = []
    for time, scores in responses_by_time.items():
        gello_times.append(time)
        avg_scores.append(np.mean(scores))

    if gello_times:
        # Convert to numpy arrays
        gello_times = np.array(gello_times)
        avg_scores = np.array(avg_scores)

        # Plot points
        ax.scatter(gello_times, avg_scores,
                  color='#0173B2',  # Blue color
                  s=200, alpha=0.7)

        # Add regression line
        if len(gello_times) > 1:
            z = np.polyfit(gello_times, avg_scores, 1)
            p = np.poly1d(z)
            x_line = np.linspace(gello_times.min(), gello_times.max(), 100)
            correlation = np.corrcoef(gello_times, avg_scores)[0, 1]
            ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2.5,
                   label=f'Linear fit (r={correlation:.2f})', zorder=5)

    # Labels and title
    response_label = 'Perceived Safety' if response_type == 'safety' else 'Goal Alignment'
    ax.set_xlabel('User Completion Time (seconds)', fontweight='bold')
    ax.set_ylabel(f'Average {response_label} (1-5 scale)', fontweight='bold')
    ax.set_title(f'Average {response_label} vs. User Completion Time',
                fontweight='bold')
    ax.set_ylim([0.5, 5.5])
    ax.grid(True, alpha=0.3)
    if len(gello_times) > 1:
        ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_response_vs_traj_length(data_list, response_type, output_path):
    """
    Plot response (safety or alignment) vs trajectory execution time.
    Point size indicates density (number of overlapping responses).

    Args:
        data_list: List of (traj_type, response_score, traj_time_sec) tuples
        response_type: 'safety' or 'alignment'
        output_path: Path to save the figure
    """
    from collections import Counter

    fig, ax = plt.subplots(figsize=(10, 8))

    # Group data by trajectory type and count overlaps
    for traj_type in ['basetraj', 'modifiednoshowobstacles', 'modified']:
        traj_data = [(score, time) for tt, score, time in data_list if tt == traj_type]

        if traj_data:
            # Count occurrences of each (time, score) coordinate
            point_counts = Counter(traj_data)

            # Extract unique points with their counts
            unique_points = list(point_counts.keys())
            counts = np.array([point_counts[p] for p in unique_points])

            # Separate x and y coordinates
            scores = np.array([p[0] for p in unique_points])
            times = np.array([p[1] for p in unique_points])

            # Size points based on count (larger = more responses)
            # Scale from 100 to 400 based on density
            if len(counts) > 1 and counts.max() > counts.min():
                sizes = 100 + 300 * (counts - counts.min()) / (counts.max() - counts.min())
            else:
                sizes = np.full(len(counts), 200)

            ax.scatter(times, scores,
                      color=COLOR_MAP[traj_type],
                      label=get_trajectory_type_display_name(traj_type).replace('\n', ' '),
                      s=sizes, alpha=0.7)

    # Labels and title
    response_label = 'Perceived Safety' if response_type == 'safety' else 'Goal Alignment'
    ax.set_xlabel('Trajectory Execution Time (seconds)', fontweight='bold')
    ax.set_ylabel(f'{response_label} (1-5 scale)', fontweight='bold')
    ax.set_title(f'{response_label} vs. Trajectory Execution Time\n(Point size indicates response density)',
                fontweight='bold')
    ax.set_ylim([0.5, 5.5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ============================================================================
# STATISTICS REPORTING
# ============================================================================

def print_trajectory_length_statistics(data):
    """Print statistics about trajectory lengths."""
    print("\n" + "="*80)
    print("TRAJECTORY LENGTH STATISTICS")
    print("="*80)

    # Gello times
    print("\nUser Completion Times (Gello):")
    if data['safety_vs_gello_time']:
        gello_times = [time for _, _, time in data['safety_vs_gello_time']]
        print(f"  Mean: {np.mean(gello_times):.2f} seconds")
        print(f"  Std: {np.std(gello_times):.2f} seconds")
        print(f"  Range: [{np.min(gello_times):.2f}, {np.max(gello_times):.2f}] seconds")
        print(f"  N: {len(gello_times)}")

    # Trajectory execution times by type
    print("\nTrajectory Execution Times:")
    for traj_type in ['basetraj', 'modifiednoshowobstacles', 'modified']:
        traj_times = [time for tt, _, time in data['safety_vs_traj_length'] if tt == traj_type]
        if traj_times:
            # Remove duplicates (same trajectory shown to multiple users)
            unique_times = list(set(traj_times))
            print(f"\n  {traj_type}:")
            print(f"    Mean: {np.mean(unique_times):.2f} seconds")
            print(f"    Std: {np.std(unique_times):.2f} seconds")
            print(f"    Range: [{np.min(unique_times):.2f}, {np.max(unique_times):.2f}] seconds")
            print(f"    N (unique): {len(unique_times)}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import os

    # Configuration
    zarr_path = "output/obstacles_on_path_10_userstudy.zarr"
    output_dir = "output/user_study_plots"

    print("="*80)
    print("USER STUDY DATA ANALYSIS AND VISUALIZATION")
    print("="*80)
    print(f"\nInput: {zarr_path}")
    print(f"Output: {output_dir}/\n")

    # Load data
    print("Loading question answers from zarr file...")
    question_answers = load_question_answers(zarr_path)

    # Extract data for basic plots
    print("Extracting response data by question type...")
    data_by_question_type = extract_data_by_question_type(question_answers)
    alignment_values, safety_values, traj_types = extract_paired_data(question_answers)

    # Extract trajectory length data
    print("Extracting trajectory length data...")
    traj_data = extract_trajectory_data_with_lengths(zarr_path, question_answers)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    print("\nGenerating plots...")
    print("-" * 80)

    # Bar charts
    plot_perceived_safety_by_trajectory_type(
        data_by_question_type,
        os.path.join(output_dir, "perceived_safety_by_trajectory_type.png")
    )

    plot_alignment_by_trajectory_type(
        data_by_question_type,
        os.path.join(output_dir, "alignment_by_trajectory_type.png")
    )

    # Safety vs alignment scatter
    plot_safety_vs_alignment_scatter(
        alignment_values, safety_values, traj_types,
        os.path.join(output_dir, "safety_vs_alignment_scatter.png")
    )

    # Trajectory length plots - user completion time
    plot_response_vs_gello_time(
        traj_data['safety_vs_gello_time'],
        'safety',
        os.path.join(output_dir, "safety_vs_gello_completion_time.png")
    )

    plot_response_vs_gello_time(
        traj_data['alignment_vs_gello_time'],
        'alignment',
        os.path.join(output_dir, "alignment_vs_gello_completion_time.png")
    )

    # Trajectory length plots - execution time
    plot_response_vs_traj_length(
        traj_data['safety_vs_traj_length'],
        'safety',
        os.path.join(output_dir, "safety_vs_trajectory_execution_time.png")
    )

    plot_response_vs_traj_length(
        traj_data['alignment_vs_traj_length'],
        'alignment',
        os.path.join(output_dir, "alignment_vs_trajectory_execution_time.png")
    )

    # Print statistics
    print_summary_statistics(data_by_question_type, alignment_values, safety_values)
    print_trajectory_length_statistics(traj_data)

    # Summary
    print(f"\n{'='*80}")
    print("COMPLETE")
    print("="*80)
    print(f"All 7 plots saved to: {output_dir}/")
    print("="*80 + "\n")