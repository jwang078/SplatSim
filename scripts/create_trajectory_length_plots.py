#!/usr/bin/env python3
"""
Script to create plots relating trajectory lengths to user responses.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import zarr
import re
from load_question_answers import (
    load_question_answers,
    classify_trajectory_type
)

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

# Timestep durations
GELLO_DT = 0.5  # seconds per timestep for gello trajectories
TRAJ_DT = 0.1   # seconds per timestep for base/modified trajectories


def identify_question_type(question):
    """Identify question type based on keywords."""
    question_lower = question.lower()
    if 'safe' in question_lower:
        return 'safety'
    elif 'align' in question_lower or 'intended' in question_lower:
        return 'alignment'
    elif 'deviat' in question_lower and 'reason' in question_lower:
        return 'exclude'
    else:
        return 'other'


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
    Plot response (safety or alignment) vs gello completion time.
    Point size indicates density (number of overlapping responses).

    Args:
        data_list: List of (traj_type, response_score, gello_time_sec) tuples
        response_type: 'safety' or 'alignment'
        output_path: Path to save the figure
    """
    from collections import Counter

    # Colorblind-friendly palette
    color_map = {
        'basetraj': '#0173B2',
        'modifiednoshowobstacles': '#DE8F05',
        'modified': '#CC78BC'
    }

    display_names = {
        'basetraj': 'Base Trajectory',
        'modifiednoshowobstacles': 'Modified (No Obstacles Shown)',
        'modified': 'Modified Trajectory'
    }

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
                      color=color_map[traj_type],
                      label=display_names[traj_type],
                      s=sizes, alpha=0.7)

    # Labels and title
    response_label = 'Perceived Safety' if response_type == 'safety' else 'Goal Alignment'
    ax.set_xlabel('User Completion Time (seconds)', fontweight='bold')
    ax.set_ylabel(f'{response_label} (1-5 scale)', fontweight='bold')
    ax.set_title(f'{response_label} vs. User Completion Time\n(Point size indicates response density)',
                fontweight='bold')
    ax.set_ylim([0.5, 5.5])
    ax.grid(True, alpha=0.3)
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

    # Colorblind-friendly palette
    color_map = {
        'basetraj': '#0173B2',
        'modifiednoshowobstacles': '#DE8F05',
        'modified': '#CC78BC'
    }

    display_names = {
        'basetraj': 'Base Trajectory',
        'modifiednoshowobstacles': 'Modified (No Obstacles Shown)',
        'modified': 'Modified Trajectory'
    }

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
                      color=color_map[traj_type],
                      label=display_names[traj_type],
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


if __name__ == "__main__":
    import os

    # Load data
    zarr_path = "output/obstacles_on_path_10_userstudy.zarr"
    print(f"Loading data from: {zarr_path}\n")
    question_answers = load_question_answers(zarr_path)

    # Extract trajectory data with lengths
    print("Extracting trajectory lengths and responses...")
    traj_data = extract_trajectory_data_with_lengths(zarr_path, question_answers)

    # Create output directory
    output_dir = "output/user_study_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating trajectory length plots...\n")

    # Plot 1: Safety vs Gello completion time
    plot_response_vs_gello_time(
        traj_data['safety_vs_gello_time'],
        'safety',
        os.path.join(output_dir, "safety_vs_gello_completion_time.png")
    )

    # Plot 2: Alignment vs Gello completion time
    plot_response_vs_gello_time(
        traj_data['alignment_vs_gello_time'],
        'alignment',
        os.path.join(output_dir, "alignment_vs_gello_completion_time.png")
    )

    # Plot 3: Safety vs trajectory execution time
    plot_response_vs_traj_length(
        traj_data['safety_vs_traj_length'],
        'safety',
        os.path.join(output_dir, "safety_vs_trajectory_execution_time.png")
    )

    # Plot 4: Alignment vs trajectory execution time
    plot_response_vs_traj_length(
        traj_data['alignment_vs_traj_length'],
        'alignment',
        os.path.join(output_dir, "alignment_vs_trajectory_execution_time.png")
    )

    # Print statistics
    print_trajectory_length_statistics(traj_data)

    print(f"\n{'='*80}")
    print(f"All trajectory length plots saved to: {output_dir}/")
    print(f"{'='*80}\n")
