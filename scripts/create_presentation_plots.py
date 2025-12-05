#!/usr/bin/env python3
"""
Create presentation-ready plots for user study results.

This script generates the key visualizations for a 6-minute project presentation:
1. Bar/Box plot of Goal Alignment scores (H1 & H2)
2. Paired differences plot for Goal Alignment
3. Bar/Box plot of Perceived Safety scores (H3)
4. Post-hoc comparison table visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import existing utilities
from load_question_answers import (
    load_question_answers,
    classify_trajectory_type
)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("talk")  # Larger text for presentations
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11


def extract_alignment_data(question_answers):
    """
    Extract alignment scores for all three trajectory types.

    Returns:
        dict: {traj_type: [scores]}
    """
    alignment_data = {
        'basetraj': [],
        'modified': [],
        'modifiednoshowobstacles': []
    }

    for scenario, trajectories in question_answers.items():
        for traj_name, users in trajectories.items():
            traj_type = classify_trajectory_type(traj_name)

            if traj_type in alignment_data:
                for user_name, answers in users.items():
                    # Find alignment score
                    for question, answer in answers.items():
                        question_lower = question.lower()
                        if 'align' in question_lower or 'intended' in question_lower:
                            if isinstance(answer, (int, float)):
                                alignment_data[traj_type].append(answer)
                            break

    return alignment_data


def extract_safety_data(question_answers):
    """
    Extract safety scores for all three trajectory types.

    Returns:
        dict: {traj_type: [scores]}
    """
    safety_data = {
        'basetraj': [],
        'modified': [],
        'modifiednoshowobstacles': []
    }

    for scenario, trajectories in question_answers.items():
        for traj_name, users in trajectories.items():
            traj_type = classify_trajectory_type(traj_name)

            if traj_type in safety_data:
                for user_name, answers in users.items():
                    # Find safety score
                    for question, answer in answers.items():
                        question_lower = question.lower()
                        if 'safe' in question_lower:
                            if isinstance(answer, (int, float)):
                                safety_data[traj_type].append(answer)
                            break

    return safety_data


def extract_paired_alignment_differences(question_answers):
    """
    Extract paired differences (modified - modifiednoshowobstacles) for alignment.

    Returns:
        np.array: paired differences
    """
    differences = []

    for scenario, trajectories in question_answers.items():
        # Find trajectory names for each type
        mod_traj = None
        modnoshow_traj = None

        for traj_name in trajectories.keys():
            traj_type = classify_trajectory_type(traj_name)
            if traj_type == 'modified':
                mod_traj = traj_name
            elif traj_type == 'modifiednoshowobstacles':
                modnoshow_traj = traj_name

        if mod_traj and modnoshow_traj:
            # Find users who rated BOTH trajectories
            users_mod = set(trajectories[mod_traj].keys())
            users_modnoshow = set(trajectories[modnoshow_traj].keys())
            common_users = users_mod & users_modnoshow

            for user in common_users:
                mod_answers = trajectories[mod_traj][user]
                modnoshow_answers = trajectories[modnoshow_traj][user]

                # Extract alignment scores
                mod_score = None
                modnoshow_score = None

                for question, answer in mod_answers.items():
                    if 'align' in question.lower() or 'intended' in question.lower():
                        if isinstance(answer, (int, float)):
                            mod_score = answer
                        break

                for question, answer in modnoshow_answers.items():
                    if 'align' in question.lower() or 'intended' in question.lower():
                        if isinstance(answer, (int, float)):
                            modnoshow_score = answer
                        break

                if mod_score is not None and modnoshow_score is not None:
                    differences.append(mod_score - modnoshow_score)

    return np.array(differences)


def plot_alignment_boxplot(alignment_data, output_dir='output/plots'):
    """
    Create box plot for goal alignment scores (Slide 1, Plot 1).
    """
    # Prepare data for plotting
    conditions = ['Base\nTrajectory', 'Modified\n(Visible)', 'Modified\n(Hidden)']
    data = [
        alignment_data['basetraj'],
        alignment_data['modified'],
        alignment_data['modifiednoshowobstacles']
    ]

    # Calculate statistics for annotation
    stats = []
    for d in data:
        if d:
            median = np.median(d)
            q1 = np.percentile(d, 25)
            q3 = np.percentile(d, 75)
            stats.append({'median': median, 'q1': q1, 'q3': q3, 'n': len(d)})
        else:
            stats.append({'median': np.nan, 'q1': np.nan, 'q3': np.nan, 'n': 0})

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create box plot with same colors as safety plot
    colors = ['#0173B2', '#DE8F05', '#CC78BC']

    bp = ax.boxplot(data, labels=conditions, patch_artist=True,
                    widths=0.6,
                    medianprops=dict(color='red', linewidth=2),
                    boxprops=dict(edgecolor='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=6, alpha=0.5))

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add median and IQR annotations
    for i, (stat, pos) in enumerate(zip(stats, range(1, len(conditions) + 1))):
        if not np.isnan(stat['median']):
            # Add text annotation above each box
            ax.text(pos, 5.3, f"Mdn: {stat['median']:.1f}\nIQR: [{stat['q1']:.1f}, {stat['q3']:.1f}]\nn={stat['n']}",
                   ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Styling
    ax.set_ylabel('Goal Alignment Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trajectory Condition', fontsize=14, fontweight='bold')
    ax.set_title('Goal Alignment Ratings Across Conditions',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0.5, 5.8])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line at median for reference
    ax.axhline(y=4, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'alignment_boxplot.png', bbox_inches='tight')
    plt.savefig(output_path / 'alignment_boxplot.pdf', bbox_inches='tight')
    print(f"Saved alignment box plot to {output_path / 'alignment_boxplot.png'}")

    return fig


def plot_paired_differences(differences, output_dir='output/plots'):
    """
    Create histogram of paired differences for alignment (Slide 1, Plot 2).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram
    n, bins, patches = ax.hist(differences,
                               bins=np.arange(differences.min() - 0.5, differences.max() + 1.5, 1),
                               edgecolor='black', linewidth=1.5, alpha=0.7, color='steelblue')

    # Color bars based on sign
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center > 0:
            patch.set_facecolor('green')
            patch.set_alpha(0.6)
        elif bin_center < 0:
            patch.set_facecolor('red')
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor('gray')
            patch.set_alpha(0.4)

    # Add vertical lines for median and mean
    median_diff = np.median(differences)
    mean_diff = np.mean(differences)

    ax.axvline(median_diff, color='darkgreen', linestyle='--', linewidth=2.5,
              label=f'Median = {median_diff:.1f}')
    ax.axvline(mean_diff, color='darkblue', linestyle=':', linewidth=2.5,
              label=f'Mean = {mean_diff:.2f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5,
              label='No difference')

    # Add statistics annotation
    n_positive = np.sum(differences > 0)
    n_negative = np.sum(differences < 0)
    n_zero = np.sum(differences == 0)
    total = len(differences)

    stats_text = (f"Modified > Hidden: {n_positive}/{total} ({100*n_positive/total:.0f}%)\n"
                 f"Modified < Hidden: {n_negative}/{total} ({100*n_negative/total:.0f}%)\n"
                 f"Tied: {n_zero}/{total} ({100*n_zero/total:.0f}%)")

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Styling
    ax.set_xlabel('Alignment Score Difference\n(Modified Visible - Modified Hidden)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Paired Differences in Goal Alignment\n(Modified Visible vs. Modified Hidden)',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'alignment_paired_differences.png', bbox_inches='tight')
    plt.savefig(output_path / 'alignment_paired_differences.pdf', bbox_inches='tight')
    print(f"Saved paired differences plot to {output_path / 'alignment_paired_differences.png'}")

    return fig


def plot_safety_boxplot(safety_data, output_dir='output/plots'):
    """
    Create box plot for perceived safety scores (Slide 2, Plot 3).
    """
    # Prepare data for plotting
    conditions = ['Base\nTrajectory', 'Modified\n(Visible)', 'Modified\n(Hidden)']
    data = [
        safety_data['basetraj'],
        safety_data['modified'],
        safety_data['modifiednoshowobstacles']
    ]

    # Calculate statistics for annotation
    stats = []
    for d in data:
        if d:
            median = np.median(d)
            q1 = np.percentile(d, 25)
            q3 = np.percentile(d, 75)
            stats.append({'median': median, 'q1': q1, 'q3': q3, 'n': len(d)})
        else:
            stats.append({'median': np.nan, 'q1': np.nan, 'q3': np.nan, 'n': 0})

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create box plot with different colors to highlight modified
    # colors = ['lightblue', 'lightgreen', 'lightcoral']
    colors = ['#0173B2', '#DE8F05', '#CC78BC']

    bp = ax.boxplot(data, labels=conditions, patch_artist=True,
                    widths=0.6,
                    medianprops=dict(color='red', linewidth=2),
                    boxprops=dict(edgecolor='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=6, alpha=0.5))

    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add median and IQR annotations
    for i, (stat, pos) in enumerate(zip(stats, range(1, len(conditions) + 1))):
        if not np.isnan(stat['median']):
            # Add text annotation above each box
            ax.text(pos, 5.3, f"Mdn: {stat['median']:.1f}\nIQR: [{stat['q1']:.1f}, {stat['q3']:.1f}]\nn={stat['n']}",
                   ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Styling
    ax.set_ylabel('Perceived Safety Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trajectory Condition', fontsize=14, fontweight='bold')
    ax.set_title('Perceived Safety Ratings Across Conditions',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0.5, 5.8])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line at median for reference
    ax.axhline(y=4, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'safety_boxplot.png', bbox_inches='tight')
    plt.savefig(output_path / 'safety_boxplot.pdf', bbox_inches='tight')
    print(f"Saved safety box plot to {output_path / 'safety_boxplot.png'}")

    return fig


def plot_posthoc_table(output_dir='output/plots'):
    """
    Create visualization of post-hoc comparison table (Slide 2, Plot 4).
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    # Define table data
    comparisons = [
        ['Modified (Visible) vs.\nModified (Hidden)', '0.002', '✓ Significant'],
        ['Modified (Visible) vs.\nBase Trajectory', '0.354', '✗ Not Significant'],
        ['Base Trajectory vs.\nModified (Hidden)', '0.038', '✗ Not Significant']
    ]

    headers = ['Comparison\n(Safety)', f'Adjusted p-value\n(α_crit = 0.0056)', 'Result']

    # Create table
    table_data = [headers] + comparisons

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.4, 0.3, 0.3])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Color header row
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=13)

    # Color result cells based on significance
    for i, row in enumerate(comparisons, start=1):
        # Alternate row colors
        row_color = '#F2F2F2' if i % 2 == 0 else 'white'
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)

        # Highlight significant result
        result_cell = table[(i, 2)]
        if '✓' in row[2]:
            result_cell.set_facecolor('#C6EFCE')
            result_cell.set_text_props(weight='bold', color='#006100')
        else:
            result_cell.set_facecolor('#FFC7CE')
            result_cell.set_text_props(color='#9C0006')

        # Highlight significant p-value
        pval_cell = table[(i, 1)]
        if row[1] == '0.002':
            pval_cell.set_text_props(weight='bold')

    # Add title
    ax.set_title('Post-Hoc Pairwise Comparisons for Perceived Safety\n(Wilcoxon Signed-Rank Tests with Bonferroni Correction)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'posthoc_table.png', bbox_inches='tight')
    plt.savefig(output_path / 'posthoc_table.pdf', bbox_inches='tight')
    print(f"Saved post-hoc table to {output_path / 'posthoc_table.png'}")

    return fig


def print_summary_statistics(alignment_data, safety_data, differences):
    """
    Print summary statistics for all plots.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR PLOTS")
    print("="*80)

    print("\n1. GOAL ALIGNMENT SCORES:")
    print("-" * 80)
    for name, label in [('basetraj', 'Base Trajectory'),
                        ('modified', 'Modified (Visible)'),
                        ('modifiednoshowobstacles', 'Modified (Hidden)')]:
        data = alignment_data[name]
        if data:
            print(f"\n  {label}:")
            print(f"    Median: {np.median(data):.1f}")
            print(f"    IQR: [{np.percentile(data, 25):.1f}, {np.percentile(data, 75):.1f}]")
            print(f"    Mean ± SD: {np.mean(data):.2f} ± {np.std(data, ddof=1):.2f}")
            print(f"    Range: [{np.min(data):.0f}, {np.max(data):.0f}]")
            print(f"    n = {len(data)}")

    print("\n\n2. PAIRED DIFFERENCES (Modified Visible - Modified Hidden):")
    print("-" * 80)
    print(f"  Median: {np.median(differences):.1f}")
    print(f"  Mean ± SD: {np.mean(differences):.2f} ± {np.std(differences, ddof=1):.2f}")
    print(f"  Range: [{np.min(differences):.0f}, {np.max(differences):.0f}]")
    print(f"  Positive differences: {np.sum(differences > 0)}/{len(differences)} ({100*np.sum(differences > 0)/len(differences):.0f}%)")
    print(f"  Negative differences: {np.sum(differences < 0)}/{len(differences)} ({100*np.sum(differences < 0)/len(differences):.0f}%)")
    print(f"  Ties: {np.sum(differences == 0)}/{len(differences)} ({100*np.sum(differences == 0)/len(differences):.0f}%)")

    print("\n\n3. PERCEIVED SAFETY SCORES:")
    print("-" * 80)
    for name, label in [('basetraj', 'Base Trajectory'),
                        ('modified', 'Modified (Visible)'),
                        ('modifiednoshowobstacles', 'Modified (Hidden)')]:
        data = safety_data[name]
        if data:
            print(f"\n  {label}:")
            print(f"    Median: {np.median(data):.1f}")
            print(f"    IQR: [{np.percentile(data, 25):.1f}, {np.percentile(data, 75):.1f}]")
            print(f"    Mean ± SD: {np.mean(data):.2f} ± {np.std(data, ddof=1):.2f}")
            print(f"    Range: [{np.min(data):.0f}, {np.max(data):.0f}]")
            print(f"    n = {len(data)}")

    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    print("="*80)
    print("CREATING PRESENTATION PLOTS")
    print("="*80)

    # Exclude specific users (case-insensitive)
    excluded_users = ["test"]

    # Load data
    zarr_path = "output/obstacles_on_path_10_userstudy.zarr"
    print(f"\nLoading data from: {zarr_path}")
    if excluded_users:
        print(f"Excluding users containing: {', '.join(excluded_users)}")
    question_answers = load_question_answers(zarr_path, excluded_users=excluded_users)
    print(f"Data loaded successfully.\n")

    # Extract data
    print("Extracting alignment and safety scores...")
    alignment_data = extract_alignment_data(question_answers)
    safety_data = extract_safety_data(question_answers)
    differences = extract_paired_alignment_differences(question_answers)
    print("Data extraction complete.\n")

    # Print summary statistics
    print_summary_statistics(alignment_data, safety_data, differences)

    # Create plots
    print("Creating plots...")
    print("-" * 80)

    plot_alignment_boxplot(alignment_data)
    plot_paired_differences(differences)
    plot_safety_boxplot(safety_data)
    plot_posthoc_table()

    print("\n" + "="*80)
    print("ALL PLOTS CREATED SUCCESSFULLY")
    print("="*80)
    print("\nPlots saved to: output/plots/")
    print("  - alignment_boxplot.png (and .pdf)")
    print("  - alignment_paired_differences.png (and .pdf)")
    print("  - safety_boxplot.png (and .pdf)")
    print("  - posthoc_table.png (and .pdf)")
    print("\nReady for presentation!")


if __name__ == "__main__":
    main()
