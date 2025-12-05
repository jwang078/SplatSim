#!/usr/bin/env python3
"""
Statistical Analysis of User Study Hypotheses

This script performs rigorous statistical analysis on user study data to test
three hypotheses about goal alignment and perceived safety.

Hypotheses:
H1: Trajectories are more goal-aligned when the robot takes a shorter path
H2: Trajectory deviations are more goal-aligned when obstacles are visible
H3: Trajectories are perceived as safer when users know why robot deviated
"""

import numpy as np
import zarr
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, shapiro, probplot, t
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Import existing utilities
from load_question_answers import (
    load_question_answers,
    classify_trajectory_type
)

# ============================================================================
# Configuration
# ============================================================================

ALPHA = 0.05
N_HYPOTHESES = 3
ALPHA_ADJUSTED = ALPHA / N_HYPOTHESES  # 0.0167
N_BOOTSTRAP = 10000

# ============================================================================
# Section 2: Data Extraction Functions
# ============================================================================

def identify_question_type(question):
    """
    Classify question as 'safety', 'alignment', or 'other'.

    Args:
        question: Question text string

    Returns:
        str: 'safety', 'alignment', or 'other'
    """
    question_lower = question.lower()
    if 'safe' in question_lower:
        return 'safety'
    elif 'align' in question_lower or 'intended' in question_lower:
        return 'alignment'
    else:
        return 'other'


def extract_score(user_answers, question_type):
    """
    Extract numerical score for specified question type.

    Args:
        user_answers: Dictionary of {question: answer}
        question_type: 'safety' or 'alignment'

    Returns:
        int/float or None: Score if found, None otherwise
    """
    for question, answer in user_answers.items():
        if identify_question_type(question) == question_type:
            if isinstance(answer, (int, float)):
                return answer
    return None


def extract_paired_data_for_h1(question_answers):
    """
    Extract paired (basetraj, modifiednoshowobstacles) alignment scores.

    Args:
        question_answers: Nested dict from load_question_answers()

    Returns:
        tuple: (base_values, modnoshow_values, metadata)
            base_values: np.array of basetraj alignment scores
            modnoshow_values: np.array of modifiednoshowobstacles alignment scores
            metadata: list of (user, scenario) tuples
    """
    base_values = []
    modnoshow_values = []
    metadata = []

    for scenario, trajectories in question_answers.items():
        # Find trajectory names for each type
        basetraj_traj = None
        modnoshow_traj = None

        for traj_name in trajectories.keys():
            traj_type = classify_trajectory_type(traj_name)
            if traj_type == 'basetraj':
                basetraj_traj = traj_name
            elif traj_type == 'modifiednoshowobstacles':
                modnoshow_traj = traj_name

        if basetraj_traj and modnoshow_traj:
            # Find users who rated BOTH trajectories
            users_base = set(trajectories[basetraj_traj].keys())
            users_modnoshow = set(trajectories[modnoshow_traj].keys())
            common_users = users_base & users_modnoshow

            for user in common_users:
                # Extract alignment scores
                base_answers = trajectories[basetraj_traj][user]
                modnoshow_answers = trajectories[modnoshow_traj][user]

                base_score = extract_score(base_answers, 'alignment')
                modnoshow_score = extract_score(modnoshow_answers, 'alignment')

                if base_score is not None and modnoshow_score is not None:
                    base_values.append(base_score)
                    modnoshow_values.append(modnoshow_score)
                    metadata.append((user, scenario))

    return np.array(base_values), np.array(modnoshow_values), metadata


def extract_paired_data_for_h2(question_answers):
    """
    Extract paired (modified, modifiednoshowobstacles) alignment scores.

    Args:
        question_answers: Nested dict from load_question_answers()

    Returns:
        tuple: (mod_values, modnoshow_values, metadata)
    """
    mod_values = []
    modnoshow_values = []
    metadata = []

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
                # Extract alignment scores
                mod_answers = trajectories[mod_traj][user]
                modnoshow_answers = trajectories[modnoshow_traj][user]

                mod_score = extract_score(mod_answers, 'alignment')
                modnoshow_score = extract_score(modnoshow_answers, 'alignment')

                if mod_score is not None and modnoshow_score is not None:
                    mod_values.append(mod_score)
                    modnoshow_values.append(modnoshow_score)
                    metadata.append((user, scenario))

    return np.array(mod_values), np.array(modnoshow_values), metadata


def extract_triplet_data_for_h3(question_answers):
    """
    Extract triplet (basetraj, modifiednoshowobstacles, modified) safety scores.

    Args:
        question_answers: Nested dict from load_question_answers()

    Returns:
        tuple: (base_values, modnoshow_values, mod_values, metadata)
    """
    base_values = []
    modnoshow_values = []
    mod_values = []
    metadata = []

    for scenario, trajectories in question_answers.items():
        # Find trajectory names for each type
        basetraj_traj = None
        modnoshow_traj = None
        mod_traj = None

        for traj_name in trajectories.keys():
            traj_type = classify_trajectory_type(traj_name)
            if traj_type == 'basetraj':
                basetraj_traj = traj_name
            elif traj_type == 'modifiednoshowobstacles':
                modnoshow_traj = traj_name
            elif traj_type == 'modified':
                mod_traj = traj_name

        if basetraj_traj and modnoshow_traj and mod_traj:
            # Find users who rated ALL THREE trajectories
            users_base = set(trajectories[basetraj_traj].keys())
            users_modnoshow = set(trajectories[modnoshow_traj].keys())
            users_mod = set(trajectories[mod_traj].keys())
            common_users = users_base & users_modnoshow & users_mod

            for user in common_users:
                # Extract safety scores
                base_answers = trajectories[basetraj_traj][user]
                modnoshow_answers = trajectories[modnoshow_traj][user]
                mod_answers = trajectories[mod_traj][user]

                base_score = extract_score(base_answers, 'safety')
                modnoshow_score = extract_score(modnoshow_answers, 'safety')
                mod_score = extract_score(mod_answers, 'safety')

                if (base_score is not None and
                    modnoshow_score is not None and
                    mod_score is not None):
                    base_values.append(base_score)
                    modnoshow_values.append(modnoshow_score)
                    mod_values.append(mod_score)
                    metadata.append((user, scenario))

    return (np.array(base_values), np.array(modnoshow_values),
            np.array(mod_values), metadata)


# ============================================================================
# Section 3: Assumption Checking Functions
# ============================================================================

def check_normality(differences):
    """
    Test normality of paired differences using Shapiro-Wilk test.

    Args:
        differences: np.array of paired differences

    Returns:
        dict: {'statistic': float, 'p_value': float, 'is_normal': bool}
    """
    if len(differences) < 3:
        return {'statistic': None, 'p_value': None, 'is_normal': False}

    statistic, p_value = shapiro(differences)
    is_normal = p_value > 0.05  # Conservative threshold

    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': is_normal
    }


def check_symmetry(differences):
    """
    Check symmetry of differences distribution.

    Args:
        differences: np.array of paired differences

    Returns:
        dict: {'skewness': float, 'is_symmetric': bool}
    """
    from scipy.stats import skew

    if len(differences) < 3:
        return {'skewness': None, 'is_symmetric': False}

    skewness = skew(differences)
    # Consider symmetric if |skewness| < 1 (moderate threshold)
    is_symmetric = abs(skewness) < 1.0

    return {
        'skewness': skewness,
        'is_symmetric': is_symmetric
    }


def verify_pairing_integrity(metadata):
    """
    Verify pairing integrity of metadata.

    Args:
        metadata: list of (user, scenario) tuples

    Returns:
        dict: {'valid': bool, 'n_pairs': int, 'issues': list}
    """
    issues = []
    n_pairs = len(metadata)

    # Check for duplicates
    unique_pairs = set(metadata)
    if len(unique_pairs) != n_pairs:
        n_duplicates = n_pairs - len(unique_pairs)
        issues.append(f"Found {n_duplicates} duplicate user-scenario pairs")

    # Check for empty metadata
    if n_pairs == 0:
        issues.append("No paired observations found")

    valid = len(issues) == 0

    return {
        'valid': valid,
        'n_pairs': n_pairs,
        'issues': issues
    }


# ============================================================================
# Section 4: Statistical Test Functions
# ============================================================================

def wilcoxon_signed_rank_test(group1, group2, alternative='greater'):
    """
    Perform Wilcoxon signed-rank test with effect size.

    Args:
        group1: np.array of first group scores
        group2: np.array of second group scores
        alternative: 'greater', 'less', or 'two-sided'

    Returns:
        dict with test results including effect size
    """
    n = len(group1)
    differences = group1 - group2

    # Count ties and direction
    n_positive = np.sum(differences > 0)
    n_negative = np.sum(differences < 0)
    n_ties = np.sum(differences == 0)

    # Perform test
    try:
        statistic, p_value = wilcoxon(
            group1,
            group2,
            alternative=alternative,
            zero_method='wilcox',  # Handle zeros
            correction=False  # No continuity correction for small n
        )
    except ValueError as e:
        # Handle case where all differences are zero
        return {
            'statistic': None,
            'p_value': 1.0,
            'effect_size': 0.0,
            'n': n,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'n_ties': n_ties,
            'error': str(e)
        }

    # Calculate rank-biserial effect size
    effect_size = rank_biserial_correlation(group1, group2)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'n': n,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_ties': n_ties
    }


def friedman_test_with_posthoc(groups, alpha_posthoc):
    """
    Perform Friedman test with post-hoc pairwise comparisons.

    Args:
        groups: list of 3 np.arrays [group1, group2, group3]
        alpha_posthoc: Bonferroni-adjusted alpha for post-hoc tests

    Returns:
        dict with Friedman results and post-hoc comparisons
    """
    group1, group2, group3 = groups
    n = len(group1)

    # Friedman test
    statistic, p_value = friedmanchisquare(group1, group2, group3)

    # Calculate Kendall's W effect size
    k = 3  # number of groups
    W = kendalls_w(statistic, n, k)

    # Post-hoc pairwise comparisons (one-tailed for directional hypotheses)
    comparisons = [
        ('modified', 'basetraj', group3, group1, 'greater'),
        ('modified', 'modifiednoshowobstacles', group3, group2, 'greater'),
        ('basetraj', 'modifiednoshowobstacles', group1, group2, 'two-sided')
    ]

    posthoc_results = []
    for name1, name2, grp1, grp2, alt in comparisons:
        test_result = wilcoxon_signed_rank_test(grp1, grp2, alternative=alt)
        p_adjusted = test_result['p_value'] * 3  # Bonferroni correction

        posthoc_results.append({
            'comparison': f'{name1} vs {name2}',
            'alternative': alt,
            'statistic': test_result['statistic'],
            'p_value': test_result['p_value'],
            'p_adjusted': p_adjusted,
            'effect_size': test_result['effect_size'],
            'significant': p_adjusted < alpha_posthoc
        })

    return {
        'friedman_statistic': statistic,
        'friedman_p': p_value,
        'effect_size': W,
        'n': n,
        'k': k,
        'posthoc_results': posthoc_results
    }


# ============================================================================
# Section 5: Effect Size Functions
# ============================================================================

def rank_biserial_correlation(group1, group2):
    """
    Calculate rank-biserial correlation effect size for Wilcoxon test.

    Formula: r = 1 - (2W)/(n(n+1))
    where W is the sum of positive ranks

    Args:
        group1: np.array of first group scores
        group2: np.array of second group scores

    Returns:
        float: rank-biserial correlation (-1 to 1)
    """
    differences = group1 - group2
    differences_nonzero = differences[differences != 0]

    if len(differences_nonzero) == 0:
        return 0.0

    n_nonzero = len(differences_nonzero)
    ranks = rankdata(np.abs(differences_nonzero))
    pos_ranks = ranks[differences_nonzero > 0]
    W_plus = np.sum(pos_ranks)

    r = 1 - (2 * W_plus) / (n_nonzero * (n_nonzero + 1))

    return r


def kendalls_w(chi_square, n, k):
    """
    Calculate Kendall's W (coefficient of concordance).

    Formula: W = χ²/(n(k-1))

    Args:
        chi_square: Friedman test statistic
        n: number of subjects
        k: number of conditions

    Returns:
        float: Kendall's W (0 to 1)
    """
    W = chi_square / (n * (k - 1))
    return W


def interpret_effect_size_rb(r):
    """
    Interpret rank-biserial correlation effect size.

    Args:
        r: rank-biserial correlation

    Returns:
        str: verbal interpretation
    """
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "small"
    elif abs_r < 0.5:
        return "medium"
    else:
        return "large"


def interpret_effect_size_w(w):
    """
    Interpret Kendall's W effect size.

    Args:
        w: Kendall's W

    Returns:
        str: verbal interpretation
    """
    if w < 0.1:
        return "very weak"
    elif w < 0.3:
        return "weak"
    elif w < 0.5:
        return "moderate"
    elif w < 0.7:
        return "strong"
    else:
        return "very strong"


# ============================================================================
# Section 6: Confidence Interval Functions
# ============================================================================

def bootstrap_median_ci(differences, n_boot=10000, ci=0.95):
    """
    Calculate bootstrap confidence interval for median difference.

    Args:
        differences: np.array of paired differences
        n_boot: number of bootstrap iterations
        ci: confidence level (default 0.95)

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if len(differences) == 0:
        return (np.nan, np.nan)

    medians = []
    rng = np.random.RandomState(42)  # For reproducibility

    for _ in range(n_boot):
        sample = rng.choice(differences, size=len(differences), replace=True)
        medians.append(np.median(sample))

    alpha = 1 - ci
    lower = np.percentile(medians, 100 * alpha / 2)
    upper = np.percentile(medians, 100 * (1 - alpha / 2))

    return (lower, upper)


# ============================================================================
# Section 7: Descriptive Statistics Functions
# ============================================================================

def compute_descriptives(data):
    """
    Compute descriptive statistics for a dataset.

    Args:
        data: np.array of numerical data

    Returns:
        dict with mean, sd, median, quartiles, range, n
    """
    if len(data) == 0:
        return {
            'mean': np.nan, 'sd': np.nan,
            'median': np.nan, 'q1': np.nan, 'q3': np.nan,
            'min': np.nan, 'max': np.nan, 'n': 0
        }

    return {
        'mean': np.mean(data),
        'sd': np.std(data, ddof=1),  # Sample standard deviation
        'median': np.median(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'min': np.min(data),
        'max': np.max(data),
        'n': len(data)
    }


def compute_difference_descriptives(diff):
    """
    Compute descriptive statistics for paired differences.

    Args:
        diff: np.array of differences

    Returns:
        dict with descriptives plus positive/negative/tie counts
    """
    desc = compute_descriptives(diff)

    desc['n_positive'] = np.sum(diff > 0)
    desc['n_negative'] = np.sum(diff < 0)
    desc['n_ties'] = np.sum(diff == 0)
    desc['pct_positive'] = 100 * desc['n_positive'] / len(diff) if len(diff) > 0 else 0
    desc['pct_negative'] = 100 * desc['n_negative'] / len(diff) if len(diff) > 0 else 0
    desc['pct_ties'] = 100 * desc['n_ties'] / len(diff) if len(diff) > 0 else 0

    return desc


# ============================================================================
# Section 8: Reporting Functions
# ============================================================================

def print_section_header(title, level=1):
    """Print formatted section header."""
    if level == 1:
        separator = "=" * 80
        print(f"\n{separator}")
        print(title.upper())
        print(separator)
    elif level == 2:
        separator = "-" * 80
        print(f"\n{separator}")
        print(title)
        print(separator)
    else:
        print(f"\n{title}")


def print_descriptive_statistics(desc, name):
    """Pretty print descriptive statistics."""
    print(f"\n  {name}:")
    print(f"    Mean ± SD: {desc['mean']:.2f} ± {desc['sd']:.2f}")
    print(f"    Median [IQR]: {desc['median']:.1f} [{desc['q1']:.1f}, {desc['q3']:.1f}]")
    print(f"    Range: [{desc['min']:.0f}, {desc['max']:.0f}]")
    if 'n' in desc:
        print(f"    N: {desc['n']}")


def print_assumption_checks(checks):
    """Report assumption test results."""
    print("\nASSUMPTION CHECKS:")

    for check_name, check_result in checks.items():
        if check_name == 'pairing':
            symbol = "✓" if check_result['valid'] else "✗"
            print(f"  {symbol} Pairing integrity: {check_result['n_pairs']} pairs")
            if check_result['issues']:
                for issue in check_result['issues']:
                    print(f"      ! {issue}")

        elif check_name == 'normality':
            if check_result['p_value'] is not None:
                result_str = "Normal" if check_result['is_normal'] else "Non-normal"
                print(f"  ! Normality of differences (Shapiro-Wilk): W={check_result['statistic']:.3f}, p={format_p_value(check_result['p_value'])} ({result_str})")
                if not check_result['is_normal']:
                    print(f"      → Non-parametric test justified")
            else:
                print(f"  ! Normality: Insufficient data")

        elif check_name == 'symmetry':
            if check_result['skewness'] is not None:
                symbol = "~" if check_result['is_symmetric'] else "!"
                result_str = "Symmetric" if check_result['is_symmetric'] else "Skewed"
                print(f"  {symbol} Symmetry of differences: skewness={check_result['skewness']:.2f} ({result_str})")
            else:
                print(f"  ! Symmetry: Insufficient data")


def print_test_results(results, hypothesis_num, comparison_desc, alpha):
    """Print statistical test results."""
    print(f"\nSTATISTICAL TEST: Wilcoxon Signed-Rank Test")
    print(f"  Test type: One-tailed (alternative: 'greater')")
    print(f"  Test statistic (W): {results['statistic']:.1f}" if results['statistic'] is not None else "  Test statistic: N/A")
    print(f"  p-value: {format_p_value(results['p_value'])}")
    print(f"  Adjusted p-value (Bonferroni): {format_p_value(results['p_value'] * N_HYPOTHESES)}")
    print(f"  Critical alpha: {alpha:.4f}")

    print(f"\nEFFECT SIZE:")
    print(f"  Rank-biserial correlation: r = {results['effect_size']:.3f}")
    print(f"  Interpretation: {interpret_effect_size_rb(results['effect_size']).capitalize()} effect")


def create_summary_table(h1_results, h2_results, h3_results):
    """Create final summary table."""
    print_section_header("SUMMARY TABLE", level=1)

    print("\n{:<12} {:<28} {:<10} {:<4} {:<8} {:<8} {:<10} {:<8}".format(
        "Hypothesis", "Comparison", "Test", "n", "p", "Adj.p", "Effect", "Result"))
    print("-" * 100)

    # H1
    decision1 = "✓" if h1_results['p_value'] * N_HYPOTHESES < ALPHA_ADJUSTED else "✗"
    print("{:<12} {:<28} {:<10} {:<4} {:<8} {:<8} {:<10} {:<8}".format(
        "H1", "base > modnoshow", "Wilcoxon", h1_results['n'],
        format_p_value(h1_results['p_value']),
        format_p_value(h1_results['p_value'] * N_HYPOTHESES),
        f"r={h1_results['effect_size']:.2f}",
        decision1))
    print("{:<12} {:<28}".format("", "(alignment)"))

    # H2
    decision2 = "✓" if h2_results['p_value'] * N_HYPOTHESES < ALPHA_ADJUSTED else "✗"
    print("{:<12} {:<28} {:<10} {:<4} {:<8} {:<8} {:<10} {:<8}".format(
        "H2", "mod > modnoshow", "Wilcoxon", h2_results['n'],
        format_p_value(h2_results['p_value']),
        format_p_value(h2_results['p_value'] * N_HYPOTHESES),
        f"r={h2_results['effect_size']:.2f}",
        decision2))
    print("{:<12} {:<28}".format("", "(alignment)"))

    # H3
    decision3 = "✓" if h3_results['friedman_p'] * N_HYPOTHESES < ALPHA_ADJUSTED else "✗"
    print("{:<12} {:<28} {:<10} {:<4} {:<8} {:<8} {:<10} {:<8}".format(
        "H3", "mod > {base,modnoshow}", "Friedman", h3_results['n'],
        format_p_value(h3_results['friedman_p']),
        format_p_value(h3_results['friedman_p'] * N_HYPOTHESES),
        f"W={h3_results['effect_size']:.2f}",
        decision3))
    print("{:<12} {:<28}".format("", "(safety)"))

    print("\nLegend: ✓ = Hypothesis supported (p < {:.4f}), ✗ = Not supported".format(ALPHA_ADJUSTED))


# ============================================================================
# Section 9: Hypothesis Analysis Pipelines
# ============================================================================

def analyze_h1(question_answers):
    """
    Complete H1 analysis: basetraj > modifiednoshowobstacles (alignment).

    Returns:
        dict with all results
    """
    print_section_header("HYPOTHESIS 1: Shorter paths increase goal alignment", level=1)

    print("\nResearch Question: Are trajectories perceived as more goal-aligned when the")
    print("robot takes a shorter path to the goal?")
    print("\nComparison: basetraj vs modifiednoshowobstacles (alignment scores)")
    print("Alternative hypothesis: basetraj > modifiednoshowobstacles")

    # Extract data
    base_align, modnoshow_align, metadata = extract_paired_data_for_h1(question_answers)

    # Data summary
    print("\nDATA SUMMARY:")
    print(f"  Sample size: n = {len(base_align)} paired observations")

    # Verify pairing
    pairing_check = verify_pairing_integrity(metadata)

    # Descriptive statistics
    print("\nDESCRIPTIVE STATISTICS:")
    base_desc = compute_descriptives(base_align)
    modnoshow_desc = compute_descriptives(modnoshow_align)
    print_descriptive_statistics(base_desc, "basetraj alignment")
    print_descriptive_statistics(modnoshow_desc, "modifiednoshowobstacles alignment")

    # Difference statistics
    differences = base_align - modnoshow_align
    diff_desc = compute_difference_descriptives(differences)
    print(f"\n  Paired differences (base - modnoshow):")
    print(f"    Mean ± SD: {diff_desc['mean']:.2f} ± {diff_desc['sd']:.2f}")
    print(f"    Median [IQR]: {diff_desc['median']:.1f} [{diff_desc['q1']:.1f}, {diff_desc['q3']:.1f}]")
    print(f"    Range: [{diff_desc['min']:.0f}, {diff_desc['max']:.0f}]")
    print(f"    Positive differences: {diff_desc['n_positive']}/{diff_desc['n']} ({diff_desc['pct_positive']:.0f}%)")
    print(f"    Negative differences: {diff_desc['n_negative']}/{diff_desc['n']} ({diff_desc['pct_negative']:.0f}%)")
    print(f"    Ties: {diff_desc['n_ties']}/{diff_desc['n']} ({diff_desc['pct_ties']:.0f}%)")

    # Assumption checks
    normality_check = check_normality(differences)
    symmetry_check = check_symmetry(differences)

    checks = {
        'pairing': pairing_check,
        'normality': normality_check,
        'symmetry': symmetry_check
    }
    print_assumption_checks(checks)

    # Statistical test
    test_results = wilcoxon_signed_rank_test(base_align, modnoshow_align, alternative='greater')
    print_test_results(test_results, 1, "basetraj > modifiednoshowobstacles", ALPHA_ADJUSTED)

    # Confidence interval for median difference
    ci_lower, ci_upper = bootstrap_median_ci(differences, N_BOOTSTRAP)
    print(f"  95% CI for median difference: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Power analysis
    power = compute_post_hoc_power(test_results['effect_size'], test_results['n'],
                                    ALPHA_ADJUSTED, 'larger')
    print(f"\nPOWER ANALYSIS:")
    print(f"  Achieved power: {power:.2f}")
    if power < 0.8:
        print(f"  ! WARNING: Low power. Non-significant results may be due to insufficient sample size.")

    # Decision
    p_adjusted = test_results['p_value'] * N_HYPOTHESES
    decision = decision_string(p_adjusted, ALPHA_ADJUSTED)
    print(f"\nDECISION:")
    print(f"  {decision} null hypothesis at alpha = {ALPHA_ADJUSTED:.4f}")

    # Interpretation
    print(f"\nINTERPRETATION:")
    if p_adjusted < ALPHA_ADJUSTED:
        print(f"  There is statistically significant evidence (adjusted p = {format_p_value(p_adjusted)})")
        print(f"  that users perceive trajectories as more goal-aligned when the robot takes")
        print(f"  a shorter path (basetraj) compared to when it deviates without showing")
        print(f"  obstacles (modifiednoshowobstacles). The effect size is {interpret_effect_size_rb(test_results['effect_size'])}.")
    else:
        print(f"  The data do not provide sufficient evidence to conclude that shorter paths")
        print(f"  result in higher perceived goal alignment at the adjusted significance level")
        print(f"  (adjusted p = {format_p_value(p_adjusted)} > {ALPHA_ADJUSTED:.4f}).")

    # Diagnostic plots
    plot_difference_histogram(differences, "H1: Differences (basetraj - modifiednoshowobstacles)")
    plot_qq_normal(differences, "H1: Q-Q Plot for Normality")

    return test_results


def analyze_h2(question_answers):
    """
    Complete H2 analysis: modified > modifiednoshowobstacles (alignment).

    Returns:
        dict with all results
    """
    print_section_header("HYPOTHESIS 2: Obstacle visibility increases perceived goal alignment", level=1)

    print("\nResearch Question: Are trajectory deviations perceived as more goal-aligned")
    print("when obstacles are visible than when they are hidden?")
    print("\nComparison: modified vs modifiednoshowobstacles (alignment scores)")
    print("Alternative hypothesis: modified > modifiednoshowobstacles")

    # Extract data
    mod_align, modnoshow_align, metadata = extract_paired_data_for_h2(question_answers)

    # Data summary
    print("\nDATA SUMMARY:")
    print(f"  Sample size: n = {len(mod_align)} paired observations")

    # Verify pairing
    pairing_check = verify_pairing_integrity(metadata)

    # Descriptive statistics
    print("\nDESCRIPTIVE STATISTICS:")
    mod_desc = compute_descriptives(mod_align)
    modnoshow_desc = compute_descriptives(modnoshow_align)
    print_descriptive_statistics(mod_desc, "modified alignment")
    print_descriptive_statistics(modnoshow_desc, "modifiednoshowobstacles alignment")

    # Difference statistics
    differences = mod_align - modnoshow_align
    diff_desc = compute_difference_descriptives(differences)
    print(f"\n  Paired differences (modified - modnoshow):")
    print(f"    Mean ± SD: {diff_desc['mean']:.2f} ± {diff_desc['sd']:.2f}")
    print(f"    Median [IQR]: {diff_desc['median']:.1f} [{diff_desc['q1']:.1f}, {diff_desc['q3']:.1f}]")
    print(f"    Range: [{diff_desc['min']:.0f}, {diff_desc['max']:.0f}]")
    print(f"    Positive differences: {diff_desc['n_positive']}/{diff_desc['n']} ({diff_desc['pct_positive']:.0f}%)")
    print(f"    Negative differences: {diff_desc['n_negative']}/{diff_desc['n']} ({diff_desc['pct_negative']:.0f}%)")
    print(f"    Ties: {diff_desc['n_ties']}/{diff_desc['n']} ({diff_desc['pct_ties']:.0f}%)")

    # Assumption checks
    normality_check = check_normality(differences)
    symmetry_check = check_symmetry(differences)

    checks = {
        'pairing': pairing_check,
        'normality': normality_check,
        'symmetry': symmetry_check
    }
    print_assumption_checks(checks)

    # Statistical test
    test_results = wilcoxon_signed_rank_test(mod_align, modnoshow_align, alternative='greater')
    print_test_results(test_results, 2, "modified > modifiednoshowobstacles", ALPHA_ADJUSTED)

    # Confidence interval for median difference
    ci_lower, ci_upper = bootstrap_median_ci(differences, N_BOOTSTRAP)
    print(f"  95% CI for median difference: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Power analysis
    power = compute_post_hoc_power(test_results['effect_size'], test_results['n'],
                                    ALPHA_ADJUSTED, 'larger')
    print(f"\nPOWER ANALYSIS:")
    print(f"  Achieved power: {power:.2f}")
    if power < 0.8:
        print(f"  ! WARNING: Low power. Non-significant results may be due to insufficient sample size.")

    # Decision
    p_adjusted = test_results['p_value'] * N_HYPOTHESES
    decision = decision_string(p_adjusted, ALPHA_ADJUSTED)
    print(f"\nDECISION:")
    print(f"  {decision} null hypothesis at alpha = {ALPHA_ADJUSTED:.4f}")

    # Interpretation
    print(f"\nINTERPRETATION:")
    if p_adjusted < ALPHA_ADJUSTED:
        print(f"  There is statistically significant evidence (adjusted p = {format_p_value(p_adjusted)})")
        print(f"  that trajectory deviations are perceived as more goal-aligned when obstacles")
        print(f"  are visible (modified) compared to when they are hidden (modifiednoshowobstacles).")
        print(f"  The effect size is {interpret_effect_size_rb(test_results['effect_size'])}.")
    else:
        print(f"  The data do not provide sufficient evidence to conclude that obstacle visibility")
        print(f"  increases perceived goal alignment at the adjusted significance level")
        print(f"  (adjusted p = {format_p_value(p_adjusted)} > {ALPHA_ADJUSTED:.4f}).")

    # Diagnostic plots
    plot_difference_histogram(differences, "H2: Differences (modified - modifiednoshowobstacles)")
    plot_qq_normal(differences, "H2: Q-Q Plot for Normality")

    return test_results


def analyze_h3(question_answers):
    """
    Complete H3 analysis: modified > {basetraj, modifiednoshowobstacles} (safety).

    Returns:
        dict with all results
    """
    print_section_header("HYPOTHESIS 3: Obstacle visibility increases perceived safety", level=1)

    print("\nResearch Question: Are trajectories perceived as safer when the user knows")
    print("why the robot deviated from the intended path?")
    print("\nComparison: modified vs {basetraj, modifiednoshowobstacles} (safety scores)")
    print("Alternative hypothesis: modified has higher safety ratings")

    # Extract data
    base_safety, modnoshow_safety, mod_safety, metadata = extract_triplet_data_for_h3(question_answers)

    # Data summary
    print("\nDATA SUMMARY:")
    print(f"  Sample size: n = {len(base_safety)} complete triplets")

    # Verify pairing
    pairing_check = verify_pairing_integrity(metadata)

    # Descriptive statistics
    print("\nDESCRIPTIVE STATISTICS:")
    base_desc = compute_descriptives(base_safety)
    modnoshow_desc = compute_descriptives(modnoshow_safety)
    mod_desc = compute_descriptives(mod_safety)
    print_descriptive_statistics(base_desc, "basetraj safety")
    print_descriptive_statistics(modnoshow_desc, "modifiednoshowobstacles safety")
    print_descriptive_statistics(mod_desc, "modified safety")

    # Assumption checks
    checks = {'pairing': pairing_check}
    print_assumption_checks(checks)

    # Friedman test with post-hoc
    groups = [base_safety, modnoshow_safety, mod_safety]
    alpha_posthoc = ALPHA_ADJUSTED / 3  # Bonferroni for 3 comparisons
    friedman_results = friedman_test_with_posthoc(groups, alpha_posthoc)

    print(f"\nSTATISTICAL TEST: Friedman Test")
    print(f"  Test type: Two-tailed (omnibus test)")
    print(f"  Test statistic (χ²): {friedman_results['friedman_statistic']:.2f}")
    print(f"  Degrees of freedom: {friedman_results['k'] - 1}")
    print(f"  p-value: {format_p_value(friedman_results['friedman_p'])}")
    print(f"  Adjusted p-value (Bonferroni): {format_p_value(friedman_results['friedman_p'] * N_HYPOTHESES)}")
    print(f"  Critical alpha: {ALPHA_ADJUSTED:.4f}")

    print(f"\nEFFECT SIZE:")
    print(f"  Kendall's W: {friedman_results['effect_size']:.3f}")
    print(f"  Interpretation: {interpret_effect_size_w(friedman_results['effect_size']).capitalize()} agreement")

    # Post-hoc comparisons
    print(f"\nPOST-HOC PAIRWISE COMPARISONS:")
    print(f"  Using Wilcoxon signed-rank tests with Bonferroni correction")
    print(f"  Adjusted alpha for 3 comparisons: {ALPHA_ADJUSTED:.4f}/3 = {alpha_posthoc:.4f}")

    for posthoc in friedman_results['posthoc_results']:
        print(f"\n  Comparison: {posthoc['comparison']}")
        print(f"    Alternative: {posthoc['alternative']}")
        print(f"    W = {posthoc['statistic']:.1f}, p = {format_p_value(posthoc['p_value'])}, adjusted p = {format_p_value(posthoc['p_adjusted'])}")
        print(f"    Effect size (r) = {posthoc['effect_size']:.3f}")
        result_str = "Significant" if posthoc['significant'] else "Not significant"
        print(f"    Result: {result_str}")

    # Decision
    p_adjusted = friedman_results['friedman_p'] * N_HYPOTHESES
    decision = decision_string(p_adjusted, ALPHA_ADJUSTED)
    print(f"\nDECISION:")
    print(f"  {decision} null hypothesis at alpha = {ALPHA_ADJUSTED:.4f}")

    # Interpretation
    print(f"\nINTERPRETATION:")
    if p_adjusted < ALPHA_ADJUSTED:
        print(f"  There is statistically significant evidence (adjusted p = {format_p_value(p_adjusted)})")
        print(f"  that perceived safety differs across the three trajectory types.")

        # Check post-hoc results
        mod_vs_base = friedman_results['posthoc_results'][0]
        mod_vs_modnoshow = friedman_results['posthoc_results'][1]

        if mod_vs_base['significant'] or mod_vs_modnoshow['significant']:
            print(f"\n  Post-hoc analyses reveal:")
            if mod_vs_base['significant']:
                print(f"    - Modified trajectories are perceived as significantly safer than base trajectories")
            if mod_vs_modnoshow['significant']:
                print(f"    - Modified trajectories are perceived as significantly safer than modified-no-show-obstacles")
        else:
            print(f"\n  However, post-hoc pairwise comparisons did not reach significance after")
            print(f"  Bonferroni correction, suggesting the omnibus effect may be weak.")
    else:
        print(f"  The data do not provide sufficient evidence to conclude that obstacle visibility")
        print(f"  affects perceived safety at the adjusted significance level")
        print(f"  (adjusted p = {format_p_value(p_adjusted)} > {ALPHA_ADJUSTED:.4f}).")

    return friedman_results


# ============================================================================
# Section 10: Visualization Functions
# ============================================================================

def plot_difference_histogram(differences, title):
    """Plot histogram of paired differences."""
    plt.figure(figsize=(8, 5))
    plt.hist(differences, bins=np.arange(differences.min()-0.5, differences.max()+1.5, 1),
             edgecolor='black', alpha=0.7)
    plt.axvline(np.median(differences), color='red', linestyle='--', linewidth=2,
                label=f'Median = {np.median(differences):.1f}')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_qq_normal(differences, title):
    """Plot Q-Q plot for normality assessment."""
    plt.figure(figsize=(8, 5))
    probplot(differences, dist="norm", plot=plt)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Section 11: Power Analysis Functions
# ============================================================================

def compute_post_hoc_power(effect_size, n, alpha, alternative):
    """
    Calculate achieved statistical power.

    Args:
        effect_size: observed effect size (rank-biserial r)
        n: sample size
        alpha: significance level
        alternative: 'larger', 'smaller', or 'two-sided'

    Returns:
        float: achieved power (0-1)
    """
    # Use t-test power as approximation for Wilcoxon
    # Rank-biserial correlation approximates Cohen's d for moderate effects

    try:
        # Calculate non-centrality parameter
        ncp = abs(effect_size) * np.sqrt(n)

        # Determine critical t-value
        if alternative == 'larger' or alternative == 'smaller':
            # One-tailed test
            t_crit = t.ppf(1 - alpha, n - 1)
        else:
            # Two-tailed test
            t_crit = t.ppf(1 - alpha/2, n - 1)

        # Calculate power using non-central t distribution
        from scipy.stats import nct

        if alternative == 'larger':
            power = 1 - nct.cdf(t_crit, n - 1, ncp)
        elif alternative == 'smaller':
            power = nct.cdf(-t_crit, n - 1, ncp)
        else:  # two-sided
            power = 1 - nct.cdf(t_crit, n - 1, ncp) + nct.cdf(-t_crit, n - 1, ncp)

        return power
    except:
        return np.nan


# ============================================================================
# Section 12: Utility Functions
# ============================================================================

def format_p_value(p):
    """Format p-value for display."""
    if p is None or np.isnan(p):
        return "N/A"
    elif p < 0.001:
        return "< 0.001"
    else:
        return f"{p:.3f}"


def decision_string(p, alpha):
    """Return decision string."""
    if p < alpha:
        return "✓ REJECT"
    else:
        return "✗ FAIL TO REJECT"


# ============================================================================
# Section 13: Main Execution
# ============================================================================

def main():
    """Main execution function."""
    # Print header
    print("=" * 80)
    print("STATISTICAL ANALYSIS: USER STUDY HYPOTHESES")
    print("=" * 80)

    # Exclude specific users (case-insensitive)
    excluded_users = ["test"]

    print("\nANALYSIS PARAMETERS:")
    print(f"  Significance level (alpha): {ALPHA}")
    print(f"  Number of hypotheses: {N_HYPOTHESES}")
    print(f"  Multiple comparison correction: Bonferroni")
    print(f"  Adjusted alpha per test: {ALPHA_ADJUSTED:.4f}")
    print(f"  Statistical approach: Non-parametric (ordinal data, small sample)")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP:,}")
    if excluded_users:
        print(f"  Excluded users containing: {', '.join(excluded_users)}")

    # Load data
    zarr_path = "output/obstacles_on_path_10_userstudy.zarr"
    print(f"\nLoading data from: {zarr_path}")
    question_answers = load_question_answers(zarr_path, excluded_users=excluded_users)
    print(f"Data loaded successfully.")

    # Analyze each hypothesis
    h1_results = analyze_h1(question_answers)
    h2_results = analyze_h2(question_answers)
    h3_results = analyze_h3(question_answers)

    # Summary table
    create_summary_table(h1_results, h2_results, h3_results)

    # Limitations
    print_section_header("LIMITATIONS AND CAVEATS", level=1)
    print("\n1. Sample Size: Small sample (n=12-13) limits statistical power. Non-significant")
    print("   results may reflect insufficient power rather than true null effects.")
    print("\n2. Unbalanced Users: Data heavily weighted toward one user. Results may not")
    print("   generalize equally across users.")
    print("\n3. Ordinal Data: Likert scales have limited precision. Equal intervals between")
    print("   scale points should not be assumed.")
    print("\n4. Multiple Comparisons: Bonferroni correction is conservative and may miss")
    print("   true effects. Consider exploratory analysis without correction.")
    print("\n5. Missing Data: One incomplete observation set excluded. Appears to be MCAR")
    print("   (missing completely at random), minimal impact expected.")
    print("\n6. Effect Sizes: With small samples, effect size estimates have wide confidence")
    print("   intervals. Interpret with caution.")

    # Recommendations
    print_section_header("RECOMMENDATIONS FOR REPORTING", level=1)
    print("\nWhen presenting results in a paper:")
    print("  1. Report exact p-values (not just 'p < 0.05')")
    print("  2. Report effect sizes with confidence intervals")
    print("  3. Include descriptive statistics (means, medians, SDs, ranges)")
    print("  4. Justify choice of non-parametric tests")
    print("  5. Acknowledge limitations (especially small sample size)")
    print("  6. Consider reporting both adjusted and unadjusted p-values")
    print("  7. Discuss practical significance alongside statistical significance")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
