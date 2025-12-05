#!/usr/bin/env python3
"""
Script to load and analyze question_answers from a user study zarr file.
"""

import zarr
import json
from collections import defaultdict
import re


def load_question_answers(zarr_path, excluded_users=None):
    """
    Load question_answers from the zarr file structure.

    Args:
        zarr_path: Path to the .zarr folder
        excluded_users: List of user names to exclude (case-insensitive).
                       Users whose names contain any of these strings will be excluded.

    Returns:
        dict: Nested dictionary with structure:
            {scenario_name: {trajectory_name: {user_name: {question: answer}}}}
    """
    # Open the zarr file
    root = zarr.open(zarr_path, mode='r')
    scenarios_group = root['trajectories']

    all_question_answers = {}

    # Pattern to match scenario folders
    scenario_re = re.compile(r"^scenario_(\d+)$")

    # Normalize excluded users to lowercase for case-insensitive matching
    excluded_users_lower = [u.lower() for u in excluded_users] if excluded_users else []

    # Iterate through all scenario groups
    for scenario_name in sorted(scenarios_group.keys()):
        if scenario_re.match(scenario_name):
            scenario_group = scenarios_group[scenario_name]

            # Check if question_answers exists in attributes
            if 'question_answers' in scenario_group.attrs:
                question_answers = scenario_group.attrs['question_answers']

                # Filter out excluded users
                if excluded_users_lower:
                    filtered_question_answers = {}
                    for traj_name, users in question_answers.items():
                        filtered_users = {
                            user_name: answers
                            for user_name, answers in users.items()
                            if not any(excluded.lower() in user_name.lower() for excluded in excluded_users)
                        }
                        if filtered_users:  # Only add trajectory if it has users after filtering
                            filtered_question_answers[traj_name] = filtered_users
                    question_answers = filtered_question_answers

                # Store in our results dictionary
                if question_answers:  # Only add scenario if it has data after filtering
                    all_question_answers[scenario_name] = question_answers

    return all_question_answers


def print_question_answers(question_answers_dict):
    """
    Pretty print the question answers in a readable format.

    Args:
        question_answers_dict: Dictionary returned by load_question_answers()
    """
    for scenario_name, trajectories in sorted(question_answers_dict.items()):
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*80}")

        for traj_name, users in sorted(trajectories.items()):
            print(f"\n  Trajectory: {traj_name}")
            print(f"  {'-'*76}")

            for user_name, answers in sorted(users.items()):
                print(f"\n    User: {user_name}")
                for question, answer in answers.items():
                    print(f"      {question}: {answer}")


def get_answers_by_user(question_answers_dict):
    """
    Reorganize data to group by user instead of scenario.

    Args:
        question_answers_dict: Dictionary returned by load_question_answers()

    Returns:
        dict: {user_name: {scenario: {trajectory: {question: answer}}}}
    """
    by_user = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for scenario_name, trajectories in question_answers_dict.items():
        for traj_name, users in trajectories.items():
            for user_name, answers in users.items():
                by_user[user_name][scenario_name][traj_name] = answers

    return dict(by_user)


def get_answers_by_question(question_answers_dict):
    """
    Reorganize data to group by question for analysis.

    Args:
        question_answers_dict: Dictionary returned by load_question_answers()

    Returns:
        dict: {question: {scenario: {trajectory: {user: answer}}}}
    """
    by_question = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for scenario_name, trajectories in question_answers_dict.items():
        for traj_name, users in trajectories.items():
            for user_name, answers in users.items():
                for question, answer in answers.items():
                    by_question[question][scenario_name][traj_name][user_name] = answer

    return dict(by_question)


def classify_trajectory_type(traj_name):
    """
    Classify trajectory based on its name.

    Args:
        traj_name: Name of the trajectory

    Returns:
        str: One of 'basetraj', 'modified', 'modifiednoshowobstacles', 'gello_traj', or 'unknown'
    """
    if 'basetraj' in traj_name:
        return 'basetraj'
    elif 'modifiednoshowobstacles' in traj_name:
        return 'modifiednoshowobstacles'
    elif 'modified' in traj_name:
        return 'modified'
    elif 'gello' in traj_name or 'traj_' in traj_name:
        return 'gello_traj'
    else:
        return 'unknown'


def get_answers_by_trajectory_type(question_answers_dict):
    """
    Reorganize data to group by trajectory type.

    Args:
        question_answers_dict: Dictionary returned by load_question_answers()

    Returns:
        dict: {traj_type: {scenario: {trajectory: {user: {question: answer}}}}}
    """
    by_traj_type = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for scenario_name, trajectories in question_answers_dict.items():
        for traj_name, users in trajectories.items():
            traj_type = classify_trajectory_type(traj_name)
            for user_name, answers in users.items():
                by_traj_type[traj_type][scenario_name][traj_name][user_name] = answers

    return dict(by_traj_type)


def print_answers_by_trajectory_type(question_answers_dict):
    """
    Print question answers organized by trajectory type.

    Args:
        question_answers_dict: Dictionary returned by load_question_answers()
    """
    by_traj_type = get_answers_by_trajectory_type(question_answers_dict)

    for traj_type in sorted(by_traj_type.keys()):
        print(f"\n{'='*80}")
        print(f"TRAJECTORY TYPE: {traj_type}")
        print(f"{'='*80}")

        scenarios = by_traj_type[traj_type]
        for scenario_name, trajectories in sorted(scenarios.items()):
            print(f"\n  Scenario: {scenario_name}")

            for traj_name, users in sorted(trajectories.items()):
                print(f"\n    Trajectory: {traj_name}")
                print(f"    {'-'*76}")

                for user_name, answers in sorted(users.items()):
                    print(f"\n      User: {user_name}")
                    for question, answer in answers.items():
                        print(f"        {question}: {answer}")


def compute_statistics(question_answers_dict):
    """
    Compute basic statistics for numerical answers (1-5 scale questions).

    Args:
        question_answers_dict: Dictionary returned by load_question_answers()

    Returns:
        dict: Statistics for each question
    """
    import numpy as np

    by_question = get_answers_by_question(question_answers_dict)

    stats = {}
    for question, scenarios in by_question.items():
        all_answers = []
        for scenario_name, trajectories in scenarios.items():
            for traj_name, users in trajectories.items():
                for user_name, answer in users.items():
                    if isinstance(answer, (int, float)):
                        all_answers.append(answer)

        if all_answers:
            stats[question] = {
                'mean': np.mean(all_answers),
                'std': np.std(all_answers),
                'min': np.min(all_answers),
                'max': np.max(all_answers),
                'count': len(all_answers),
                'values': all_answers
            }

    return stats


def compute_statistics_by_trajectory_type(question_answers_dict):
    """
    Compute statistics for each question, grouped by trajectory type.

    Args:
        question_answers_dict: Dictionary returned by load_question_answers()

    Returns:
        dict: {traj_type: {question: {mean, std, min, max, count, values}}}
    """
    import numpy as np

    by_traj_type = get_answers_by_trajectory_type(question_answers_dict)

    stats_by_type = {}
    for traj_type, scenarios in by_traj_type.items():
        stats_by_type[traj_type] = {}

        # Collect all answers by question for this trajectory type
        by_question = defaultdict(list)
        for scenario_name, trajectories in scenarios.items():
            for traj_name, users in trajectories.items():
                for user_name, answers in users.items():
                    for question, answer in answers.items():
                        if isinstance(answer, (int, float)):
                            by_question[question].append(answer)

        # Calculate statistics for each question
        for question, all_answers in by_question.items():
            if all_answers:
                stats_by_type[traj_type][question] = {
                    'mean': np.mean(all_answers),
                    'std': np.std(all_answers),
                    'min': np.min(all_answers),
                    'max': np.max(all_answers),
                    'count': len(all_answers),
                    'values': all_answers
                }

    return stats_by_type


if __name__ == "__main__":
    zarr_path = "output/obstacles_on_path_10_userstudy.zarr"

    print(f"Loading question answers from: {zarr_path}")
    print()

    # Load the data
    question_answers = load_question_answers(zarr_path, excluded_users=["test"])

    # Print all question answers
    print_question_answers(question_answers)

    # Print answers organized by trajectory type
    print(f"\n\n{'='*80}")
    print("DATA BY TRAJECTORY TYPE")
    print(f"{'='*80}")
    print_answers_by_trajectory_type(question_answers)

    # Print overall statistics
    print(f"\n\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}\n")

    stats = compute_statistics(question_answers)
    for question, stat_dict in stats.items():
        print(f"\nQuestion: {question}")
        print(f"  Mean: {stat_dict['mean']:.2f}")
        print(f"  Std Dev: {stat_dict['std']:.2f}")
        print(f"  Range: [{stat_dict['min']}, {stat_dict['max']}]")
        print(f"  Count: {stat_dict['count']}")

    # Print statistics by trajectory type
    print(f"\n\n{'='*80}")
    print("STATISTICS BY TRAJECTORY TYPE")
    print(f"{'='*80}\n")

    stats_by_type = compute_statistics_by_trajectory_type(question_answers)
    for traj_type in sorted(stats_by_type.keys()):
        print(f"\n{'-'*80}")
        print(f"Trajectory Type: {traj_type}")
        print(f"{'-'*80}")
        for question, stat_dict in stats_by_type[traj_type].items():
            print(f"\n  Question: {question}")
            print(f"    Mean: {stat_dict['mean']:.2f}")
            print(f"    Std Dev: {stat_dict['std']:.2f}")
            print(f"    Range: [{stat_dict['min']}, {stat_dict['max']}]")
            print(f"    Count: {stat_dict['count']}")

    # Example: Get all answers by a specific user
    print(f"\n\n{'='*80}")
    print("DATA BY USER")
    print(f"{'='*80}\n")

    by_user = get_answers_by_user(question_answers)
    for user_name in sorted(by_user.keys()):
        print(f"\nUser: {user_name}")
        total_responses = sum(len(trajs) for trajs in by_user[user_name].values())
        print(f"  Total responses: {total_responses}")
