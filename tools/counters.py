"""
Helper functions to count trials and trial sites over groups.

Note that some design and default argument choices are specific for
our analyses.
"""

import pandas as pd


def sum_over_groupby(data, sum_column_name, groupby_column_names):
    """Sum up a column for a group.

    Note: there is no duplicate removal in this function! This is because
    you could have two rows like ['study 1', 'USA', 42] and ['study 2',
    'USA', 42]; then if you group by country and sum up, you would of course
    want to keep both studies. However, this also means that if you have
    exact duplicates in your dataframe (like ['study 1', 'USA', 42],
    ['study 1', 'USA', 42]), you will get double counting errors!

    Args:
        data: a pandas dataframe
        sum_column_name: the column with the values to sum over
        groupby_column_names: the columns to group the sum by

    Returns:
        a dataframe with groupy_column_names and the sum over the sum column name
    """
    return (
        data[[sum_column_name] + groupby_column_names]
        .groupby(groupby_column_names)
        .sum()
        .reset_index()
    )


def count_trials(trial_data, groupby_column_names, trial_id_column_name="nct_number"):
    """Count the number of trials for a group by counting the number of
    unique trial IDs.

    Args:
        trial_data: a pandas dataframe with at least a trial ID column and a groupby column
        groupby_column_names: the columns to group the count by
        trial_id_column_name: the column with the trial ID

    Returns:
        trial_data: a dataframe with groupy_column_names and the number of trials per group
    """
    return (
        trial_data[[trial_id_column_name] + groupby_column_names]
        .drop_duplicates()
        .groupby(groupby_column_names)
        .count()
        .reset_index()
        .rename(columns={"nct_number": "n_trials"})
    )
